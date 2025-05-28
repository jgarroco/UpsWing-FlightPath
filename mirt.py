import numpy as np
import pandas as pd
from scipy.optimize import minimize
import ast

items_df = pd.read_csv("./mirt_items.csv")

a_cols = [col for col in items_df.columns if col.startswith("a") and col[1:].isdigit()]
n_dimensions = len(a_cols)

EPS = 1e-3


def mirt_3pl(theta_vec, a_vec, b, c):
    dot_product = np.dot(theta_vec, a_vec)
    e_term = np.exp(dot_product - b)
    p = c + (1 - c) * (e_term / (1 + e_term))
    return p


def neg_log_likelihood(theta_vec, items_asked, responses):
    ll = 0
    for _, item in items_asked.iterrows():
        u = responses[item["id"]]
        a_vec = np.array([item[f"a{i + 1}"] for i in range(n_dimensions)])
        p = mirt_3pl(theta_vec, a_vec, item["b"], item["c"])
        p = np.clip(p, 1e-6, 1 - 1e-6)
        ll += u * np.log(p) + (1 - u) * np.log(1 - p)
    return -ll


def fisher_information_matrix(item, theta_vec):
    a_vec = np.array([item[f"a{i + 1}"] for i in range(n_dimensions)])
    p = mirt_3pl(theta_vec, a_vec, item["b"], item["c"])

    q = 1 - p

    scalar = ((p - item["c"]) ** 2 / ((1 - item["c"]) ** 2)) * (q / p)

    fim = scalar * np.outer(a_vec, a_vec)
    print(f"Fisher Information Matrix (fim):\n{fim}")

    return fim


def select_next_item(items_df, asked_ids, theta_vec, current_fim=None):
    candidates = items_df[~items_df["id"].isin(asked_ids)].copy()
    if candidates.empty:
        return None

    if current_fim is None:
        current_fim = EPS * np.eye(n_dimensions)

    def d_optimality_score(item):
        item_fim = fisher_information_matrix(item, theta_vec)
        combined_fim = current_fim + item_fim + EPS * np.eye(n_dimensions)
        try:
            det_score = np.linalg.det(combined_fim)
            print(f"Item {item['id']}: determinant = {det_score}")
            return det_score
        except np.linalg.LinAlgError:
            return 0

    candidates["info"] = candidates.apply(d_optimality_score, axis=1)
    next_item = candidates.loc[candidates["info"].idxmax()]

    return next_item


def update_theta(items_asked, responses, init_theta=None):
    if init_theta is None:
        init_theta = np.zeros(n_dimensions)

    result = minimize(
        neg_log_likelihood,
        init_theta,
        args=(items_asked, responses),
        method="L-BFGS-B",
        bounds=[(-4.0, 4.0)] * n_dimensions,
        options={"ftol": 1e-6, "maxiter": 1000},
    )
    print(f"Updated theta: {result.x}")
    print(f"Optimization success: {result.success}")

    hess_inv = (
        result.hess_inv.todense()
        if hasattr(result.hess_inv, "todense")
        else result.hess_inv
    )
    hess_inv_reg = hess_inv + EPS * np.eye(n_dimensions)

    return result.x, hess_inv_reg


def should_stop(hessian_inv, threshold=0.3):
    if isinstance(hessian_inv, np.ndarray):
        std_errors = np.sqrt(np.diag(hessian_inv))
        stop = all(err < threshold for err in std_errors)
        print(f"Standard errors: {std_errors}, Stop: {stop}")
        return stop
    return False


def run_mirt_cat_test():
    theta_vec = np.zeros(n_dimensions)
    asked_items = pd.DataFrame(columns=items_df.columns)
    responses = {}
    cumulative_fim = EPS * np.eye(n_dimensions)
    max_items = 10

    for i in range(max_items):
        print(f"\n--- Iteration {i + 1} ---")
        next_item = select_next_item(
            items_df, asked_items["id"].tolist(), theta_vec, cumulative_fim
        )
        if next_item is None:
            print("No more items to ask.")
            break

        print(f"Selected item ID: {next_item['id']}")
        print(
            f"Item parameters: a={[next_item[f'a{j + 1}'] for j in range(n_dimensions)]}, b={next_item['b']}, c={next_item['c']}"
        )
        print(f"Question: {next_item['question']}")
        options = ast.literal_eval(next_item["options"])
        for idx, opt in enumerate(options, 1):
            print(f"  {idx}. {opt}")
        while True:
            try:
                answer = int(input("Your answer (1-4): "))
                if answer in [1, 2, 3, 4]:
                    break
                else:
                    print("Please enter a number from 1 to 4.")
            except Exception as e:
                print(f"Invalid input. Please enter a number from 1 to 4. Error: {e}")

        correct = 1 if answer == next_item["correct"] else 0
        print(
            f"User answered: {answer} (Correct: {next_item['correct']}, {'Correct' if correct else 'Incorrect'})"
        )
        asked_items = pd.concat(
            [asked_items, next_item.to_frame().T], ignore_index=True
        )
        responses[next_item["id"]] = correct

        print(f"Responses so far: {responses}")
        theta_vec, hessian_inv = update_theta(asked_items, responses, theta_vec)
        print(f"Updated ability estimate (theta): {theta_vec}")
        print(f"Hessian inverse:\n{hessian_inv}")

        item_fim = fisher_information_matrix(next_item, theta_vec)
        cumulative_fim += item_fim

        if should_stop(hessian_inv):
            print("Stopping rule met (sufficient certainty in ability estimate).")
            break

    print(f"\nTest finished. Final estimated ability (theta): {theta_vec}")


if __name__ == "__main__":
    run_mirt_cat_test()
