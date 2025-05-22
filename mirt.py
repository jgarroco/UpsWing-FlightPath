import numpy as np
import pandas as pd
from scipy.optimize import minimize
import ast

items_df = pd.read_csv("./mirt_items.csv")

a_cols = [col for col in items_df.columns if col.startswith("a") and col[1:].isdigit()]
n_dimensions = len(a_cols)


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


def fisher_information(item, theta_vec):
    a_vec = np.array([item[f"a{i + 1}"] for i in range(n_dimensions)])
    p = mirt_3pl(theta_vec, a_vec, item["b"], item["c"])
    q = 1 - p
    info = np.dot(a_vec, a_vec) * (
        (q / p) * ((p - item["c"]) ** 2) / ((1 - item["c"]) ** 2)
    )
    return info


def select_next_item(items_df, asked_ids, theta_vec):
    candidates = items_df[~items_df["id"].isin(asked_ids)].copy()
    if candidates.empty:
        return None
    candidates["info"] = candidates.apply(
        lambda item: fisher_information(item, theta_vec), axis=1
    )
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
        bounds=[(-2.5, 2.5)] * n_dimensions,
    )
    return result.x, result.hess_inv.todense() if hasattr(
        result.hess_inv, "todense"
    ) else result.hess_inv


def should_stop(hessian_inv, threshold=0.3):
    if isinstance(hessian_inv, np.ndarray):
        std_errors = np.sqrt(np.diag(hessian_inv))
        return all(err < threshold for err in std_errors)
    return False


def run_mirt_cat_test():
    theta_vec = np.zeros(n_dimensions)
    asked_items = pd.DataFrame(columns=items_df.columns)
    responses = {}
    max_items = 10

    print(f"Initial ability estimate (theta): {theta_vec}\n")
    for i in range(max_items):
        next_item = select_next_item(items_df, asked_items["id"].tolist(), theta_vec)
        if next_item is None:
            print("No more items to ask.")
            break

        print("Quesition no.", i + 1, ": ", next_item["question"])
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
            except:
                print("Invalid input. Please enter a number from 1 to 4.")

        correct = 1 if answer == next_item["correct"] else 0
        asked_items = pd.concat(
            [asked_items, next_item.to_frame().T], ignore_index=True
        )
        responses[next_item["id"]] = correct

        theta_vec, hessian_inv = update_theta(asked_items, responses, theta_vec)
        print(f"Updated ability estimate (theta): {theta_vec}")

        if should_stop(hessian_inv):
            print("Stopping rule met (sufficient certainty in ability estimate).\n")
            break

    print(f"\nTest finished. Final estimated ability (theta): {theta_vec}")


if __name__ == "__main__":
    run_mirt_cat_test()
