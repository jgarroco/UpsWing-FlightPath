import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

items_df = pd.read_csv("items.csv")


def three_pl(theta, a, b, c):
    e_term = np.exp(a * (theta - b))
    p = c + (1 - c) * (e_term / (1 + e_term))
    return p


def neg_log_likelihood(theta, items_asked, responses):
    ll = 0
    for _, item in items_asked.iterrows():
        u = responses[item["id"]]
        p = three_pl(theta, item["a"], item["b"], item["c"])
        p = np.clip(p, 1e-6, 1 - 1e-6)
        ll += u * np.log(p) + (1 - u) * np.log(1 - p)
    return -ll


def fisher_information(item, theta):
    a, b, c = item["a"], item["b"], item["c"]
    p = three_pl(theta, a, b, c)
    q = 1 - p
    info = (a**2) * ((q) / (p)) * ((p - c) ** 2) / ((1 - c) ** 2)
    return info


def select_next_item(items_df, asked_ids, theta):
    candidates = items_df[~items_df["id"].isin(asked_ids)].copy()
    if candidates.empty:
        return None
    candidates["info"] = candidates.apply(
        lambda item: fisher_information(item, theta), axis=1
    )
    next_item = candidates.loc[candidates["info"].idxmax()]
    return next_item


def update_theta(items_asked, responses):
    result = minimize_scalar(
        neg_log_likelihood,
        bounds=(-4, 4),
        args=(items_asked, responses),
        method="bounded",
    )
    return result.x


def run_cat_test():
    theta = 0
    asked_items = pd.DataFrame(columns=items_df.columns)
    responses = {}
    max_items = 5

    print(f"Initial ability estimate (theta): {theta:.3f}\n")
    for _ in range(max_items):
        next_item = select_next_item(items_df, asked_items["id"].tolist(), theta)
        if next_item is None:
            print("No more items to ask.")
            break

        print(next_item["question"])
        for idx, opt in enumerate(
            eval(next_item["options"]), 1
        ):  # Use eval to parse options from string
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

        theta = update_theta(asked_items, responses)
        print(f"Updated ability estimate (theta): {theta:.3f}\n")

    print(f"Test finished. Final estimated ability (theta): {theta:.3f}")


run_cat_test()
