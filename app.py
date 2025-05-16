import numpy as np
from scipy.optimize import minimize_scalar

items = [
    {
        'id': 1, 'a': 1.2, 'b': -1.0, 'c': 0.2,
        'question': "Choose the correct word: She _____ to the store yesterday.",
        'options': ["go", "goes", "went", "gone"],
        'correct': 3 
    },
    {
        'id': 2, 'a': 0.8, 'b': 0.0, 'c': 0.25,
        'question': "Which sentence is grammatically correct?",
        'options': [
            "He don't like apples.",
            "He doesn't likes apples.",
            "He doesn't like apples.",
            "He not like apples."
        ],
        'correct': 3
    },
    {
        'id': 3, 'a': 1.0, 'b': 1.0, 'c': 0.2,
        'question': "Fill in the blank: If I _____ you, I would study harder.",
        'options': ["was", "were", "am", "be"],
        'correct': 2
    },
    {
        'id': 4, 'a': 1.5, 'b': 0.5, 'c': 0.15,
        'question': "Choose the synonym for 'rapid':",
        'options': ["slow", "quick", "quiet", "rare"],
        'correct': 2
    },
    {
        'id': 5, 'a': 1.1, 'b': -0.5, 'c': 0.2,
        'question': "Which word is a noun?",
        'options': ["run", "beautiful", "happiness", "quickly"],
        'correct': 3
    },
]

def three_pl(theta, a, b, c):
    e_term = np.exp(a * (theta - b))
    p = c + (1 - c) * (e_term / (1 + e_term))
    return p

def neg_log_likelihood(theta, items_asked, responses):
    ll = 0
    for item, u in zip(items_asked, responses):
        p = three_pl(theta, item['a'], item['b'], item['c'])
        p = np.clip(p, 1e-6, 1 - 1e-6)
        ll += u * np.log(p) + (1 - u) * np.log(1 - p)
    return -ll

def fisher_information(item, theta):
    a, b, c = item['a'], item['b'], item['c']
    p = three_pl(theta, a, b, c)
    q = 1 - p
    info = (a**2) * ((q) / (p)) * ((p - c)**2) / ((1 - c)**2)
    return info

def select_next_item(items, asked_ids, theta):
    candidates = [item for item in items if item['id'] not in asked_ids]
    if not candidates:
        return None
    infos = [fisher_information(item, theta) for item in candidates]
    max_idx = np.argmax(infos)
    return candidates[max_idx]

def update_theta(items_asked, responses):
    result = minimize_scalar(neg_log_likelihood, bounds=(-4, 4), args=(items_asked, responses), method='bounded')
    return result.x

def run_cat_test():
    theta = 0
    asked_items = []
    responses = []
    max_items = 5

    for _ in range(max_items):
        next_item = select_next_item(items, [item['id'] for item in asked_items], theta)
        if next_item is None:
            print("No more items to ask.")
            break

        print(next_item['question'])
        for idx, opt in enumerate(next_item['options'], 1):
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

        correct = 1 if answer == next_item['correct'] else 0
        asked_items.append(next_item)
        responses.append(correct)

        theta = update_theta(asked_items, responses)
        print(f"Updated ability estimate (theta): {theta:.3f}\n")

    print(f"Test finished. Final estimated ability (theta): {theta:.3f}")

run_cat_test()
