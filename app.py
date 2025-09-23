from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from ast import literal_eval

# Try to import py-irt
PYIRT_AVAILABLE = False
try:
    import py_irt
    from py_irt.scoring import calculate_theta
    PYIRT_AVAILABLE = True
except Exception as e:
    print("py_irt not available:", e)
    PYIRT_AVAILABLE = False

# Load item bank
items_df = pd.read_csv("items.csv")
if "id" in items_df.columns:
    items_df["id"] = items_df["id"].astype(int)

# 3PL & related functions (current ones)
def three_pl(theta, a, b, c):
    e_term = np.exp(a * (theta - b))
    return c + (1 - c) * (e_term / (1 + e_term))

def neg_log_likelihood(theta, items_asked, responses):
    ll = 0.0
    for _, item in items_asked.iterrows():
        u = responses[int(item["id"])]
        p = three_pl(theta, item["a"], item["b"], item["c"])
        p = np.clip(p, 1e-6, 1 - 1e-6)
        ll += u * np.log(p) + (1 - u) * np.log(1 - p)
    return -ll

def fisher_information(item, theta):
    a, b, c = item["a"], item["b"], item["c"]
    p = three_pl(theta, a, b, c)
    q = 1 - p
    return (a**2) * (q / p) * ((p - c) ** 2) / ((1 - c) ** 2)

def select_next_item(items_df, asked_ids, theta):
    candidates = items_df[~items_df["id"].isin(asked_ids)].copy()
    if candidates.empty:
        return None
    candidates["info"] = candidates.apply(lambda it: fisher_information(it, theta), axis=1)
    return candidates.loc[candidates["info"].idxmax()]

# theta update: fallback via MLE
def update_theta_mle(items_asked: pd.DataFrame, responses: dict):
    if items_asked.empty:
        return 0.0
    res = minimize_scalar(
        neg_log_likelihood,
        bounds=(-4, 4),
        args=(items_asked, responses),
        method="bounded",
    )
    return float(res.x)

# theta update via py-irt
def update_theta_pyirt(items_asked: pd.DataFrame, responses: dict):
    # Build arrays for py_irt
    # Use difficulty b; if py_irt supports discrimination or guessing, adapt
    difficulties = []
    response_pattern = []
    for _, item in items_asked.iterrows():
        difficulties.append(item["b"])
        response_pattern.append(responses[int(item["id"])])

    difficulties = np.array(difficulties)
    response_pattern = np.array(response_pattern)

    try:
        theta_est = calculate_theta(difficulties, response_pattern, num_obs=-1)
        return float(theta_est)
    except Exception as e:
        print("py_irt theta estimation error:", e)
        # fallback
        return update_theta_mle(items_asked, responses)

def update_theta(items_asked: pd.DataFrame, responses: dict):
    if PYIRT_AVAILABLE:
        return update_theta_pyirt(items_asked, responses)
    else:
        return update_theta_mle(items_asked, responses)

# Flask setup
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message":"FlightPath API running", "py_irt_enabled": PYIRT_AVAILABLE}), 200

@app.route("/api/start", methods=["POST"])
def start():
    return jsonify({"theta":0.0, "asked_ids": [], "responses": {}}), 200

@app.route("/api/next", methods=["POST"])
def next_question():
    data = request.get_json(force=True)
    theta = float(data["theta"])
    asked_ids = [int(i) for i in data.get("asked_ids", [])]

    next_item = select_next_item(items_df, asked_ids, theta)
    if next_item is None:
        return jsonify({"message":"No more items"}), 200

    options = []
    try:
        options = literal_eval(next_item["options"]) if isinstance(next_item["options"], str) else next_item["options"]
    except Exception:
        options = []

    return jsonify({"id":int(next_item["id"]), "question": next_item["question"], "options": options}), 200

@app.route("/api/answer", methods=["POST"])
def answer_question():
    data = request.get_json(force=True)
    theta = float(data["theta"])
    asked_ids = [int(i) for i in data.get("asked_ids", [])]
    responses = {int(k): int(v) for k, v in data.get("responses", {}).items()}

    items_asked = items_df[items_df["id"].isin(asked_ids)]
    updated = update_theta(items_asked, responses)
    return jsonify({"theta": updated}), 200

@app.route("/api/finish", methods=["POST"])
def finish():
    data = request.get_json(force=True)
    theta = float(data["theta"])

    if theta < -1:
        cefr = "A1"
    elif theta < 0:
        cefr = "A2"
    elif theta < 1:
        cefr = "B1"
    elif theta < 2:
        cefr = "B2"
    else:
        cefr = "C1+"

    return jsonify({"final_theta": theta, "cefr": cefr}), 200

if __name__ == "__main__":
    app.run(debug=True)
