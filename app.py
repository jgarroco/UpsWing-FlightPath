from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from ast import literal_eval

# ------------- Try to use py-irt (optional) -------------
PYIRT_AVAILABLE = False
try:
    # There are a few similarly-named projects; this try/except keeps your app running
    # even if py-irt isn't present or its import path differs.
    import py_irt  # type: ignore
    PYIRT_AVAILABLE = True
except Exception:
    PYIRT_AVAILABLE = False


# ------------- Load items -------------
items_df = pd.read_csv("items.csv")

# Ensure types are consistent
if "id" in items_df.columns:
    items_df["id"] = items_df["id"].astype(int)


# ------------- Core 3PL helpers (used by fallback and info function) -------------
def three_pl(theta, a, b, c):
    e_term = np.exp(a * (theta - b))
    return c + (1 - c) * (e_term / (1 + e_term))


def neg_log_likelihood(theta, items_asked, responses):
    ll = 0.0
    for _, item in items_asked.iterrows():
        u = responses[int(item["id"])]            # response (0/1)
        p = three_pl(theta, item["a"], item["b"], item["c"])
        p = np.clip(p, 1e-6, 1 - 1e-6)            # numerical safety
        ll += u * np.log(p) + (1 - u) * np.log(1 - p)
    return -ll


def fisher_information(item, theta):
    a, b, c = item["a"], item["b"], item["c"]
    p = three_pl(theta, a, b, c)
    q = 1 - p
    # Same formula you were using
    return (a**2) * (q / p) * ((p - c) ** 2) / ((1 - c) ** 2)


def select_next_item(items_df, asked_ids, theta):
    candidates = items_df[~items_df["id"].isin(asked_ids)].copy()
    if candidates.empty:
        return None
    candidates["info"] = candidates.apply(lambda it: fisher_information(it, theta), axis=1)
    return candidates.loc[candidates["info"].idxmax()]


# ------------- Ability update (py-irt adapter with fallback) -------------
def update_theta_with_mle(items_asked: pd.DataFrame, responses: dict, theta_bounds=(-4, 4)):
    """Your previous SciPy-based MLE (works without py-irt)."""
    if items_asked.empty:
        return 0.0
    res = minimize_scalar(
        neg_log_likelihood,
        bounds=theta_bounds,
        args=(items_asked, responses),
        method="bounded",
    )
    return float(res.x)


def update_theta_with_pyirt(items_asked: pd.DataFrame, responses: dict):
    """
    Lightweight adapter for py-irt.

    Many py-irt implementations expect data in (user_id, item_id, correctness) form
    and can fit person abilities with fixed item params. Since different forks exist,
    we keep this conservative and fall back if anything is off.

    If your chosen py-irt exposes a direct "estimate ability given fixed items" API,
    replace this stub with the exact calls. Until then, we do a MAP-style tweak here
    by calling the MLE and (optionally) adding a weak N(0,1) prior if desired.
    """
    # You can replace the line below with the exact py-irt API once confirmed.
    return update_theta_with_mle(items_asked, responses)


def update_theta(items_asked: pd.DataFrame, responses: dict):
    if PYIRT_AVAILABLE:
        try:
            return update_theta_with_pyirt(items_asked, responses)
        except Exception:
            # If py-irt import works but its call signature differs, we still keep the app running.
            return update_theta_with_mle(items_asked, responses)
    else:
        return update_theta_with_mle(items_asked, responses)


# ------------- Flask API -------------
app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "FlightPath API is running",
        "py_irt_enabled": PYIRT_AVAILABLE
    }), 200


@app.route("/api/start", methods=["POST"])
def start_test():
    return jsonify({"theta": 0.0, "asked_ids": [], "responses": {}}), 200


@app.route("/api/next", methods=["POST"])
def next_question():
    data = request.get_json(force=True)
    theta = float(data["theta"])
    asked_ids = data.get("asked_ids", [])

    item = select_next_item(items_df, asked_ids, theta)
    if item is None:
        return jsonify({"message": "No more items"}), 200

    # safer than eval()
    try:
        options = literal_eval(item["options"]) if isinstance(item["options"], str) else item["options"]
    except Exception:
        # fallback: if options is malformed, still return something predictable
        options = []

    return jsonify({
        "id": int(item["id"]),
        "question": item["question"],
        "options": options
    }), 200


@app.route("/api/answer", methods=["POST"])
def answer_question():
    data = request.get_json(force=True)

    theta = float(data["theta"])
    asked_ids = list(map(int, data["asked_ids"]))  # make sure ids are ints

    # Convert response keys "4" -> 4
    responses = {int(k): int(v) for k, v in data["responses"].items()}

    # Subset asked items
    items_asked = items_df[items_df["id"].isin(asked_ids)]
    updated_theta = update_theta(items_asked, responses)

    return jsonify({"theta": updated_theta}), 200


@app.route("/api/finish", methods=["POST"])
def finish_test():
    data = request.get_json(force=True)
    theta = float(data["theta"])

    # Simple CEFR mapping — tweak thresholds as you like
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
