"""
FlightPath Adaptive Assessment API (Enhanced Version)
-----------------------------------------------------
Implements an adaptive testing backend using Item Response Theory (3PL Model).
Supports py_irt integration when available. Includes CEFR mapping based on
final estimated theta value.

Author: (Your Name)
Date: October 2025
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from ast import literal_eval

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

APP_VERSION = "1.1.0"
PYIRT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional dependency: py-irt
# ---------------------------------------------------------------------------

try:
    import py_irt
    from py_irt.scoring import calculate_theta
    PYIRT_AVAILABLE = True
except Exception as e:
    print(f"[Warning] py_irt not available: {e}")
    PYIRT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Load item bank
# ---------------------------------------------------------------------------

try:
    items_df = pd.read_csv("items.csv")
    if "id" in items_df.columns:
        items_df["id"] = items_df["id"].astype(int)
except Exception as e:
    raise FileNotFoundError("Missing or invalid items.csv file!") from e

# ---------------------------------------------------------------------------
# IRT Model: 3-Parameter Logistic (3PL)
# ---------------------------------------------------------------------------

def three_pl_model(theta: float, a: float, b: float, c: float) -> float:
    """3-Parameter Logistic model for probability of a correct response."""
    exp_term = np.exp(a * (theta - b))
    return c + (1 - c) * (exp_term / (1 + exp_term))

def neg_log_likelihood(theta: float, items_asked: pd.DataFrame, responses: dict) -> float:
    """Compute negative log-likelihood for given responses."""
    ll = 0.0
    for _, item in items_asked.iterrows():
        u = responses[int(item["id"])]
        p = np.clip(three_pl_model(theta, item["a"], item["b"], item["c"]), 1e-6, 1 - 1e-6)
        ll += u * np.log(p) + (1 - u) * np.log(1 - p)
    return -ll

def fisher_info(item_row: pd.Series, theta: float) -> float:
    """Compute Fisher Information for item selection."""
    a, b, c = item_row["a"], item_row["b"], item_row["c"]
    p = three_pl_model(theta, a, b, c)
    q = 1 - p
    return (a ** 2) * (q / p) * ((p - c) ** 2) / ((1 - c) ** 2)

def select_next_item(items: pd.DataFrame, asked_ids: list[int], theta: float):
    """Select next item maximizing Fisher Information."""
    pool = items[~items["id"].isin(asked_ids)].copy()
    if pool.empty:
        return None
    pool["info"] = pool.apply(lambda r: fisher_info(r, theta), axis=1)
    return pool.loc[pool["info"].idxmax()]

# ---------------------------------------------------------------------------
# Theta estimation functions
# ---------------------------------------------------------------------------

def update_theta_mle(items_asked: pd.DataFrame, responses: dict) -> float:
    """Estimate ability (theta) via Maximum Likelihood Estimation."""
    if items_asked.empty:
        return 0.0
    res = minimize_scalar(
        neg_log_likelihood,
        bounds=(-4, 4),
        args=(items_asked, responses),
        method="bounded",
    )
    return float(res.x)

def update_theta_pyirt(items_asked: pd.DataFrame, responses: dict) -> float:
    """Estimate ability using py_irt (if installed)."""
    difficulties = np.array(items_asked["b"])
    response_pattern = np.array([responses[int(i)] for i in items_asked["id"]])
    try:
        theta_est = calculate_theta(difficulties, response_pattern, num_obs=-1)
        return float(theta_est)
    except Exception as e:
        print(f"[Warning] py_irt estimation failed, using MLE fallback: {e}")
        return update_theta_mle(items_asked, responses)

def update_theta(items_asked: pd.DataFrame, responses: dict) -> float:
    """Unified theta updater (prefers py_irt if available)."""
    if PYIRT_AVAILABLE:
        return update_theta_pyirt(items_asked, responses)
    return update_theta_mle(items_asked, responses)

# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Health check and info route."""
    return jsonify({
        "message": "FlightPath API operational",
        "version": APP_VERSION,
        "py_irt_enabled": PYIRT_AVAILABLE
    }), 200

@app.route("/api/start", methods=["POST"])
def start_session():
    """Initialize a new test session."""
    return jsonify({
        "theta": 0.0,
        "asked_ids": [],
        "responses": {}
    }), 200

@app.route("/api/next", methods=["POST"])
def next_question():
    """Select and return the next most informative question."""
    try:
        data = request.get_json(force=True)
        theta = float(data.get("theta", 0.0))
        asked_ids = [int(i) for i in data.get("asked_ids", [])]
        next_item = select_next_item(items_df, asked_ids, theta)
        if next_item is None:
            return jsonify({"message": "No more items available."}), 200

        options = []
        try:
            if isinstance(next_item["options"], str):
                options = literal_eval(next_item["options"])
            else:
                options = next_item["options"]
        except Exception:
            options = []

        return jsonify({
            "id": int(next_item["id"]),
            "question": next_item["question"],
            "options": options
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/answer", methods=["POST"])
def answer_question():
    """Process user's response and update theta."""
    try:
        data = request.get_json(force=True)
        theta = float(data.get("theta", 0.0))
        asked_ids = [int(i) for i in data.get("asked_ids", [])]
        responses = {int(k): int(v) for k, v in data.get("responses", {}).items()}

        items_asked = items_df[items_df["id"].isin(asked_ids)]
        updated_theta = update_theta(items_asked, responses)

        return jsonify({"theta": updated_theta}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/finish", methods=["POST"])
def finish_session():
    """Finalize session and map final theta to CEFR level."""
    try:
        data = request.get_json(force=True)
        theta = float(data.get("theta", 0.0))

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

        return jsonify({
            "final_theta": theta,
            "cefr_level": cefr
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
