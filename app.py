#This is the version 1 of the REST API

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# ---------------------------
# CAT Engine (3PL IRT logic)
# ---------------------------

items_df = pd.read_csv("items.csv")


def three_pl(theta, a, b, c):
    e_term = np.exp(a * (theta - b))
    return c + (1 - c) * (e_term / (1 + e_term))


def neg_log_likelihood(theta, items_asked, responses):
    ll = 0
    for _, item in items_asked.iterrows():
        # make sure we always match int ids
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
    candidates["info"] = candidates.apply(
        lambda item: fisher_information(item, theta), axis=1
    )
    return candidates.loc[candidates["info"].idxmax()]


def update_theta(items_asked, responses):
    result = minimize_scalar(
        neg_log_likelihood,
        bounds=(-4, 4),
        args=(items_asked, responses),
        method="bounded",
    )
    return result.x


# ---------------------------
# Flask REST API
# ---------------------------

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "FlightPath API is running"}), 200


@app.route("/api/start", methods=["POST"])
def start_test():
    theta = 0
    return jsonify({"theta": theta, "asked_ids": [], "responses": {}}), 200


@app.route("/api/next", methods=["POST"])
def next_question():
    data = request.get_json()
    theta = data["theta"]
    asked_ids = data.get("asked_ids", [])

    item = select_next_item(items_df, asked_ids, theta)
    if item is None:
        return jsonify({"message": "No more items"}), 200

    return jsonify(
        {
            "id": int(item["id"]),
            "question": item["question"],
            "options": eval(item["options"]),
        }
    ), 200


@app.route("/api/answer", methods=["POST"])
def answer_question():
    data = request.get_json()
    theta = data["theta"]
    asked_ids = data["asked_ids"]

    # convert keys in responses to int
    responses = {int(k): v for k, v in data["responses"].items()}

    items_asked = items_df[items_df["id"].isin(asked_ids)]
    updated_theta = update_theta(items_asked, responses)

    return jsonify({"theta": updated_theta}), 200


@app.route("/api/finish", methods=["POST"])
def finish_test():
    data = request.get_json()
    theta = data["theta"]

    # Example CEFR mapping
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
