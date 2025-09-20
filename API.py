from flask import Flask, request, jsonify
from flask_cors import CORS
import app  # import your CAT engine logic (items_df, functions)

app_api = Flask(__name__)
CORS(app_api)

@app_api.route("/", methods=["GET"])
def home():
    return jsonify({"message": "FlightPath API is running"}), 200

@app_api.route("/api/start", methods=["POST"])
def start_test():
    theta = 0
    return jsonify({
        "theta": theta,
        "asked_ids": [],
        "responses": {}
    }), 200

@app_api.route("/api/next", methods=["POST"])
def next_question():
    data = request.get_json()
    theta = data["theta"]
    asked_ids = data.get("asked_ids", [])

    item = app.select_next_item(app.items_df, asked_ids, theta)
    if item is None:
        return jsonify({"message": "No more items"}), 200

    return jsonify({
        "id": int(item["id"]),
        "question": item["question"],
        "options": eval(item["options"])  # convert stored string to list
    }), 200

@app_api.route("/api/answer", methods=["POST"])
def answer_question():
    data = request.get_json()
    theta = data["theta"]
    asked_ids = data["asked_ids"]
    responses = data["responses"]

    items_asked = app.items_df[app.items_df["id"].isin(asked_ids)]
    updated_theta = app.update_theta(items_asked, responses)

    return jsonify({"theta": updated_theta}), 200

@app_api.route("/api/finish", methods=["POST"])
def finish_test():
    data = request.get_json()
    theta = data["theta"]

    # Map theta to CEFR levels (example thresholds)
    if theta < -1: cefr = "A1"
    elif theta < 0: cefr = "A2"
    elif theta < 1: cefr = "B1"
    elif theta < 2: cefr = "B2"
    else: cefr = "C1+"

    return jsonify({
        "final_theta": theta,
        "cefr": cefr
    }), 200

if __name__ == "__main__":
    app_api.run(debug=True)
