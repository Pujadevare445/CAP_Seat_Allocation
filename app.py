import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "collegename_model.pkl")

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    print(f"Warning: {MODEL_PATH} not found. Prediction endpoint will fail until added.")

# Mappings based on dataset categories
GENDER_MAP = {"Male": 1, "Female": 0}

CATEGORY_MAP = {
    "GOPENH": 0, "GOPENO": 1, "GSCH": 2, "GSCO": 3,
    "GSTH": 4, "GSTO": 5, "GVJH": 6, "GVJO": 7,
    "GNT1H": 8, "GNT1O": 9, "GNT2H": 10, "GNT2O": 11,
    "GNT3H": 12, "GNT3O": 13, "GOBCH": 14, "GOBCO": 15,
    "TFWS": 16, "EWS": 17, "LOPENH": 18, "LOPENO": 19,
    "LSCH": 20, "LSCO": 21, "LSTH": 22, "LSTO": 23,
    "LVJH": 24, "LVJO": 25, "LNT1H": 26, "LNT1O": 27,
    "LNT2H": 28, "LNT2O": 29, "LNT3H": 30, "LNT3O": 31,
    "LOBCH": 32, "LOBCO": 33, "PWD": 34, "DEF": 35
}

SEAT_ALLOTED_MAP = {
    "CAP Round I": 1,
    "CAP Round II": 2,
    "CAP Round III": 3
}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model file not loaded correctly on server."}), 500

    try:
        data = request.get_json()
        
        percentile = float(data.get("percentile", 0.0))
        gender_val = GENDER_MAP.get(data.get("gender"), 0)
        category_val = CATEGORY_MAP.get(data.get("category"), 0)
        seat_val = SEAT_ALLOTED_MAP.get(data.get("seat_alloted"), 1)

        # Feature Order: [MHTCET Percentile, Gender, Category, Seat Alloted]
        features = np.array([[percentile, gender_val, category_val, seat_val]])
        
        prediction = model.predict(features)[0]
        
        return jsonify({
            "success": True,
            "prediction": str(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
