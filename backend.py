
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import numpy as np

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data.get("features", [])
    if not features:
        return jsonify({"error": "No features provided"}), 400

    # Example: return dummy score (replace with real logic)
    score = sum(features) / len(features)
    return jsonify({"compatibility_score": score})

app.run()
