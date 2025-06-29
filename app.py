from flask import render_template
from flask_cors import CORS

CORS(app)

@app.route("/")
def home():
    return render_template("index.html")
from flask import Flask, request, jsonify
import numpy as np
import joblib
import shap
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

from rdkit import Chem
from rdkit.Chem import Descriptors

app = Flask(_name_)

# Load models
model_compat = load_model("compat_model.h5")
model_toxic = load_model("toxic_model.h5")
scaler = joblib.load("scaler.save")  # save with joblib.dump(scaler, "scaler.save")

# Features for SHAP
X_full = pd.read_csv("X_features.csv")  # Export this from your training notebook
explainer = shap.Explainer(lambda x: model_compat.predict(scaler.transform(x)), X_full)

# Ideal vector
ideal = X_full.iloc[model_compat.predict(scaler.transform(X_full)).flatten().argsort()[-10:]].mean()

def featurize(smiles, pathway_coverage, binding_affinity):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        features = [
            float(pathway_coverage),
            float(binding_affinity),
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol)
        ]
        return features
    except:
        return None

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    smiles = data.get("smiles")
    pathway = float(data.get("pathway_coverage", 0.6))
    affinity = float(data.get("binding_affinity", 0.0))

    features = featurize(smiles, pathway, affinity)
    if not features:
        return jsonify({"error": "Invalid SMILES"}), 400

    input_scaled = scaler.transform([features])
    compatibility = float(model_compat.predict(input_scaled)[0][0])
    toxicity = float(model_toxic.predict(input_scaled)[0][0])

    shap_values = explainer([features])
    shap_df = pd.DataFrame({
        "feature": X_full.columns,
        "impact": shap_values.values[0]
    }).sort_values("impact", key=abs, ascending=False)

    reasons = [
        f"{row['feature']} {'positively' if row['impact'] > 0 else 'negatively'} contributed"
        for _, row in shap_df.iterrows()
    ][:3]

    diff = ideal - np.array(features)
    improvement = []
    for i, val in enumerate(diff):
        if abs(val) > 0.1:
            direction = "increase" if val > 0 else "decrease"
            improvement.append(f"{direction} {X_full.columns[i]} by {abs(round(val, 2))}")

    return jsonify({
        "compatibility_score": round(compatibility, 2),
        "toxicity_score": round(toxicity, 2),
        "reasons": reasons,
        "why_not_100": improvement or ["Near perfect match."],
        "suggestions": improvement or ["Structure is strong."]
    })

if _name_ == "_main_":
    app.run(debug=True)
