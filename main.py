from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

# Serve static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at root route
@app.get("/")
def read_index():
    return FileResponse(os.path.join("static", "index.html"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import your_model  # your_model.py file

# Load CSVs once at startup
disease_df = pd.read_csv("data/orphan_diseases.csv")
drug_df = pd.read_csv("data/drug_testing_data.csv")

# FastAPI app
app = FastAPI()

# Input model from frontend
class DrugInput(BaseModel):
    drug_name: str
    orphan_disease_name: str

@app.get("/")
def home():
    return {"message": "Drug repurposing backend is live!"}

@app.post("/predict")
def predict(input: DrugInput):
    # Search by name
    drug_row = drug_df[drug_df['Drug Name'].str.lower() == input.drug_name.lower()]
    disease_row = disease_df[disease_df['OrphanDiseases Name'].str.lower() == input.orphan_disease_name.lower()]

    if drug_row.empty or disease_row.empty:
        raise HTTPException(status_code=404, detail="Drug or disease not found.")

    # Extract key info
    drug_smiles = drug_row.iloc[0]['SMILES']
    drug_mechanism = drug_row.iloc[0]['Mechanism of Action']
    drug_binding = drug_row.iloc[0]['Binding Affinity']
    drug_toxicity = drug_row.iloc[0]['Toxicity Score']
    target_protein = drug_row.iloc[0]['Target Protein']

    disease_gene = disease_row.iloc[0]['Gene']
    disease_pathway = disease_row.iloc[0]['pathway']
    disease_mechanism = disease_row.iloc[0]['mechanism']

    # Send to model
    result = your_model.predict(
        smiles=drug_smiles,
        gene=disease_gene,
        pathway=disease_pathway,
        drug_mechanism=drug_mechanism,
        target_protein=target_protein,
        disease_mechanism=disease_mechanism,
        binding_affinity=drug_binding,
        toxicity=drug_toxicity
    )

    return result
