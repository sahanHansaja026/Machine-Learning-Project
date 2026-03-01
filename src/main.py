# src/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/best_model.pkl")

app = FastAPI(title="Churn Prediction API")

# Define input data structure
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    # Add other relevant columns as needed
    # Example categorical features:
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    Contract: str
    PaymentMethod: str

@app.post("/predict")
def predict(data: CustomerData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Make prediction
    pred_prob = model.predict_proba(df)[:, 1][0]
    pred_label = "Yes" if pred_prob > 0.5 else "No"

    return {
        "churn_probability": float(pred_prob),
        "prediction": pred_label
    }