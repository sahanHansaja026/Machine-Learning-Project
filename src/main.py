import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load("models/best_model.pkl")

app = FastAPI(title="Churn Prediction API")

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    Contract: str
    PaymentMethod: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    PaperlessBilling: str

@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])

    # Encode categorical variables like in training
    df = pd.get_dummies(df)
    
    # Add missing columns with 0 (so columns match training)
    trained_columns = model.feature_names_in_  # scikit-learn keeps this
    for col in trained_columns:
        if col not in df.columns:
            df[col] = 0
    # Reorder columns
    df = df[trained_columns]

    # Scale numeric columns like in training
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])  # optional if you want simple scaling

    # Predict
    pred_prob = model.predict_proba(df)[:, 1][0]
    pred_label = "Yes" if pred_prob > 0.5 else "No"

    return {
        "churn_probability": float(pred_prob),
        "prediction": pred_label
    }