
# Customer Churn Prediction with MLOps

This repository implements a **production-ready machine learning system** for predicting customer churn in a subscription-based service. The system integrates **DVC, MLflow, Airflow, Docker, FastAPI**, and versioning on **DAGsHub**.

---

## Project Overview

The goal of this project is to predict whether a customer is likely to churn, based on their subscription, usage, and billing data. The project follows an end-to-end MLOps pipeline:

- **Data ingestion & preprocessing** using DVC
- **Model training and evaluation** with MLflow
- **Airflow DAG orchestration**
- **REST API deployment** using FastAPI and Docker
- **Experiment tracking and model versioning** via DAGsHub

---

## Dataset

The dataset used is the [Telco Customer Churn dataset](data/raw/Churn Prediction DataSet.csv) with **7043 rows** and **21 columns** including:

- Customer demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`)
- Subscription details (`Contract`, `PaymentMethod`, `PaperlessBilling`)
- Service usage (`PhoneService`, `InternetService`, `StreamingTV`, etc.)
- Billing (`MonthlyCharges`, `TotalCharges`)
- Target: `Churn` (`Yes`/`No`)

---

## Project Structure

```text
churn-mlops-project/
│
├── data/
│   ├── raw/
│   |   └── Churn Prediction DataSet.csv 
│   └── processed/
│       ├── X_test.csv
│       ├── X_train.csv
│       ├── Y_test.csv
│       └── Y_train.csv 
├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── main.py          # FastAPI app
├── api/
├── models/
├── mlruns/
├── Dockerfile
├── requirements.txt
├── dvc.yaml
└── README.md
````

---


## DVC Pipeline

The DVC stages:

1. **data_ingestion** – Loads raw data from `data/raw`.
2. **preprocessing** – Cleans and encodes data, splits into train/test.
3. **training** – Trains Logistic Regression, Random Forest, and XGBoost models, logs experiments with MLflow.
4. **evaluation** – Evaluates the models and generates confusion matrices and ROC curves.


---

## MLflow Tracking

* Experiments, metrics, parameters, and model artifacts are tracked with MLflow.
* MLflow UI can be accessed at: `http://localhost:5000`

---

## API Deployment

* The best model is deployed via **FastAPI**.
* **Endpoint**: `POST /predict`
* **Input JSON Example**:

```json
{
  "tenure": 12,
  "MonthlyCharges": 75.5,
  "TotalCharges": 910.0,
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check"
}
```

* **Response Example**:

```json
{
  "churn_probability": 0.62,
  "prediction": "Yes"
}
```

* API runs on port `8000`, Docker container exposes ports `8000` (API) and `5000` (MLflow UI).

---

## Running with Docker

1. Build the image:

```bash
docker build -t churn-api:latest .
```

2. Run the container:

```bash
docker run -d -p 8000:8000 -p 5000:5000 churn-api:latest
```

3. Access API and MLflow UI:

* API: `http://localhost:8000/docs`
* MLflow: `http://localhost:5000`

---

## Future Improvements

* Integrate LLM-based retention incentives.
* Add hyperparameter tuning with Optuna.
* Production-grade CI/CD deployment.

---

## License

MIT License
