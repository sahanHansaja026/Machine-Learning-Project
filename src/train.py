import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ----------------- MLflow setup -----------------
mlflow.set_tracking_uri("http://localhost:5000")  # MLflow server URL
mlflow.set_experiment("Churn_Prediction")        # Experiment name
os.makedirs("mlruns", exist_ok=True)             # Ensure local mlruns folder exists
os.makedirs("models", exist_ok=True)            # Ensure models folder exists

# ----------------- Data loading -----------------
def load_data():
    """Load preprocessed training and testing data"""
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test

# ----------------- Model evaluation -----------------
def evaluate_model(model, X_test, y_test):
    """Compute common classification metrics"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    return metrics

# ----------------- Train and log -----------------
def train_and_log(model, model_name, X_train, X_test, y_train, y_test):
    """Train model, evaluate metrics, and log everything to MLflow"""
    with mlflow.start_run(run_name=model_name):
        # Train
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Log parameters
        mlflow.log_param("model_name", model_name)

        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log model artifact safely
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"  # Folder name inside this run; must be simple
        )

        print(f"\n{model_name} Results:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        return metrics

# ----------------- Main function -----------------
def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Define models
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    }

    best_model = None
    best_score = 0

    # Train all models and log to MLflow
    for name, model in models.items():
        metrics = train_and_log(model, name, X_train, X_test, y_train, y_test)
        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = model

    # Save best model locally
    joblib.dump(best_model, "models/best_model.pkl")
    print("\n✅ Best model saved successfully at models/best_model.pkl")

if __name__ == "__main__":
    main()