import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ----------------- Paths -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # path to src folder
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------- MLflow setup -----------------
mlflow.set_tracking_uri("https://dagshub.com/hansajasahan50/mlops-churn-prediction.mlflow")
mlflow.set_experiment("Churn_Prediction")

# ----------------- Data loading -----------------
def load_data():
    X_train_path = os.path.join(PROCESSED_DIR, "X_train.csv")
    X_test_path = os.path.join(PROCESSED_DIR, "X_test.csv")
    y_train_path = os.path.join(PROCESSED_DIR, "y_train.csv")
    y_test_path = os.path.join(PROCESSED_DIR, "y_test.csv")

    # Check existence
    for path in [X_train_path, X_test_path, y_train_path, y_test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test = pd.read_csv(y_test_path).values.ravel()
    return X_train, X_test, y_train, y_test

# ----------------- Model evaluation -----------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

# ----------------- Train and log -----------------
def train_and_log(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        # Log params and metrics
        mlflow.log_param("model_name", model_name)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log model artifact
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        print(f"\n{model_name} Results:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        return metrics

# ----------------- Main -----------------
def main():
    print("🚀 Loading preprocessed data...")
    X_train, X_test, y_train, y_test = load_data()

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        metrics = train_and_log(model, name, X_train, X_test, y_train, y_test)
        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = model

    # Save best model
    best_model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"\n✅ Best model saved successfully at {best_model_path}")

if __name__ == "__main__":
    main()