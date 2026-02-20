import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def load_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
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


def train_and_log(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        # Log parameters (basic example)
        mlflow.log_param("model_name", model_name)

        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"\n{model_name} Results:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        return metrics


def main():
    mlflow.set_experiment("Churn_Prediction")

    X_train, X_test, y_train, y_test = load_data()

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        metrics = train_and_log(model, name, X_train, X_test, y_train, y_test)

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = model

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")

    print("\n✅ Best model saved successfully!")


if __name__ == "__main__":
    main()