import os
import pandas as pd
import joblib
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

def load_data():
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    return X_test, y_test

def main():
    mlflow.set_experiment("Churn_Prediction")

    X_test, y_test = load_data()

    model = joblib.load("models/best_model.pkl")

    with mlflow.start_run(run_name="Evaluation"):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure()
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr)
        plt.title("ROC Curve")
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")

        mlflow.log_metric("final_roc_auc", roc_auc)

    print("✅ Evaluation completed")

if __name__ == "__main__":
    main()