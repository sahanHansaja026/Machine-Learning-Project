import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # path to src folder
RAW_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "Churn Prediction DataSet.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")


# -------------------------
# Functions
# -------------------------
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def clean_data(df):
    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop customerID
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    return df


def encode_data(df):
    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, drop_first=True)
    return df


def scale_data(df):
    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def split_data(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def save_data(X_train, X_test, y_train, y_test):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)


# -------------------------
# Main
# -------------------------
def main():
    print("🚀 Starting preprocessing...")

    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    df = encode_data(df)
    df = scale_data(df)

    X_train, X_test, y_train, y_test = split_data(df)
    save_data(X_train, X_test, y_train, y_test)

    print("✅ Preprocessing completed successfully!")


if __name__ == "__main__":
    main()