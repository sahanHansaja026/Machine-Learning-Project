import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop customerID
    df.drop("customerID", axis=1, inplace=True)

    return df


def encode_data(df):
    # One-hot encoding
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
    os.makedirs("data/processed", exist_ok=True)

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)


def main():
    data_path = "data/raw/Churn Prediction DataSet.csv"

    df = load_data(data_path)
    df = clean_data(df)
    df = encode_data(df)
    df = scale_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    save_data(X_train, X_test, y_train, y_test)

    print("✅ Preprocessing completed successfully!")


if __name__ == "__main__":
    main()