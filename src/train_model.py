"""
train_model.py
------------------------
Random Forest classifier train karta hai depression predict karne ke liye.
"""

import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = os.path.join("data", "processed", "featured_data.csv")
MODEL_DIR = "models"


def train():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["depression"])
    y = df["depression"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",  # chhota/imbalanced dataset ke liye madadgar
    )
    model.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "mental_health_model.pkl")
    joblib.dump(model, model_path)

    # Column order save karna zaroori hai taake predict.py mein sahi order mile
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_columns.pkl"))

    X_test.to_csv(os.path.join("data", "processed", "X_test.csv"), index=False)
    y_test.to_csv(os.path.join("data", "processed", "y_test.csv"), index=False)

    print(f"Model train ho gaya aur save hua: {model_path}")
    print(f"Features used: {list(X.columns)}")
    return model


if __name__ == "__main__":
    train()