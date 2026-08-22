"""
evaluate_model.py
------------------------
Trained model ki performance check karta hai.
"""

import pandas as pd
import os
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate():
    model = joblib.load(os.path.join("models", "mental_health_model.pkl"))

    X_test = pd.read_csv(os.path.join("data", "processed", "X_test.csv"))
    y_test = pd.read_csv(os.path.join("data", "processed", "y_test.csv")).squeeze()

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
    print("Recall   :", round(recall_score(y_test, y_pred, zero_division=0), 4))
    print("F1-Score :", round(f1_score(y_test, y_pred, zero_division=0), 4))

    if len(set(y_test)) > 1:
        print("ROC-AUC  :", round(roc_auc_score(y_test, y_proba), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    evaluate()