"""
predict.py
------------------------
Naye student ka data de kar depression risk predict karne ke liye.
"""

import pandas as pd
import joblib
import os


def predict_student(student_data: dict):
    """
    student_data example:
    {
        "gender": "Female",
        "age": 21,
        "course": "Engineering",
        "year_of_study": "year 2",
        "cgpa": "3.50 - 4.00",
        "marital_status": "No",
        "anxiety": "No",
        "panic_attack": "No",
        "sought_treatment": "No"
    }
    """
    model = joblib.load(os.path.join("models", "mental_health_model.pkl"))
    encoders = joblib.load(os.path.join("models", "encoders.pkl"))
    feature_columns = joblib.load(os.path.join("models", "feature_columns.pkl"))

    df = pd.DataFrame([student_data])

    # Yes/No fields ko 1/0 mein convert karna
    for col in ["marital_status", "anxiety", "panic_attack", "sought_treatment"]:
        if col in df.columns:
            df[col] = df[col].str.lower().map({"yes": 1, "no": 0})

    # Categorical fields ko wahi encoders se transform karna jo training mein use hue the
    for col, encoder in encoders.items():
        if col in df.columns and col != "depression":
            # Agar koi naya/unseen value ho to sabse common class assume kar lete hain
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in encoder.classes_ else encoder.classes_[0]
            )
            df[col] = encoder.transform(df[col])

    df = df[feature_columns]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    result = "Higher Risk" if prediction == 1 else "Lower Risk"
    print(f"Prediction: {result}")
    print(f"Risk Probability: {round(probability * 100, 2)}%")

    return prediction, probability


if __name__ == "__main__":
    sample_student = {
        "gender": "Female",
        "age": 21,
        "course": "Engineering",
        "year_of_study": "year 2",
        "cgpa": "3.50 - 4.00",
        "marital_status": "No",
        "anxiety": "No",
        "panic_attack": "No",
        "sought_treatment": "No",
    }

    predict_student(sample_student)