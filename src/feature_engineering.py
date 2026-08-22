"""
feature_engineering.py
------------------------
Cleaned data ke categorical columns ko numbers mein convert karta hai
(models sirf numbers samajhte hain, text nahi).
"""

import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder

INPUT_PATH = os.path.join("data", "processed", "cleaned_data.csv")
OUTPUT_PATH = os.path.join("data", "processed", "featured_data.csv")
ENCODERS_PATH = os.path.join("models", "encoders.pkl")

# Yes/No columns jinhe 1/0 mein convert karna hai
YES_NO_COLUMNS = ["marital_status", "anxiety", "panic_attack", "sought_treatment", "depression"]

# Categorical columns jinhe LabelEncoder se encode karna hai
CATEGORICAL_COLUMNS = ["gender", "course", "year_of_study", "cgpa"]


def encode_yes_no(df):
    for col in YES_NO_COLUMNS:
        if col in df.columns:
            df[col] = df[col].str.lower().map({"yes": 1, "no": 0})
    return df


def encode_categoricals(df):
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder
    return df, encoders


def run_feature_engineering():
    df = pd.read_csv(INPUT_PATH)

    df = encode_yes_no(df)
    df, encoders = encode_categoricals(df)

    # Agar koi row missing values de rahi ho encoding ke baad
    df = df.dropna()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    os.makedirs(os.path.dirname(ENCODERS_PATH), exist_ok=True)
    joblib.dump(encoders, ENCODERS_PATH)

    print(f"Feature engineering complete: {OUTPUT_PATH}")
    print(f"Encoders save hue: {ENCODERS_PATH}")
    print(df.head())
    return df


if __name__ == "__main__":
    run_feature_engineering()