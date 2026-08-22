"""
data_preprocessing.py
------------------------
Kaggle "Student Mental Health" dataset ko clean/prepare karta hai.

Original columns (Kaggle se):
- Timestamp
- Choose your gender
- Age
- What is your course?
- Your current year of Study
- What is your CGPA?
- Marital status
- Do you have Depression?
- Do you have Anxiety?
- Do you have Panic attack?
- Did you seek any specialist for a treatment?
"""

import pandas as pd
import os

RAW_PATH = os.path.join("data", "raw", "student_mental_health.csv")
OUTPUT_PATH = os.path.join("data", "processed", "cleaned_data.csv")

# Original Kaggle column names ko simple, code-friendly naam dena
COLUMN_RENAME_MAP = {
    "Choose your gender": "gender",
    "Age": "age",
    "What is your course?": "course",
    "Your current year of Study": "year_of_study",
    "What is your CGPA?": "cgpa",
    "Marital status": "marital_status",
    "Do you have Depression?": "depression",
    "Do you have Anxiety?": "anxiety",
    "Do you have Panic attack?": "panic_attack",
    "Did you seek any specialist for a treatment?": "sought_treatment",
}


def load_data(path=RAW_PATH):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    # Timestamp ki zaroorat nahi model ke liye
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    df = df.rename(columns=COLUMN_RENAME_MAP)

    # Column names ke aage/peeche ke extra spaces hata dena
    df.columns = [c.strip() for c in df.columns]

    # Duplicate rows aur missing values hatana
    df = df.drop_duplicates()
    df = df.dropna()

    # Text values ko consistent banana (extra spaces, capitalization)
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    # Year of study jaise "year 1", "Year 1" ko normalize karna
    if "year_of_study" in df.columns:
        df["year_of_study"] = df["year_of_study"].str.lower().str.strip()

    return df


def preprocess():
    df = load_data()
    df = clean_data(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Cleaned data save ho gaya: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    preprocess()