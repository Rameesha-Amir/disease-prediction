# MindCheck — Student Mental Health Prediction

This project predicts a student's depression risk using survey data on
academic life, lifestyle, and demographics. It uses the Kaggle
**"Student Mental Health"** dataset.

**Disclaimer:** This is a learning project, not a medical or diagnostic tool.
It should never be used to make real health decisions.

## Folder Structure

```
disease-prediction/
│
├── data/
│   ├── raw/
│   │   └── student_mental_health.csv     ← place the downloaded Kaggle CSV here
│   └── processed/
│       ├── cleaned_data.csv
│       ├── featured_data.csv
│       ├── X_test.csv
│       └── y_test.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── templates/
│   └── index.html                         ← "MindCheck" web interface
│
├── app.py
├── models/
│   ├── mental_health_model.pkl
│   ├── encoders.pkl
│   └── feature_columns.pkl
│
├── requirements.txt
└── README.md
```

## Step 1: Get the Dataset

1. Go to: https://www.kaggle.com/datasets/shariful07/student-mental-health
2. Download the CSV (Kaggle login required)
3. Rename it to `student_mental_health.csv` and place it in `data/raw/`

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```
(On Windows, if `pip` isn't recognized, use `py -m pip install -r requirements.txt`)

## Step 3: Run the Pipeline (in order)

```bash
python src/data_preprocessing.py
python src/feature_engineering.py
python src/train_model.py
python src/evaluate_model.py
python src/predict.py
```

## What Happens in Each Step

1. **data_preprocessing.py** - Cleans the raw CSV, renames columns to code-friendly names
2. **feature_engineering.py** - Encodes text columns (gender, course, CGPA, etc.) into numbers
3. **train_model.py** - Trains a Random Forest classifier to predict depression risk
4. **evaluate_model.py** - Reports accuracy, precision, recall, F1-score, ROC-AUC
5. **predict.py** - Predicts risk for a single new student's data

## Web Interface — "MindCheck"

A warm, friendly interface where a student can answer a short form and see
a live wellbeing risk estimate.

To run:
```bash
python app.py
```

Open the link shown in the terminal (e.g. `http://127.0.0.1:5000`) in your browser.

## Tech Stack

- Python
- pandas, scikit-learn
- Flask
- HTML, CSS, JavaScript
