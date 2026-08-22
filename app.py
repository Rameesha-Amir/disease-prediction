"""
app.py
------------------------
Flask backend jo "MindCheck" interface ko serve karta hai
aur trained model se real prediction deta hai.

Run karne ke liye:
    pip install -r requirements.txt
    python app.py
Phir browser mein kholain: http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_DIR = "models"

model = None
encoders = None
feature_columns = None


def load_assets():
    global model, encoders, feature_columns
    if model is None:
        model_path = os.path.join(MODEL_DIR, "mental_health_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model nahi mila. Pehle 'python src/train_model.py' run karein."
            )
        model = joblib.load(model_path)
        encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
        feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    return model, encoders, feature_columns


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        clf, enc, feature_columns_local = load_assets()

        df = pd.DataFrame([data])
        df["age"] = pd.to_numeric(df["age"])

        for col in ["marital_status", "anxiety", "panic_attack", "sought_treatment"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().map({"yes": 1, "no": 0})

        for col, encoder in enc.items():
            if col in df.columns:
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in encoder.classes_ else encoder.classes_[0]
                )
                df[col] = encoder.transform(df[col])

        df = df[feature_columns_local]

        prediction = int(clf.predict(df)[0])
        probability = float(clf.predict_proba(df)[0][1])

        result = "Higher Risk" if prediction == 1 else "Lower Risk"

        return jsonify({
            "success": True,
            "result": result,
            "prediction": prediction,
            "probability": round(probability * 100, 2),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)