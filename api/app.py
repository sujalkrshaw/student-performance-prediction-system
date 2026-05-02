from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Student Performance API")

# Load model
model = joblib.load("models/model.pkl")

# Risk logic
def get_risk(prob):
    if prob > 0.75:
        return "Low Risk"
    elif prob > 0.5:
        return "Medium Risk"
    else:
        return "High Risk"

# Intervention suggestions
def get_interventions(data):
    advice = []
    if data["study_hours"] < 4:
        advice.append("Increase study hours")
    if data["attendance"] < 75:
        advice.append("Improve attendance")
    if data["assignments"] < 60:
        advice.append("Focus on assignments")
    if data["quiz_score"] < 60:
        advice.append("Practice quizzes")
    return advice

# Root check
@app.get("/")
def home():
    return {"message": "API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(student: dict):

    df = pd.DataFrame([student])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": "Pass" if prediction == 1 else "Fail",
        "probability": round(float(probability), 2),
        "risk": get_risk(probability),
        "interventions": get_interventions(student)
    }