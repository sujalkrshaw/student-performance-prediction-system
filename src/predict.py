import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

def predict_student(study_hours, attendance, assignments, quiz_score):

    data = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "assignments": [assignments],
        "quiz_score": [quiz_score]
    })

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "prediction": "Pass" if prediction == 1 else "Fail",
        "probability": round(float(probability), 2)
    }


if __name__ == "__main__":
    result = predict_student(6, 80, 70, 75)
    print(result)