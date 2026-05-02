import shap
import joblib
import pandas as pd

# load model once
model = joblib.load("models/model.pkl")

# TreeExplainer works well for RandomForest
explainer = shap.TreeExplainer(model)

def explain_instance(student_dict: dict):
    df = pd.DataFrame([student_dict])

    # SHAP values
    shap_values = explainer.shap_values(df)

    # For binary classification, use class 1 (Pass)
    values = shap_values[1][0]  # contribution per feature

    explanation = {}
    for col, val in zip(df.columns, values):
        explanation[col] = float(val)

    return explanation