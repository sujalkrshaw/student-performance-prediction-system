import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Student Dashboard", layout="wide")

# -------------------------
# 🔥 API CONFIG (UPDATED)
# -------------------------
# For now keep localhost. Replace later with Render URL
API_URL = "http://127.0.0.1:8000/predict"

# HEADER
st.title("🎓 Student Performance Dashboard")
st.markdown("Predict student success and identify at-risk learners")

# SIDEBAR INPUT
st.sidebar.header("📥 Input Student Data")

study_hours = st.sidebar.slider("Study Hours", 0, 12, 5)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 70)
assignments = st.sidebar.slider("Assignment Score", 0, 100, 60)
quiz_score = st.sidebar.slider("Quiz Score", 0, 100, 60)

# API CALL FUNCTION
def get_prediction(data):
    try:
        response = requests.post(API_URL, json=data)
        return response.json()
    except:
        st.error("⚠ API not running. Please start backend.")
        st.stop()

# -------------------------
# 🔮 SINGLE PREDICTION
# -------------------------
if st.sidebar.button("🚀 Predict"):

    result = get_prediction({
        "study_hours": study_hours,
        "attendance": attendance,
        "assignments": assignments,
        "quiz_score": quiz_score
    })

    st.subheader("📊 Prediction Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Result", result["prediction"])
    col2.metric("Confidence", result["probability"])
    col3.metric("Risk Level", result["risk"])

    st.markdown("---")

    if result["risk"] == "High Risk":
        st.error("🚨 High Risk Student")
    elif result["risk"] == "Medium Risk":
        st.warning("⚠ Medium Risk Student")
    else:
        st.success("✅ Low Risk Student")

    st.subheader("📌 Recommended Actions")

    if result["interventions"]:
        for action in result["interventions"]:
            st.write(f"✔ {action}")
    else:
        st.write("No action needed 🎉")

    # MODEL INSIGHTS
    st.markdown("---")
    st.subheader("📊 Model Insights")

    model = joblib.load("models/model.pkl")
    feature_names = ["study_hours", "attendance", "assignments", "quiz_score"]
    importances = model.feature_importances_

    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh(feature_names, importances)
    st.pyplot(fig_imp)

    # Study Hours vs Probability
    st.subheader("📈 Study Hours vs Probability")

    hours_range = list(range(1, 11))
    probs = []

    for h in hours_range:
        res = get_prediction({
            "study_hours": h,
            "attendance": attendance,
            "assignments": assignments,
            "quiz_score": quiz_score
        })
        probs.append(res["probability"])

    fig_curve, ax_curve = plt.subplots()
    ax_curve.plot(hours_range, probs, marker='o')
    ax_curve.set_ylim(0, 1)
    ax_curve.grid(True)
    st.pyplot(fig_curve)

# -------------------------
# 📂 BATCH PREDICTION
# -------------------------
st.markdown("---")
st.subheader("📂 Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV file")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ✅ VALIDATION (NEW)
    required_cols = ["study_hours", "attendance", "assignments", "quiz_score"]
    if not all(col in df.columns for col in required_cols):
        st.error("❌ Invalid CSV format")
        st.stop()

    predictions, risks, probabilities, interventions_list = [], [], [], []

    for _, row in df.iterrows():
        res = get_prediction(row.to_dict())

        predictions.append(res["prediction"])
        risks.append(res["risk"])
        probabilities.append(res["probability"])

        interventions_list.append(
            ", ".join(res["interventions"]) if res["interventions"] else "No action needed"
        )

    df["prediction"] = predictions
    df["probability"] = probabilities
    df["risk_level"] = risks
    df["interventions"] = interventions_list

    # KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("Students", len(df))
    col2.metric("High Risk", (df["risk_level"] == "High Risk").sum())
    col3.metric("Pass Rate", f"{(df['prediction']=='Pass').mean():.2f}")

    st.markdown("---")
    st.dataframe(df)

    # Risk Distribution
    st.subheader("📈 Risk Distribution")

    risk_order = ["Low Risk", "Medium Risk", "High Risk"]
    risk_counts = df["risk_level"].value_counts().reindex(risk_order, fill_value=0)

    fig, ax = plt.subplots()
    ax.bar(risk_counts.index, risk_counts.values, color=["green", "orange", "red"])
    ax.grid(axis='y')
    st.pyplot(fig)

    # Line Graphs
    st.subheader("📊 Study Hours vs Prediction")

    df_sorted = df.sort_values("study_hours")

    fig2, ax2 = plt.subplots()
    ax2.plot(df_sorted["study_hours"], df_sorted["prediction"].map({"Pass": 1, "Fail": 0}), marker='o')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True)
    st.pyplot(fig2)

    st.subheader("📊 Attendance vs Risk")

    risk_map = {"Low Risk": 1, "Medium Risk": 2, "High Risk": 3}
    df["Risk_Num"] = df["risk_level"].map(risk_map)

    df_sorted2 = df.sort_values("attendance")

    fig3, ax3 = plt.subplots()
    ax3.plot(df_sorted2["attendance"], df_sorted2["Risk_Num"], marker='o')
    ax3.set_ylim(0.5, 3.5)
    ax3.grid(True)
    st.pyplot(fig3)

    # Download
    st.download_button(
        "⬇ Download Results",
        df.to_csv(index=False),
        file_name="results.csv"
    )