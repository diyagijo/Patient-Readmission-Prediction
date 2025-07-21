import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from src.shap_utils import explain_single_patient
from src.genai import generate_recommendations

# =========================
# Load Artifacts
# =========================
st.set_page_config(page_title="Patient Readmission Risk", layout="wide")
st.title(" AI-Powered Patient Readmission Prediction")

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("models/preprocessor.pkl")
    model = joblib.load("C:/Users/diyag/OneDrive/Pictures/Desktop/PROJECTS_IN/Patient Readmission Prediction/models/xgb_readmit.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return preprocessor, model, feature_names

preprocessor, model, feature_names = load_artifacts()

# =========================
# Upload Data
# =========================
uploaded = st.file_uploader("Upload Patient Data CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())

    # Predict
    X_t = preprocessor.transform(df)
    proba = model.predict_proba(X_t)[:, 1]
    df["Readmission_Prob"] = proba
    df["Risk_Level"] = pd.cut(proba, bins=[0, 0.3, 0.6, 1], labels=["Low", "Medium", "High"])

    st.subheader("Predictions")
    st.dataframe(df.sort_values("Readmission_Prob", ascending=False).head(20))

    # Select patient
    idx = st.number_input("Select patient row (0-indexed):", min_value=0, max_value=len(df)-1, value=0)
    patient_info = df.iloc[idx].drop(["Readmission_Prob", "Risk_Level"]).to_dict()

    # SHAP explanation
    st.subheader("SHAP Local Explanation")
    shap_result = explain_single_patient(model, preprocessor, df.iloc[[idx]], feature_names)
    st.pyplot(shap_result["plot"])

    # GenAI recommendations
    st.subheader("GenAI Recommendations")
    recs = generate_recommendations(patient_info, df.iloc[idx]["Readmission_Prob"], shap_result["top_factors"])
    st.write(recs)

else:
    st.info("Upload a patient CSV to see predictions.")
