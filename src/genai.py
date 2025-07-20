# src/genai.py
import json
import os

try:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    USE_OPENAI = True
except ImportError:
    USE_OPENAI = False

PROMPT_TEMPLATE = """
You are a healthcare assistant AI.
Patient Risk Score: {risk:.2f}
Top Risk Factors: {factors}
Patient Info: {patient_json}

Generate 3 actionable care recommendations to reduce readmission risk.
Return a bullet list.
"""

def generate_recommendations(patient_info, risk_score, top_factors):
    """
    Generates human-readable recommendations based on top SHAP factors
    and patient information.
    
    Args:
        patient_info (dict): Patient's input data.
        risk_score (float): Predicted readmission probability.
        top_factors (list): Top features impacting prediction.

    Returns:
        str: Human-readable recommendation text.
    """
    risk_level = (
        "high" if risk_score > 0.6 else
        "medium" if risk_score > 0.3 else
        "low"
    )

    # Translate key factors into natural language
    factor_messages = []
    for factor in top_factors:
        if "discharge_disposition" in factor.lower():
            factor_messages.append("review the patient's discharge plan")
        elif "inpatient" in factor.lower():
            factor_messages.append("consider reducing inpatient visits or closer monitoring")
        elif "diabetesmed" in factor.lower():
            factor_messages.append("ensure diabetes medication compliance")
        elif "admission_type" in factor.lower():
            factor_messages.append("evaluate the admission type and care pathway")
        elif "diagnoses" in factor.lower():
            factor_messages.append("manage multiple diagnoses more effectively")
        else:
            factor_messages.append(f"monitor {factor.replace('_', ' ')} closely")

    recommendations = (
        f"Predicted readmission risk is **{risk_level.upper()}** ({risk_score:.2%}). "
        f"Key factors include {', '.join(factor_messages[:3])}. "
        f"Recommended next steps: schedule follow-up visits, optimize treatment plans, and address the top risk drivers."
    )
    
    return recommendations

