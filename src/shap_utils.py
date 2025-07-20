import shap
import numpy as np
import matplotlib.pyplot as plt

def explain_single_patient(model, preprocessor, df_row, feature_names):
    """
    Generate SHAP values for a single patient (row).
    Returns a dict: { "plot": fig, "top_factors": [..] }
    """
    # Transform row
    row_t = preprocessor.transform(df_row)

    # Use TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row_t)

    # Top factors (positive contributions)
    contrib = list(zip(feature_names, shap_values[0]))
    contrib_sorted = sorted(contrib, key=lambda x: x[1], reverse=True)
    top_factors = [f for f, v in contrib_sorted[:5]]

    # SHAP force plot (matplotlib fallback)
    fig, ax = plt.subplots(figsize=(6, 2))
    shap.waterfall_plot = shap.plots._waterfall.waterfall_legacy  # compatibility
    shap.waterfall_plot(
        explainer.expected_value,
        shap_values[0],
        feature_names=feature_names,
        max_display=10,
        show=False
    )
    plt.tight_layout()

    return {"plot": fig, "top_factors": top_factors}
