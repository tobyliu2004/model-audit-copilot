# copilot/explainability/shap_explainer.py

import shap
import pandas as pd
import matplotlib.pyplot as plt

def compute_shap_summary(model, X: pd.DataFrame, max_display=10):
    """
    Generates a SHAP summary plot using TreeExplainer only (safe for CPU environments).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_shap = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": pd.DataFrame(shap_values).abs().mean().values
    }).sort_values(by="mean_abs_shap", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(mean_shap["feature"], mean_shap["mean_abs_shap"], color="steelblue")
    ax.set_title("SHAP Summary (TreeExplainer)")
    plt.tight_layout()
    return fig
