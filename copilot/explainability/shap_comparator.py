# copilot/explainability/shap_comparator.py

import shap
import pandas as pd

def compute_shap_values(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    mean_abs = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": shap_values.abs.mean(0).values
    })
    return mean_abs.sort_values(by="mean_abs_shap", ascending=False)

def compare_shap_importance(model_a, model_b, X, top_n=10):
    shap_a = compute_shap_values(model_a, X).set_index("feature")
    shap_b = compute_shap_values(model_b, X).set_index("feature")

    comparison = shap_a.join(shap_b, lsuffix="_a", rsuffix="_b").fillna(0)
    comparison["delta"] = comparison["mean_abs_shap_b"] - comparison["mean_abs_shap_a"]
    comparison = comparison.sort_values("delta", ascending=False)

    return comparison.head(top_n).reset_index()
