# copilot/auditor/audit_orchestrator.py

from copilot.drift.drift_detector import compare_datasets
from copilot.fairness.fairness_audit import audit_group_fairness
import pandas as pd
from copilot.leakage.leakage_detector import (
    detect_target_leakage,
    detect_duplicate_and_id_leakage,
    detect_train_test_overlap
)
from copilot.schema.schema_validator import compare_schemas, find_missing_or_extra_columns
from copilot.explainability.shap_explainer import compute_shap_summary

class AuditOrchestrator:
    def __init__(self, reference_df, current_df):
        self.ref_df = reference_df
        self.cur_df = current_df
        self.results = {}
        self._drift_report = None  # <-- internal storage

    def run_drift_check(self, method="ks"):
        self._drift_report = compare_datasets(self.ref_df, self.cur_df, method=method)
        self.results["drift"] = self._drift_report.summary()
        return self
    
    def run_fairness_check(self, y_true, y_pred, group_feature: pd.Series):
        report = audit_group_fairness(y_true, y_pred, group_feature)
        self.results["fairness"] = report
        return self

    def get_fairness_report(self):
        return self.results.get("fairness", None)

    def get_drift_plot(self):
        if self._drift_report:
            return self._drift_report.plot(return_fig=True)
        return None

    def get_results(self):
        return self.results
    
    def run_leakage_check(self, dataset: pd.DataFrame, target_col: str):
        leakage_report = {}

        # 1. Target leakage
        target_leak = detect_target_leakage(dataset, target_col=target_col)
        leakage_report["target_leakage"] = target_leak

        # 2. Duplicate and ID-like columns
        dup_and_id = detect_duplicate_and_id_leakage(dataset)
        leakage_report["duplicate_rows"] = dup_and_id["duplicate_rows"]
        leakage_report["id_like_columns"] = dup_and_id["id_like_columns"]

        # 3. Train/test row overlap
        if self.ref_df is not None and self.cur_df is not None:
            overlap = detect_train_test_overlap(self.ref_df, self.cur_df)
            leakage_report["train_test_overlap"] = overlap

        self.results["leakage"] = leakage_report
        return self
    
    def get_leakage_report(self):
        return self.results.get("leakage", {})
    
    def run_schema_check(self):
        if self.ref_df is None or self.cur_df is None:
            raise ValueError("Schema comparison requires both reference and current datasets.")

        schema_report = compare_schemas(self.ref_df, self.cur_df)
        column_diff = find_missing_or_extra_columns(self.ref_df, self.cur_df)

        self.results["schema"] = {
            "schema_comparison": schema_report,
            "column_diff": column_diff
        }

        return self
    
    def get_schema_report(self):
        return self.results.get("schema", {})
    
    def run_explainability(self, model, X: pd.DataFrame):
        """
        Compute SHAP global summary plot and store the figure object.
        """
        fig = compute_shap_summary(model, X)
        self.results["shap_summary_plot"] = fig
        return self
    
    def get_shap_plot(self):
        return self.results.get("shap_summary_plot", None)






#Sample Usage
#auditor = AuditOrchestrator(ref_df, cur_df)

# Drift check
#auditor.run_drift_check(method="ks")

# Fairness check
#auditor.run_fairness_check(
#    y_true=df['true_cost'],
#    y_pred=df['predicted_cost'],
#    group_feature=df['race']
#)

# Retrieve results
#drift = auditor.get_results()["drift"]
#fairness = auditor.get_results()["fairness"]



#This is core infrastructure that lets all your modules — drift, fairness, explainability, leakage, etc. — plug into a single, reusable interface.

#🎯 Goals:
#Central interface for running multiple audits on a dataset

#Clean structure for adding more modules later

#Support output as unified report (e.g., dict, JSON, or DataFrame)

#This gives you:

#A cleaner API for your CLI, dashboard, and pipeline integration

#Future compatibility with auto-reporting, batch runs, and CI hooks