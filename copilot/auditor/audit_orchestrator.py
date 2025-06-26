"""Central orchestrator for running comprehensive model audits."""

import pandas as pd
from typing import Optional, Dict, Any, Union, List
import numpy as np
import logging

from copilot.config import get_config
from copilot.drift.drift_detector import compare_datasets
from copilot.fairness.fairness_audit import audit_group_fairness
from copilot.leakage.leakage_detector import (
    detect_target_leakage,
    detect_duplicate_and_id_leakage,
    detect_train_test_overlap
)
from copilot.schema.schema_validator import compare_schemas, find_missing_or_extra_columns
from copilot.explainability.shap_explainer import compute_shap_summary

logger = logging.getLogger(__name__)


class AuditOrchestrator:
    """
    Central orchestrator for running comprehensive model audits.
    
    This class provides a unified interface for running various audit checks
    including drift detection, fairness analysis, data leakage detection,
    schema validation, and model explainability.
    
    Attributes:
        ref_df: Reference dataset (e.g., training data)
        cur_df: Current dataset (e.g., production data)
        results: Dictionary storing all audit results
    """
    
    def __init__(self, reference_df: Optional[pd.DataFrame], current_df: pd.DataFrame):
        """
        Initialize the audit orchestrator.
        
        Args:
            reference_df: Reference dataset (can be None for single-dataset audits)
            current_df: Current dataset to audit
            
        Raises:
            ValueError: If current_df is None or empty
        """
        if current_df is None or current_df.empty:
            raise ValueError("Current dataset cannot be None or empty")
            
        self.ref_df = reference_df
        self.cur_df = current_df
        self.results = {}
        self._drift_report = None
        self.config = get_config()
        
        logger.info(
            f"Initialized AuditOrchestrator with "
            f"reference_df shape: {reference_df.shape if reference_df is not None else 'None'}, "
            f"current_df shape: {current_df.shape}"
        )

    def run_drift_check(self, method: str = "ks", columns: Optional[List[str]] = None) -> 'AuditOrchestrator':
        """
        Run drift detection between reference and current datasets.
        
        Args:
            method: Drift detection method ('ks' or 'psi')
            columns: Specific columns to check (None = all columns)
            
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If reference dataset is not provided
        """
        if self.ref_df is None:
            raise ValueError("Drift check requires a reference dataset")
        
        logger.info(f"Running drift check with method: {method}")
        
        try:
            self._drift_report = compare_datasets(
                self.ref_df, 
                self.cur_df, 
                method=method,
                columns=columns
            )
            self.results["drift"] = self._drift_report.summary()
            
            # Log summary
            drifted_features = self._drift_report.get_drifted_features()
            logger.info(f"Drift check completed. Found drift in {len(drifted_features)} features")
            if drifted_features:
                logger.warning(f"Features with drift: {drifted_features[:10]}...")
                
        except Exception as e:
            logger.error(f"Error during drift check: {str(e)}", exc_info=True)
            raise
            
        return self
    
    def run_fairness_check(
        self, 
        y_true: Union[pd.Series, np.ndarray, list],
        y_pred: Union[pd.Series, np.ndarray, list],
        group_feature: Union[pd.Series, np.ndarray, list]
    ) -> 'AuditOrchestrator':
        """
        Run fairness audit across different groups.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            group_feature: Categorical feature defining groups
            
        Returns:
            self for method chaining
        """
        logger.info("Running fairness check")
        
        # Input validation
        if len(y_true) != len(y_pred) or len(y_true) != len(group_feature):
            raise ValueError("All inputs must have the same length")
        
        try:
            report = audit_group_fairness(y_true, y_pred, group_feature)
            self.results["fairness"] = report
            
            # Log summary
            num_groups = len(report) if isinstance(report, pd.DataFrame) else 0
            logger.info(f"Fairness check completed for {num_groups} groups")
            
            if isinstance(report, pd.DataFrame) and 'bias' in report.columns:
                max_bias = report['bias'].abs().max()
                if max_bias > self.config.fairness.bias_threshold:
                    logger.warning(f"High bias detected: max absolute bias = {max_bias:.3f}")
                    
        except Exception as e:
            logger.error(f"Error during fairness check: {str(e)}", exc_info=True)
            raise
            
        return self

    def run_leakage_check(self, dataset: Optional[pd.DataFrame] = None, target_col: str = None) -> 'AuditOrchestrator':
        """
        Run comprehensive data leakage detection.
        
        Args:
            dataset: Dataset to check (uses current_df if None)
            target_col: Target column name for leakage detection
            
        Returns:
            self for method chaining
        """
        dataset = dataset if dataset is not None else self.cur_df
        logger.info(f"Running leakage check{' with target column: ' + target_col if target_col else ''}")
        
        leakage_report = {}
        
        try:
            # 1. Target leakage
            if target_col:
                logger.debug("Checking for target leakage")
                target_leak = detect_target_leakage(dataset, target_col=target_col)
                leakage_report["target_leakage"] = target_leak
                
                if target_leak.get("high_correlation_features"):
                    logger.warning(
                        f"Found {len(target_leak['high_correlation_features'])} "
                        f"features with high correlation to target"
                    )

            # 2. Duplicate and ID-like columns
            logger.debug("Checking for duplicates and ID-like columns")
            dup_and_id = detect_duplicate_and_id_leakage(dataset)
            leakage_report["duplicate_rows"] = dup_and_id["duplicate_rows"]
            leakage_report["id_like_columns"] = dup_and_id["id_like_columns"]
            
            if dup_and_id["duplicate_rows"] > 0:
                logger.warning(f"Found {dup_and_id['duplicate_rows']} duplicate rows")
            if dup_and_id["id_like_columns"]:
                logger.warning(f"Found potential ID columns: {dup_and_id['id_like_columns']}")

            # 3. Train/test row overlap
            if self.ref_df is not None and self.cur_df is not None:
                logger.debug("Checking for train/test overlap")
                overlap = detect_train_test_overlap(self.ref_df, self.cur_df)
                leakage_report["train_test_overlap"] = overlap
                
                if overlap.get("overlapping_rows", 0) > 0:
                    logger.warning(
                        f"Found {overlap['overlapping_rows']} overlapping rows "
                        f"between reference and current datasets"
                    )

            self.results["leakage"] = leakage_report
            logger.info("Leakage check completed")
            
        except Exception as e:
            logger.error(f"Error during leakage check: {str(e)}", exc_info=True)
            raise
            
        return self
    
    def run_schema_check(self) -> 'AuditOrchestrator':
        """
        Run schema validation between reference and current datasets.
        
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If reference dataset is not provided
        """
        if self.ref_df is None:
            raise ValueError("Schema comparison requires both reference and current datasets")
        
        logger.info("Running schema check")
        
        try:
            schema_report = compare_schemas(self.ref_df, self.cur_df)
            column_diff = find_missing_or_extra_columns(self.ref_df, self.cur_df)

            self.results["schema"] = {
                "schema_comparison": schema_report,
                "column_diff": column_diff
            }
            
            # Log summary
            if column_diff.get("missing_in_current"):
                logger.warning(
                    f"Missing columns in current dataset: "
                    f"{column_diff['missing_in_current']}"
                )
            if column_diff.get("extra_in_current"):
                logger.warning(
                    f"Extra columns in current dataset: "
                    f"{column_diff['extra_in_current']}"
                )
            
            type_mismatches = sum(
                1 for v in schema_report.values() 
                if v.get("type_match") is False
            )
            if type_mismatches > 0:
                logger.warning(f"Found {type_mismatches} column type mismatches")
            
            logger.info("Schema check completed")
            
        except Exception as e:
            logger.error(f"Error during schema check: {str(e)}", exc_info=True)
            raise

        return self
    
    def run_explainability(self, model: Any, X: pd.DataFrame) -> 'AuditOrchestrator':
        """
        Compute SHAP values for model explainability.
        
        Args:
            model: Trained model object
            X: Feature dataset
            
        Returns:
            self for method chaining
        """
        logger.info(f"Running explainability analysis on {X.shape[0]} samples")
        
        try:
            fig = compute_shap_summary(model, X)
            self.results["shap_summary_plot"] = fig
            logger.info("Explainability analysis completed")
            
        except Exception as e:
            logger.error(f"Error during explainability analysis: {str(e)}", exc_info=True)
            raise
            
        return self
    
    def run_all_checks(
        self,
        drift_method: str = "ks",
        target_col: Optional[str] = None,
        prediction_col: Optional[str] = None,
        group_col: Optional[str] = None
    ) -> 'AuditOrchestrator':
        """
        Run all applicable audit checks.
        
        Args:
            drift_method: Method for drift detection
            target_col: Target column name
            prediction_col: Prediction column name
            group_col: Group column for fairness analysis
            
        Returns:
            self for method chaining
        """
        logger.info("Running comprehensive audit")
        
        # Drift check (if reference data available)
        if self.ref_df is not None:
            try:
                self.run_drift_check(method=drift_method)
            except Exception as e:
                logger.error(f"Drift check failed: {e}")
                if not self.config.continue_on_error:
                    raise
        
        # Schema check (if reference data available)
        if self.ref_df is not None:
            try:
                self.run_schema_check()
            except Exception as e:
                logger.error(f"Schema check failed: {e}")
                if not self.config.continue_on_error:
                    raise
        
        # Leakage check
        try:
            self.run_leakage_check(target_col=target_col)
        except Exception as e:
            logger.error(f"Leakage check failed: {e}")
            if not self.config.continue_on_error:
                raise
        
        # Fairness check (if required columns provided)
        if target_col and prediction_col and group_col:
            if all(col in self.cur_df.columns for col in [target_col, prediction_col, group_col]):
                try:
                    self.run_fairness_check(
                        y_true=self.cur_df[target_col],
                        y_pred=self.cur_df[prediction_col],
                        group_feature=self.cur_df[group_col]
                    )
                except Exception as e:
                    logger.error(f"Fairness check failed: {e}")
                    if not self.config.continue_on_error:
                        raise
        
        logger.info("Comprehensive audit completed")
        return self
    
    # Getter methods
    def get_drift_report(self) -> Optional[pd.DataFrame]:
        """Get drift detection results."""
        return self.results.get("drift")
    
    def get_fairness_report(self) -> Optional[pd.DataFrame]:
        """Get fairness audit results."""
        return self.results.get("fairness")
    
    def get_leakage_report(self) -> Dict[str, Any]:
        """Get data leakage detection results."""
        return self.results.get("leakage", {})
    
    def get_schema_report(self) -> Dict[str, Any]:
        """Get schema validation results."""
        return self.results.get("schema", {})
    
    def get_drift_plot(self):
        """Get drift visualization plot."""
        if self._drift_report:
            return self._drift_report.plot(return_fig=True)
        return None
    
    def get_shap_plot(self):
        """Get SHAP summary plot."""
        return self.results.get("shap_summary_plot")
    
    def get_results(self) -> Dict[str, Any]:
        """Get all audit results."""
        return self.results
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a high-level summary of all audit results.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "audits_run": list(self.results.keys()),
            "issues_found": {}
        }
        
        # Drift summary
        if "drift" in self.results and self._drift_report:
            drifted = self._drift_report.get_drifted_features()
            summary["issues_found"]["drift"] = {
                "features_with_drift": len(drifted),
                "total_features_checked": len(self._drift_report.results)
            }
        
        # Fairness summary
        if "fairness" in self.results:
            fairness = self.results["fairness"]
            if isinstance(fairness, pd.DataFrame) and 'bias' in fairness.columns:
                summary["issues_found"]["fairness"] = {
                    "groups_analyzed": len(fairness),
                    "max_absolute_bias": fairness['bias'].abs().max()
                }
        
        # Leakage summary
        if "leakage" in self.results:
            leakage = self.results["leakage"]
            summary["issues_found"]["leakage"] = {
                "duplicate_rows": leakage.get("duplicate_rows", 0),
                "id_like_columns": len(leakage.get("id_like_columns", [])),
                "high_correlation_features": len(
                    leakage.get("target_leakage", {}).get("high_correlation_features", [])
                )
            }
        
        # Schema summary
        if "schema" in self.results:
            schema = self.results["schema"]
            column_diff = schema.get("column_diff", {})
            summary["issues_found"]["schema"] = {
                "missing_columns": len(column_diff.get("missing_in_current", [])),
                "extra_columns": len(column_diff.get("extra_in_current", [])),
                "type_mismatches": sum(
                    1 for v in schema.get("schema_comparison", {}).values() 
                    if v.get("type_match") is False
                )
            }
        
        return summary