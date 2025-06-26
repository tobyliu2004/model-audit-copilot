"""Model Audit Copilot - A comprehensive ML model auditing toolkit."""

__version__ = "1.0.0"
__author__ = "Model Audit Copilot Contributors"

from copilot.auditor.audit_orchestrator import AuditOrchestrator
from copilot.drift.drift_detector import DriftReport, compare_datasets
from copilot.fairness.fairness_audit import audit_group_fairness
from copilot.leakage.leakage_detector import detect_data_leakage
from copilot.schema.schema_validator import validate_schema
from copilot.outliers.outlier_detector import detect_outliers

__all__ = [
    "AuditOrchestrator",
    "DriftReport",
    "compare_datasets",
    "audit_group_fairness",
    "detect_data_leakage",
    "validate_schema",
    "detect_outliers",
]