# scripts/batch_runner.py

import os
import argparse
import pandas as pd
from copilot.auditor.audit_orchestrator import AuditOrchestrator
from copilot.outliers.outlier_detector import detect_outliers
from scripts.reports_utils import generate_audit_markdown, save_report

def run_batch_audit(ref_path, cur_path, target, pred, group, out_dir, tag):
    # Load data
    ref_df = pd.read_csv(ref_path)
    cur_df = pd.read_csv(cur_path)

    # Create output directory
    run_dir = os.path.join(out_dir, f"audit_{tag}")
    os.makedirs(run_dir, exist_ok=True)

    # Run audits
    auditor = AuditOrchestrator(ref_df, cur_df)
    auditor.run_drift_check(method="ks")
    auditor.run_fairness_check(y_true=cur_df[target], y_pred=cur_df[pred], group_feature=cur_df[group])
    auditor.run_leakage_check(dataset=cur_df, target_col=target)
    auditor.run_schema_check()
    outliers = detect_outliers(cur_df)

    # Save report
    markdown = generate_audit_markdown(
        drift=auditor.get_results().get("drift"),
        fairness=auditor.get_results().get("fairness"),
        outliers=outliers
    )
    report_path = os.path.join(run_dir, "audit_report.md")
    save_report(markdown, report_path)
    print(f"✓ Audit saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch model audit.")
    parser.add_argument("--reference", required=True, help="Reference (train) CSV path")
    parser.add_argument("--current", required=True, help="Current (test/predict) CSV path")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--pred", required=True, help="Prediction column name")
    parser.add_argument("--group", required=True, help="Sensitive group column name")
    parser.add_argument("--output", default="batch_reports", help="Output folder")
    parser.add_argument("--tag", default="v1", help="Audit run tag (e.g. date or version)")

    args = parser.parse_args()
    run_batch_audit(
        ref_path=args.reference,
        cur_path=args.current,
        target=args.target,
        pred=args.pred,
        group=args.group,
        out_dir=args.output,
        tag=args.tag
    )
