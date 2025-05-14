# scripts/audit_runner.py

import argparse
import pandas as pd
from copilot.auditor.audit_orchestrator import AuditOrchestrator
from copilot.outliers.outlier_detector import detect_outliers
from scripts.reports_utils import generate_audit_markdown, save_report

def main(args):
    # Load datasets
    ref_df = pd.read_csv(args.reference) if args.reference else None
    cur_df = pd.read_csv(args.current)

    # Run audits
    auditor = AuditOrchestrator(ref_df, cur_df)
    auditor.run_drift_check(method=args.method)

    if args.target:
        auditor.run_fairness_check(
            y_true=cur_df[args.target],
            y_pred=cur_df[args.pred],
            group_feature=cur_df[args.group]
        )
        auditor.run_leakage_check(dataset=cur_df, target_col=args.target)

    # Detect outliers
    outliers = detect_outliers(cur_df, n_outliers=args.outliers)

    # Generate report
    markdown = generate_audit_markdown(
        drift=auditor.get_results().get("drift"),
        fairness=auditor.get_results().get("fairness"),
        outliers=outliers
    )
    save_report(markdown, args.output)
    print(f"Audit complete. Report saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model audit checks from CLI")
    parser.add_argument("--reference", help="Reference CSV (e.g., training data)")
    parser.add_argument("--current", required=True, help="Current CSV (e.g., test or production data)")
    parser.add_argument("--method", default="ks", choices=["ks", "psi"], help="Drift detection method")
    parser.add_argument("--target", help="Target column (required for fairness/leakage)")
    parser.add_argument("--pred", help="Prediction column")
    parser.add_argument("--group", help="Sensitive group column")
    parser.add_argument("--outliers", type=int, default=10, help="Number of top outliers to report")
    parser.add_argument("--output", default="audit_report.md", help="Output path for markdown report")

    args = parser.parse_args()
    main(args)