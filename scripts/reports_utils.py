# scripts/report_utils.py

from datetime import datetime
import pandas as pd

def generate_audit_markdown(drift=None, fairness=None, outliers=None):
    lines = []
    lines.append(f"# Model Audit Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("---")

    if drift is not None:
        lines.append("## Drift Detection Summary")
        drifted = drift['drift_detected'].sum()
        total = drift.shape[0]
        lines.append(f"**{drifted} of {total} features** showed significant drift.\n")
        lines.append(drift.to_markdown(index=True))
        lines.append("")

    if fairness is not None:
        lines.append("## Fairness Audit Summary")
        lines.append(fairness.to_markdown(index=False))
        lines.append("")

    if outliers is not None:
        lines.append("## Outlier Detection Summary")
        lines.append(outliers.to_markdown(index=False))
        lines.append("")

    return "\n".join(lines)

def save_report(markdown_str, path="audit_report.md"):
    with open(path, "w") as f:
        f.write(markdown_str)

#EXAMPLE USAGE
#from scripts.report_utils import generate_audit_markdown, save_report

#markdown = generate_audit_markdown(
#    drift=auditor.get_results().get("drift"),
#    fairness=auditor.get_results().get("fairness"),
#    outliers=detect_outliers(cur_df)
#)

#save_report(markdown, "audit_report.md")

#CONVERT TO PDF USING BASH
#pandoc audit_report.md -o audit_report.pdf
