
# Model Audit Copilot

Model Audit Copilot is a modular, production-grade ML auditing framework that evaluates and validates machine learning models and data pipelines. It provides drift detection, fairness audits, data leakage checks, schema validation, SHAP-based explainability, and even model version comparisons. All of this is available through a Streamlit dashboard, a CLI, or batch automation mode.

Think of it as a "pre-flight checklist" for your machine learning pipeline.

---

## Project Structure

```
model-audit-copilot/
├── copilot/                 # All audit modules
│   ├── drift/              # Drift detection (KS, PSI)
│   ├── fairness/           # Fairness audits (group metrics)
│   ├── leakage/            # Leakage checks (target, duplicates, ID)
│   ├── schema/             # Schema/type mismatch checker
│   ├── outliers/           # IsolationForest-based anomaly detector
│   ├── explainability/     # SHAP summary + comparison
│   └── auditor/            # Core orchestrator class
├── dashboard/              # Streamlit frontend (multi-tab)
│   └── main_app.py         # Dashboard entry point
├── scripts/                # CLI and utility scripts
│   ├── audit_runner.py     # One-off audit runner
│   ├── batch_runner.py     # Multi-run audit manager
│   ├── report_utils.py     # Markdown auto-report writer
│   └── sql_loader.py       # Optional SQL-to-pandas loader
├── tests/                  # Pytest tests (optional)
├── data/                   # Local .csv and .db files (gitignored)
├── reports/                # Output reports (gitignored)
├── sample_data/            # Demo SHAP CSVs and sample inputs
├── README.md               # You're here
├── requirements.txt        # All dependencies
└── setup.py                # Makes the project pip-installable
```

---

## Features

### Drift Detection
- KS Test for numeric features
- PSI for stability monitoring
- Visual bar plot

### Fairness Audits
- MAE, RMSE, and Bias by group
- Sensitive feature can be race, gender, etc.

### Data Leakage Detection
- Detects:
  - Target-feature correlation
  - Duplicate rows
  - Train/test overlap
  - ID-like columns

### Schema Validation
- Detects missing, extra, or mismatched feature types

### SHAP Explainability
- TreeExplainer-based
- Global bar plot for feature importances

### Model Comparison
- Upload two SHAP CSVs
- See feature-level SHAP delta
- Visualized side-by-side

### CLI + Batch Mode
- Run audits from terminal
- Output markdown reports
- Batch multiple audits for versioning

### SQL Integration
- Read .db files via ipython-sql or sqlite3
- Demo query notebook included

---

## Getting Started

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Install as Editable Package (Optional)
```bash
pip install -e .
```

---

## Streamlit App

```bash
streamlit run dashboard/main_app.py
```

Tabs:
- Drift Detection
- Fairness Audit
- Outlier Detection
- Leakage Detection
- Schema Validation
- Explainability
- Model Comparison

Uploads:
- .csv files
- .pkl or .joblib models
- SHAP comparison .csvs

---

## CLI Usage

### One-Off Audit
```bash
PYTHONPATH=. python scripts/audit_runner.py \
  --reference data/ref_df.csv \
  --current data/cur_df.csv \
  --target true_cost \
  --pred predicted_cost \
  --group race \
  --output reports/audit_report.md
```

### Batch Mode (Multiple Datasets)
```bash
PYTHONPATH=. python scripts/batch_runner.py \
  --reference data/ref_df.csv \
  --current data/cur_df.csv \
  --target true_cost \
  --pred predicted_cost \
  --group race \
  --tag 20240513
```

---

## SQL Workflow (SQLite)

1. Create database:
```python
import sqlite3
import pandas as pd

df = pd.read_csv("data/cur_df.csv")
conn = sqlite3.connect("data/hospital_audit.db")
df.to_sql("hospital_costs", conn, index=False, if_exists="replace")
conn.close()
```

2. Explore via notebook:
```python
%load_ext sql
%sql sqlite:///data/hospital_audit.db

%%sql
SELECT * FROM hospital_costs LIMIT 5;
```

---

## Unit Tests (Optional)
Run all tests:
```bash
pytest tests/
```

---

## Sample Audit Report Output
See sample_data/example_audit_report.md

---

## License
MIT License — use freely with credit

---

## Author
Created by Toby Liu

If you use this project or learned from it, give it a star on GitHub!
