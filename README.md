# Model Audit Copilot 🔍

<div align="center">

[![CI](https://github.com/yourusername/model-audit-copilot/workflows/CI/badge.svg)](https://github.com/yourusername/model-audit-copilot/actions)
[![codecov](https://codecov.io/gh/yourusername/model-audit-copilot/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/model-audit-copilot)
[![PyPI version](https://badge.fury.io/py/model-audit-copilot.svg)](https://badge.fury.io/py/model-audit-copilot)
[![Python Versions](https://img.shields.io/pypi/pyversions/model-audit-copilot.svg)](https://pypi.org/project/model-audit-copilot/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A comprehensive ML model auditing toolkit for detecting drift, fairness issues, and data quality problems.**

[Installation](#installation) • [Quick Start](#quick-start) • [Features](#features) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

## 🚀 Features

Model Audit Copilot provides a unified framework for auditing machine learning models in production:

- **🔄 Drift Detection**: Detect distribution changes using KS tests and PSI
- **⚖️ Fairness Analysis**: Identify bias across demographic groups
- **🔍 Data Leakage Detection**: Find target leakage, duplicates, and train/test overlap
- **📊 Schema Validation**: Ensure data consistency and type compliance
- **🎯 Outlier Detection**: Identify anomalous data points
- **🧠 Model Explainability**: SHAP-based feature importance analysis
- **📈 Interactive Dashboard**: Streamlit-based UI for visual exploration
- **⚡ CLI Tools**: Command-line interface for automation and CI/CD

## 📦 Installation

### Using pip

```bash
pip install model-audit-copilot
```

### From source

```bash
git clone https://github.com/yourusername/model-audit-copilot.git
cd model-audit-copilot
pip install -e .
```

### Using Docker

```bash
docker pull yourusername/model-audit-copilot:latest
```

## 🎯 Quick Start

### Command Line Interface

```bash
# Basic drift check
audit-runner --current data/production.csv --reference data/training.csv

# Comprehensive audit with fairness and leakage checks
audit-runner --current data/test.csv --reference data/train.csv \
  --target actual_price --pred predicted_price --group customer_segment \
  --output report.md

# Batch processing
batch-audit --config batch_config.yaml --output-dir batch_reports/
```

### Python API

```python
from copilot import AuditOrchestrator
import pandas as pd

# Load your data
train_df = pd.read_csv("train.csv")
prod_df = pd.read_csv("production.csv")

# Initialize auditor
auditor = AuditOrchestrator(train_df, prod_df)

# Run comprehensive audit
auditor.run_all_checks(
    target_col="target",
    prediction_col="prediction",
    group_col="demographic_group"
)

# Get results
results = auditor.get_results()
```

### Interactive Dashboard

```bash
# Launch the Streamlit dashboard
streamlit run dashboard/main_app.py

# Or use the CLI wrapper
audit-dashboard
```

## 📊 Core Features

### Drift Detection

Detect when your model's input data distribution changes:

```python
from copilot.drift import compare_datasets

drift_report = compare_datasets(
    reference_df=train_data,
    current_df=prod_data,
    method='ks'  # or 'psi'
)

# Visualize results
drift_report.plot()
```

### Fairness Auditing

Ensure your model treats all groups fairly:

```python
from copilot.fairness import audit_group_fairness

fairness_report = audit_group_fairness(
    y_true=df['actual'],
    y_pred=df['predicted'],
    sensitive_feature=df['demographic_group']
)
```

### Data Leakage Detection

Identify potential data leakage issues:

```python
from copilot.leakage import detect_data_leakage

leakage_report = detect_data_leakage(
    dataset=df,
    target_col='target',
    correlation_threshold=0.95
)
```

## 🔧 Configuration

Create an `audit_config.yaml` file to customize behavior:

```yaml
drift:
  ks_threshold: 0.05
  psi_threshold: 0.2
  psi_buckets: 10

fairness:
  min_group_size: 30
  bias_threshold: 0.1
  metrics: [mae, rmse, bias]

outlier:
  contamination: 0.01
  n_estimators: 100

logging:
  level: INFO
  file: audit.log
```

## 📈 Example Reports

The toolkit generates comprehensive markdown reports:

```markdown
# Model Audit Report

## Drift Analysis
- Features with drift: 3/10
- Most significant: `income` (KS=0.31, p<0.001)

## Fairness Analysis
- Max bias across groups: 0.15
- Worst performing group: Group_B (MAE=23.4)

## Data Quality
- Duplicate rows: 12 (0.1%)
- Potential ID columns: ['customer_id', 'transaction_id']
- High correlation with target: ['leaked_feature']
```

## 🏗️ Architecture

```
model-audit-copilot/
├── copilot/              # Core audit modules
│   ├── drift/            # Drift detection
│   ├── fairness/         # Bias detection
│   ├── leakage/          # Data leakage checks
│   ├── schema/           # Schema validation
│   ├── outliers/         # Anomaly detection
│   └── explainability/   # Model interpretability
├── dashboard/            # Streamlit web UI
├── scripts/              # CLI tools
└── tests/                # Test suite
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=copilot --cov-report=html

# Run specific test module
pytest tests/test_drift_detector.py -v
```

## 🐳 Docker Usage

Build and run with Docker:

```bash
# Build image
docker build -t model-audit-copilot .

# Run CLI audit
docker run -v $(pwd)/data:/data model-audit-copilot \
  audit-runner --current /data/test.csv --reference /data/train.csv

# Run dashboard
docker run -p 8501:8501 model-audit-copilot streamlit run dashboard/main_app.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/model-audit-copilot.git

# Create a feature branch
git checkout -b feature/your-feature

# Install in development mode
pip install -e ".[dev]"

# Make changes and run tests
pytest tests/

# Submit a pull request
```

## 📚 Documentation

- [Full Documentation](https://model-audit-copilot.readthedocs.io)
- [API Reference](https://model-audit-copilot.readthedocs.io/api)
- [Examples](./examples/)
- [FAQ](./docs/FAQ.md)

## 🛡️ Security

This toolkit prioritizes security:
- No direct SQL query execution (prevents SQL injection)
- Safe model serialization with joblib (no pickle)
- Path validation to prevent traversal attacks
- Input validation on all user data

Report security issues to: security@yourproject.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SHAP for model explainability
- scikit-learn for ML utilities
- Streamlit for the web interface
- The open-source community

## 📊 Roadmap

- [ ] Support for classification metrics in fairness audits
- [ ] Real-time monitoring capabilities
- [ ] Integration with MLflow and other ML platforms
- [ ] Advanced explainability methods
- [ ] Automated remediation suggestions

---

<div align="center">
Made with ❤️ by the Model Audit Copilot Team
</div>