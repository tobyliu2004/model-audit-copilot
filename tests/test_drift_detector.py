import pytest
import pandas as pd
import numpy as np
from copilot.drift.drift_detector import compare_datasets

@pytest.fixture
def simulated_data():
    np.random.seed(42)
    reference = pd.DataFrame({
        "age": np.random.normal(40, 10, 1000),
        "income": np.random.normal(50000, 8000, 1000),
    })

    # age distribution changed
    current = pd.DataFrame({
        "age": np.random.normal(45, 15, 1000),
        "income": np.random.normal(50000, 8000, 1000),
    })

    return reference, current

def test_drift_detection_summary(simulated_data):
    ref_df, cur_df = simulated_data
    report = compare_datasets(ref_df, cur_df)
    summary = report.summary()

    assert "age" in summary.index
    assert "ks_stat" in summary.columns
    assert summary.loc["age", "drift_detected"] is True
    assert summary.loc["income", "drift_detected"] is False
