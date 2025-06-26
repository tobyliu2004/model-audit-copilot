"""Shared test fixtures and configuration."""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import tempfile
import os


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'numeric_col1': np.random.normal(100, 15, n_samples),
        'numeric_col2': np.random.exponential(50, n_samples),
        'numeric_col3': np.random.uniform(0, 1, n_samples),
        'categorical_col1': np.random.choice(['A', 'B', 'C'], n_samples),
        'categorical_col2': np.random.choice(['X', 'Y', 'Z'], n_samples, p=[0.5, 0.3, 0.2]),
        'target': np.random.normal(200, 30, n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def drifted_dataframe(sample_dataframe):
    """Create a DataFrame with drift from the sample."""
    df = sample_dataframe.copy()
    
    # Introduce drift in numeric columns
    df['numeric_col1'] = df['numeric_col1'] + 20  # Location shift
    df['numeric_col2'] = df['numeric_col2'] * 1.5  # Scale change
    
    # Introduce drift in categorical columns
    df.loc[df.index[:200], 'categorical_col1'] = 'D'  # New category
    
    return df


@pytest.fixture
def biased_dataframe():
    """Create a DataFrame with bias across groups."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create groups with different error patterns
    groups = np.random.choice(['GroupA', 'GroupB', 'GroupC'], n_samples, p=[0.5, 0.3, 0.2])
    true_values = np.random.normal(100, 20, n_samples)
    
    # Add group-specific bias
    predictions = true_values.copy()
    predictions[groups == 'GroupA'] += 5   # Overestimate GroupA
    predictions[groups == 'GroupB'] -= 10  # Underestimate GroupB
    predictions[groups == 'GroupC'] += np.random.normal(0, 5, sum(groups == 'GroupC'))
    
    return pd.DataFrame({
        'y_true': true_values,
        'y_pred': predictions,
        'group': groups,
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.uniform(0, 1, n_samples)
    })


@pytest.fixture
def leakage_dataframe():
    """Create a DataFrame with various types of leakage."""
    np.random.seed(42)
    n_samples = 500
    
    # Create base features
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = np.random.uniform(0, 1, n_samples)
    target = 2 * feature1 + 3 * feature2 + np.random.normal(0, 0.1, n_samples)
    
    # Add leakage features
    data = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'target': target,
        'leaked_feature': target * 0.99 + np.random.normal(0, 0.01, n_samples),  # High correlation
        'id_column': np.arange(n_samples),  # ID-like column
        'customer_id': [f'CUST_{i:06d}' for i in range(n_samples)],  # String ID
    })
    
    # Add some duplicate rows
    duplicates = data.iloc[:10].copy()
    data = pd.concat([data, duplicates], ignore_index=True)
    
    return data


@pytest.fixture
def sample_model():
    """Create a simple trained model for testing."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    
    # Write sample data
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'c', 'd', 'e'],
        'col3': [10.1, 20.2, 30.3, 40.4, 50.5]
    })
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


@pytest.fixture
def temp_db_file():
    """Create a temporary SQLite database."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


@pytest.fixture
def mock_config(monkeypatch):
    """Mock the configuration for testing."""
    from copilot.config import AuditConfig, DriftConfig, FairnessConfig
    
    config = AuditConfig(
        drift=DriftConfig(ks_threshold=0.05, psi_threshold=0.2),
        fairness=FairnessConfig(min_group_size=10, bias_threshold=0.1)
    )
    
    # Mock get_config to return our test config
    monkeypatch.setattr('copilot.config.get_config', lambda: config)
    
    return config