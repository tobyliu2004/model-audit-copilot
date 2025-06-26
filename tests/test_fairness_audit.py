"""Tests for fairness audit module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from copilot.fairness.fairness_audit import audit_group_fairness


class TestAuditGroupFairness:
    """Test cases for fairness auditing."""
    
    def test_basic_fairness_audit(self, biased_dataframe):
        """Test basic fairness audit functionality."""
        df = biased_dataframe
        
        report = audit_group_fairness(
            y_true=df['y_true'],
            y_pred=df['y_pred'],
            sensitive_feature=df['group']
        )
        
        assert isinstance(report, pd.DataFrame)
        assert len(report) == 3  # Three groups
        assert all(col in report.columns for col in ['count', 'mae', 'rmse', 'bias'])
    
    def test_group_metrics_calculation(self):
        """Test correct calculation of group metrics."""
        # Create controlled data
        y_true = np.array([100, 100, 100, 200, 200, 200])
        y_pred = np.array([110, 110, 110, 190, 190, 190])
        groups = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
        
        report = audit_group_fairness(y_true, y_pred, groups)
        
        # Group A: overestimated by 10
        assert report.loc[report['group'] == 'A', 'bias'].values[0] == 10.0
        assert report.loc[report['group'] == 'A', 'mae'].values[0] == 10.0
        
        # Group B: underestimated by 10
        assert report.loc[report['group'] == 'B', 'bias'].values[0] == -10.0
        assert report.loc[report['group'] == 'B', 'mae'].values[0] == 10.0
    
    def test_single_group(self):
        """Test with only one group."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.1, 3.1, 4.1, 5.1]
        groups = ['A'] * 5
        
        report = audit_group_fairness(y_true, y_pred, groups)
        
        assert len(report) == 1
        assert report.iloc[0]['group'] == 'A'
        assert report.iloc[0]['count'] == 5
    
    def test_many_groups(self):
        """Test with many groups."""
        n_samples = 1000
        n_groups = 50
        
        np.random.seed(42)
        y_true = np.random.normal(100, 20, n_samples)
        y_pred = y_true + np.random.normal(0, 5, n_samples)
        groups = np.random.choice([f'Group_{i}' for i in range(n_groups)], n_samples)
        
        report = audit_group_fairness(y_true, y_pred, groups)
        
        assert len(report) <= n_groups
        assert all(report['count'] > 0)
    
    def test_with_missing_values(self):
        """Test handling of missing values."""
        y_true = [1, 2, np.nan, 4, 5]
        y_pred = [1.1, 2.1, 3.1, np.nan, 5.1]
        groups = ['A', 'A', 'B', 'B', 'B']
        
        report = audit_group_fairness(y_true, y_pred, groups)
        
        # Should handle NaN values gracefully
        assert isinstance(report, pd.DataFrame)
        assert report.loc[report['group'] == 'A', 'count'].values[0] == 2
        assert report.loc[report['group'] == 'B', 'count'].values[0] == 1  # Only one valid pair
    
    def test_input_validation(self):
        """Test input validation."""
        # Different lengths
        with pytest.raises(ValueError):
            audit_group_fairness([1, 2, 3], [1, 2], ['A', 'B', 'C'])
        
        # Empty inputs
        with pytest.raises(ValueError):
            audit_group_fairness([], [], [])
    
    def test_various_input_types(self):
        """Test with various input types."""
        # Lists
        report1 = audit_group_fairness(
            [1, 2, 3], [1.1, 2.1, 3.1], ['A', 'B', 'A']
        )
        
        # NumPy arrays
        report2 = audit_group_fairness(
            np.array([1, 2, 3]), 
            np.array([1.1, 2.1, 3.1]), 
            np.array(['A', 'B', 'A'])
        )
        
        # Pandas Series
        df = pd.DataFrame({
            'y_true': [1, 2, 3],
            'y_pred': [1.1, 2.1, 3.1],
            'group': ['A', 'B', 'A']
        })
        report3 = audit_group_fairness(
            df['y_true'], df['y_pred'], df['group']
        )
        
        # All should produce similar results
        assert len(report1) == len(report2) == len(report3)
    
    def test_numeric_groups(self):
        """Test with numeric group identifiers."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.1, 3.1, 4.1, 5.1]
        groups = [0, 0, 1, 1, 2]
        
        report = audit_group_fairness(y_true, y_pred, groups)
        
        assert len(report) == 3
        assert set(report['group']) == {0, 1, 2}
    
    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 5]
        groups = ['A', 'A', 'B', 'B', 'B']
        
        report = audit_group_fairness(y_true, y_pred, groups)
        
        # All metrics should be zero
        assert all(report['mae'] == 0)
        assert all(report['rmse'] == 0)
        assert all(report['bias'] == 0)
    
    def test_extreme_bias(self):
        """Test detection of extreme bias."""
        # Group A: severely overestimated
        # Group B: severely underestimated
        y_true = [10] * 5 + [10] * 5
        y_pred = [20] * 5 + [5] * 5
        groups = ['A'] * 5 + ['B'] * 5
        
        report = audit_group_fairness(y_true, y_pred, groups)
        
        assert report.loc[report['group'] == 'A', 'bias'].values[0] == 10.0
        assert report.loc[report['group'] == 'B', 'bias'].values[0] == -5.0
        
        # Check that RMSE reflects the error magnitude
        assert report.loc[report['group'] == 'A', 'rmse'].values[0] == 10.0
        assert report.loc[report['group'] == 'B', 'rmse'].values[0] == 5.0
    
    @patch('copilot.fairness.fairness_audit.logger')
    def test_small_group_warning(self, mock_logger):
        """Test warning for small group sizes."""
        # Create groups with varying sizes
        y_true = list(range(100))
        y_pred = [x + 0.1 for x in y_true]
        groups = ['Large'] * 50 + ['Medium'] * 30 + ['Small'] * 5 + ['Tiny'] * 2 + ['Single'] * 1 + ['Rest'] * 12
        
        report = audit_group_fairness(y_true, y_pred, groups)
        
        # Should warn about small groups
        warning_calls = [call for call in mock_logger.warning.call_args_list]
        assert len(warning_calls) > 0  # Should have warnings about small groups
    
    def test_group_ordering(self):
        """Test that groups are ordered correctly in output."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.1, 3.1, 4.1, 5.1]
        groups = ['C', 'A', 'B', 'A', 'C']
        
        report = audit_group_fairness(y_true, y_pred, groups)
        
        # Groups should be in sorted order
        assert list(report['group']) == ['A', 'B', 'C']
    
    def test_large_scale_performance(self):
        """Test performance with large dataset."""
        n_samples = 10000
        n_groups = 100
        
        np.random.seed(42)
        y_true = np.random.normal(100, 20, n_samples)
        y_pred = y_true + np.random.normal(0, 5, n_samples)
        groups = np.random.choice(range(n_groups), n_samples)
        
        # Should complete without error
        report = audit_group_fairness(y_true, y_pred, groups)
        
        assert len(report) == n_groups
        assert report['count'].sum() == n_samples