"""Tests for drift detection module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from copilot.drift.drift_detector import (
    DriftReport, calculate_psi, compare_datasets
)


class TestDriftReport:
    """Test cases for DriftReport class."""
    
    def test_drift_report_initialization(self):
        """Test DriftReport initialization and validation."""
        results = {
            'col1': {'ks_stat': 0.1, 'p_value': 0.3, 'drift_detected': False},
            'col2': {'ks_stat': 0.3, 'p_value': 0.01, 'drift_detected': True}
        }
        
        report = DriftReport(results)
        assert report.results == results
    
    def test_drift_report_invalid_input(self):
        """Test DriftReport with invalid input."""
        with pytest.raises(TypeError):
            DriftReport("not a dict")
        
        with pytest.raises(ValueError):
            DriftReport({'col1': "not a dict"})
    
    def test_summary(self):
        """Test summary generation."""
        results = {
            'col1': {'ks_stat': 0.1, 'p_value': 0.3, 'drift_detected': False},
            'col2': {'ks_stat': 0.3, 'p_value': 0.01, 'drift_detected': True}
        }
        
        report = DriftReport(results)
        summary = report.summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert list(summary.index) == ['col1', 'col2']
        assert 'drift_detected' in summary.columns
    
    def test_get_drifted_features(self):
        """Test getting list of drifted features."""
        results = {
            'col1': {'drift_detected': False},
            'col2': {'drift_detected': True},
            'col3': {'drift_detected': True},
            'col4': {'drift_detected': False}
        }
        
        report = DriftReport(results)
        drifted = report.get_drifted_features()
        
        assert set(drifted) == {'col2', 'col3'}
        assert len(drifted) == 2
    
    def test_plot_with_ks_results(self):
        """Test plot generation with KS results."""
        results = {
            'col1': {'ks_stat': 0.1, 'drift_detected': False},
            'col2': {'ks_stat': 0.3, 'drift_detected': True}
        }
        
        report = DriftReport(results)
        fig = report.plot(return_fig=True)
        
        assert fig is not None
        assert len(fig.axes) == 1
    
    def test_plot_with_psi_results(self):
        """Test plot generation with PSI results."""
        results = {
            'col1': {'psi': 0.1, 'drift_detected': False},
            'col2': {'psi': 0.5, 'drift_detected': True}
        }
        
        report = DriftReport(results)
        fig = report.plot(return_fig=True)
        
        assert fig is not None


class TestCalculatePSI:
    """Test cases for PSI calculation."""
    
    def test_psi_no_drift(self):
        """Test PSI calculation with no drift."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        
        # Same distribution should have low PSI
        psi = calculate_psi(data, data)
        assert psi < 0.1
    
    def test_psi_with_drift(self):
        """Test PSI calculation with drift."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(2, 1, 1000)  # Shifted distribution
        
        psi = calculate_psi(expected, actual)
        assert psi > 0.2  # Should detect significant drift
    
    def test_psi_categorical_data(self):
        """Test PSI calculation with categorical data."""
        expected = ['A'] * 500 + ['B'] * 300 + ['C'] * 200
        actual = ['A'] * 200 + ['B'] * 300 + ['C'] * 500  # Changed distribution
        
        psi = calculate_psi(expected, actual)
        assert psi > 0
    
    def test_psi_with_nulls(self):
        """Test PSI calculation handles null values."""
        expected = pd.Series([1, 2, 3, np.nan, 5])
        actual = pd.Series([1, np.nan, 3, 4, 5])
        
        # Should not raise error
        psi = calculate_psi(expected, actual)
        assert isinstance(psi, float)
    
    def test_psi_invalid_buckets(self):
        """Test PSI with invalid bucket count."""
        data = [1, 2, 3, 4, 5]
        
        with pytest.raises(ValueError):
            calculate_psi(data, data, buckets=1)
    
    @patch('copilot.drift.drift_detector.logger')
    def test_psi_small_sample_warning(self, mock_logger):
        """Test PSI warns on small samples."""
        small_data = [1, 2, 3]
        calculate_psi(small_data, small_data, buckets=5)
        
        # Should log warning about small sample size
        mock_logger.warning.assert_called()


class TestCompareDatasets:
    """Test cases for dataset comparison."""
    
    def test_compare_datasets_basic(self, sample_dataframe, mock_config):
        """Test basic dataset comparison."""
        ref_df = sample_dataframe
        cur_df = sample_dataframe.copy()
        
        report = compare_datasets(ref_df, cur_df, method='ks')
        
        assert isinstance(report, DriftReport)
        assert len(report.results) == len(ref_df.select_dtypes(include=[np.number]).columns)
    
    def test_compare_datasets_with_drift(self, sample_dataframe, drifted_dataframe, mock_config):
        """Test comparison detects drift."""
        report = compare_datasets(sample_dataframe, drifted_dataframe, method='ks')
        
        drifted_features = report.get_drifted_features()
        assert len(drifted_features) > 0
        assert 'numeric_col1' in drifted_features  # We added drift to this column
    
    def test_compare_datasets_psi_method(self, sample_dataframe, mock_config):
        """Test comparison using PSI method."""
        ref_df = sample_dataframe
        cur_df = sample_dataframe.copy()
        
        report = compare_datasets(ref_df, cur_df, method='psi')
        
        assert isinstance(report, DriftReport)
        # PSI works on all column types
        assert len(report.results) == len(ref_df.columns)
    
    def test_compare_specific_columns(self, sample_dataframe, mock_config):
        """Test comparison of specific columns only."""
        ref_df = sample_dataframe
        cur_df = sample_dataframe.copy()
        
        columns = ['numeric_col1', 'numeric_col2']
        report = compare_datasets(ref_df, cur_df, method='ks', columns=columns)
        
        assert len(report.results) == 2
        assert set(report.results.keys()) == set(columns)
    
    def test_compare_missing_columns(self, sample_dataframe, mock_config):
        """Test comparison with missing columns."""
        ref_df = sample_dataframe
        cur_df = sample_dataframe.drop(columns=['numeric_col1'])
        
        report = compare_datasets(ref_df, cur_df, method='ks')
        
        # Should only compare common columns
        assert 'numeric_col1' not in report.results
    
    def test_compare_invalid_method(self, sample_dataframe, mock_config):
        """Test comparison with invalid method."""
        ref_df = sample_dataframe
        cur_df = sample_dataframe.copy()
        
        with pytest.raises(ValueError):
            compare_datasets(ref_df, cur_df, method='invalid')
    
    def test_compare_non_numeric_ks(self, mock_config):
        """Test KS method skips non-numeric columns."""
        ref_df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'categorical': ['A', 'B', 'C', 'D', 'E']
        })
        cur_df = ref_df.copy()
        
        report = compare_datasets(ref_df, cur_df, method='ks')
        
        # Should skip categorical column for KS test
        assert report.results['categorical'].get('error') is not None
        assert report.results['numeric'].get('ks_stat') is not None
    
    @patch('copilot.drift.drift_detector.logger')
    def test_compare_with_errors(self, mock_logger, mock_config):
        """Test comparison handles errors gracefully."""
        # Create DataFrame that will cause errors
        ref_df = pd.DataFrame({'col1': [1, 2, 3]})
        cur_df = pd.DataFrame({'col1': ['a', 'b', 'c']})  # Type mismatch
        
        report = compare_datasets(ref_df, cur_df, method='ks')
        
        # Should log error and continue
        assert 'error' in report.results['col1']
        mock_logger.error.assert_called()
    
    def test_compare_empty_columns(self, mock_config):
        """Test comparison with empty columns."""
        ref_df = pd.DataFrame({'col1': [np.nan] * 5})
        cur_df = pd.DataFrame({'col1': [np.nan] * 5})
        
        report = compare_datasets(ref_df, cur_df, method='ks')
        
        # Should handle empty columns gracefully
        assert 'error' in report.results['col1']


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_value_columns(self, mock_config):
        """Test with columns having single unique value."""
        ref_df = pd.DataFrame({'const': [1] * 100})
        cur_df = pd.DataFrame({'const': [1] * 100})
        
        report = compare_datasets(ref_df, cur_df, method='psi')
        assert not report.results['const']['drift_detected']
    
    def test_very_small_datasets(self, mock_config):
        """Test with very small datasets."""
        ref_df = pd.DataFrame({'col1': [1, 2]})
        cur_df = pd.DataFrame({'col1': [2, 3]})
        
        # Should work but may warn about small sample size
        report = compare_datasets(ref_df, cur_df, method='ks')
        assert isinstance(report, DriftReport)
    
    def test_extreme_values(self, mock_config):
        """Test with extreme values."""
        ref_df = pd.DataFrame({'col1': [1e-10, 1e10, -1e10]})
        cur_df = pd.DataFrame({'col1': [1e-9, 1e9, -1e9]})
        
        report = compare_datasets(ref_df, cur_df, method='ks')
        assert isinstance(report, DriftReport)