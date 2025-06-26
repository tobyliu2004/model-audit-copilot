"""Tests for data leakage detection module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from copilot.leakage.leakage_detector import (
    detect_target_leakage,
    detect_duplicate_and_id_leakage,
    detect_train_test_overlap
)


class TestDetectTargetLeakage:
    """Test cases for target leakage detection."""
    
    def test_high_correlation_detection(self):
        """Test detection of features highly correlated with target."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create features with varying correlation to target
        target = np.random.normal(0, 1, n_samples)
        
        df = pd.DataFrame({
            'target': target,
            'high_corr': target * 0.95 + np.random.normal(0, 0.1, n_samples),  # High correlation
            'medium_corr': target * 0.5 + np.random.normal(0, 1, n_samples),   # Medium correlation
            'low_corr': np.random.normal(0, 1, n_samples),                     # Low correlation
            'perfect_corr': target * 1.0,                                      # Perfect correlation
            'categorical': np.random.choice(['A', 'B', 'C'], n_samples)        # Non-numeric
        })
        
        result = detect_target_leakage(df, 'target', threshold=0.9)
        
        assert 'high_corr' in result['high_correlation_features']
        assert 'perfect_corr' in result['high_correlation_features']
        assert 'medium_corr' not in result['high_correlation_features']
        assert 'low_corr' not in result['high_correlation_features']
        
        # Check correlation values
        assert result['feature_correlations']['perfect_corr'] == pytest.approx(1.0, abs=0.01)
        assert result['feature_correlations']['high_corr'] > 0.9
    
    def test_no_leakage(self):
        """Test when no leakage is present."""
        np.random.seed(42)
        df = pd.DataFrame({
            'target': np.random.normal(0, 1, 100),
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        
        result = detect_target_leakage(df, 'target', threshold=0.95)
        
        assert len(result['high_correlation_features']) == 0
        assert all(abs(corr) < 0.95 for corr in result['feature_correlations'].values())
    
    def test_target_not_in_dataframe(self):
        """Test error when target column doesn't exist."""
        df = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(ValueError):
            detect_target_leakage(df, 'nonexistent_target')
    
    def test_custom_threshold(self):
        """Test with custom correlation threshold."""
        np.random.seed(42)
        target = np.random.normal(0, 1, 100)
        
        df = pd.DataFrame({
            'target': target,
            'feature1': target * 0.7 + np.random.normal(0, 0.3, 100),  # ~0.7 correlation
            'feature2': target * 0.8 + np.random.normal(0, 0.2, 100),  # ~0.8 correlation
        })
        
        # With threshold 0.75
        result = detect_target_leakage(df, 'target', threshold=0.75)
        assert 'feature2' in result['high_correlation_features']
        assert 'feature1' not in result['high_correlation_features']
    
    def test_only_categorical_features(self):
        """Test with only categorical features."""
        df = pd.DataFrame({
            'target': [1, 2, 3, 4, 5],
            'cat1': ['A', 'B', 'C', 'D', 'E'],
            'cat2': ['X', 'Y', 'Z', 'X', 'Y']
        })
        
        result = detect_target_leakage(df, 'target')
        
        # Should handle gracefully with no numeric features
        assert len(result['high_correlation_features']) == 0
        assert len(result['feature_correlations']) == 0


class TestDetectDuplicateAndIdLeakage:
    """Test cases for duplicate and ID-like column detection."""
    
    def test_duplicate_detection(self):
        """Test detection of duplicate rows."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 1, 2],  # Rows 0&3, 1&4 are duplicates
            'col2': ['a', 'b', 'c', 'a', 'b']
        })
        
        result = detect_duplicate_and_id_leakage(df)
        
        assert result['duplicate_rows'] == 2
        assert result['duplicate_percentage'] == 40.0
    
    def test_no_duplicates(self):
        """Test when no duplicates exist."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        
        result = detect_duplicate_and_id_leakage(df)
        
        assert result['duplicate_rows'] == 0
        assert result['duplicate_percentage'] == 0.0
    
    def test_id_column_detection_numeric(self):
        """Test detection of numeric ID-like columns."""
        n_samples = 100
        df = pd.DataFrame({
            'sequential_id': range(n_samples),                              # Sequential ID
            'random_id': np.random.randint(100000, 999999, n_samples),     # Random but unique
            'normal_feature': np.random.normal(0, 1, n_samples),           # Normal feature
            'categorical': np.random.choice(['A', 'B', 'C'], n_samples)    # Repeated values
        })
        
        result = detect_duplicate_and_id_leakage(df)
        
        assert 'sequential_id' in result['id_like_columns']
        assert 'random_id' in result['id_like_columns']
        assert 'normal_feature' not in result['id_like_columns']
    
    def test_id_column_detection_string(self):
        """Test detection of string ID-like columns."""
        n_samples = 50
        df = pd.DataFrame({
            'customer_id': [f'CUST_{i:06d}' for i in range(n_samples)],
            'transaction_id': [f'TXN_{np.random.randint(100000, 999999)}' for _ in range(n_samples)],
            'category': np.random.choice(['Electronics', 'Books', 'Clothing'], n_samples),
            'unique_names': [f'Name_{i}' for i in range(n_samples)]
        })
        
        result = detect_duplicate_and_id_leakage(df)
        
        assert 'customer_id' in result['id_like_columns']
        assert 'unique_names' in result['id_like_columns']
        assert 'category' not in result['id_like_columns']
    
    def test_threshold_parameter(self):
        """Test ID detection with custom threshold."""
        df = pd.DataFrame({
            'mostly_unique': [1, 2, 3, 4, 5, 5],  # 83% unique
            'somewhat_unique': [1, 1, 2, 2, 3, 3],  # 50% unique
            'not_unique': [1, 1, 1, 2, 2, 2]  # 33% unique
        })
        
        # Default threshold (0.95)
        result1 = detect_duplicate_and_id_leakage(df)
        assert 'mostly_unique' not in result1['id_like_columns']
        
        # Lower threshold (0.8)
        result2 = detect_duplicate_and_id_leakage(df, id_threshold=0.8)
        assert 'mostly_unique' in result2['id_like_columns']
        assert 'somewhat_unique' not in result2['id_like_columns']
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = detect_duplicate_and_id_leakage(df)
        
        assert result['duplicate_rows'] == 0
        assert result['duplicate_percentage'] == 0.0
        assert len(result['id_like_columns']) == 0
    
    def test_all_duplicates(self):
        """Test when all rows are duplicates."""
        df = pd.DataFrame({
            'col1': [1] * 10,
            'col2': ['a'] * 10
        })
        
        result = detect_duplicate_and_id_leakage(df)
        
        assert result['duplicate_rows'] == 9  # All but first are duplicates
        assert result['duplicate_percentage'] == 90.0


class TestDetectTrainTestOverlap:
    """Test cases for train/test overlap detection."""
    
    def test_overlap_detection(self):
        """Test detection of overlapping rows."""
        train_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Test set with some overlapping rows
        test_df = pd.DataFrame({
            'feature1': [3, 4, 6, 7],  # 3 and 4 overlap
            'feature2': ['c', 'd', 'f', 'g']
        })
        
        result = detect_train_test_overlap(train_df, test_df)
        
        assert result['overlapping_rows'] == 2
        assert result['overlap_percentage'] == 50.0  # 2 out of 4 test rows
        assert len(result['overlapping_indices']) == 2
        assert 0 in result['overlapping_indices']  # First row of test (3,c)
        assert 1 in result['overlapping_indices']  # Second row of test (4,d)
    
    def test_no_overlap(self):
        """Test when no overlap exists."""
        train_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        test_df = pd.DataFrame({
            'col1': [4, 5, 6],
            'col2': ['d', 'e', 'f']
        })
        
        result = detect_train_test_overlap(train_df, test_df)
        
        assert result['overlapping_rows'] == 0
        assert result['overlap_percentage'] == 0.0
        assert len(result['overlapping_indices']) == 0
    
    def test_complete_overlap(self):
        """Test when test set is subset of train set."""
        train_df = pd.DataFrame({
            'col1': range(10),
            'col2': [f'val_{i}' for i in range(10)]
        })
        
        test_df = train_df.iloc[5:8].copy()  # Subset of train
        
        result = detect_train_test_overlap(train_df, test_df)
        
        assert result['overlapping_rows'] == 3
        assert result['overlap_percentage'] == 100.0
        assert len(result['overlapping_indices']) == 3
    
    def test_different_columns(self):
        """Test when DataFrames have different columns."""
        train_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [10, 20, 30]
        })
        
        test_df = pd.DataFrame({
            'col1': [1, 2, 4],
            'col2': ['a', 'b', 'd']
            # col3 missing
        })
        
        # Should compare only common columns
        result = detect_train_test_overlap(train_df, test_df)
        
        assert result['overlapping_rows'] == 2  # Rows (1,a) and (2,b) match
    
    def test_numeric_precision(self):
        """Test overlap detection with floating point numbers."""
        train_df = pd.DataFrame({
            'float_col': [1.0, 2.0, 3.0],
            'int_col': [10, 20, 30]
        })
        
        test_df = pd.DataFrame({
            'float_col': [1.0000001, 2.0, 4.0],  # Very close to 1.0
            'int_col': [10, 20, 40]
        })
        
        result = detect_train_test_overlap(train_df, test_df)
        
        # Should handle floating point comparison appropriately
        # Exact behavior depends on pandas/numpy comparison
        assert result['overlapping_rows'] >= 1  # At least (2.0, 20) should match
    
    def test_large_datasets(self):
        """Test performance with larger datasets."""
        np.random.seed(42)
        n_train = 10000
        n_test = 2000
        n_overlap = 500
        
        # Create train data
        train_data = {
            'id': range(n_train),
            'value': np.random.randn(n_train)
        }
        train_df = pd.DataFrame(train_data)
        
        # Create test data with some overlap
        test_data = {
            'id': list(range(n_overlap)) + list(range(n_train, n_train + n_test - n_overlap)),
            'value': np.concatenate([
                train_df.iloc[:n_overlap]['value'].values,
                np.random.randn(n_test - n_overlap)
            ])
        }
        test_df = pd.DataFrame(test_data)
        
        result = detect_train_test_overlap(train_df, test_df)
        
        assert result['overlapping_rows'] == n_overlap
        assert result['overlap_percentage'] == pytest.approx(25.0, abs=0.1)  # 500/2000
    
    @patch('copilot.leakage.leakage_detector.logger')
    def test_logging(self, mock_logger):
        """Test that appropriate logging occurs."""
        train_df = pd.DataFrame({'col': [1, 2, 3]})
        test_df = pd.DataFrame({'col': [2, 3, 4]})
        
        detect_train_test_overlap(train_df, test_df)
        
        # Should log info about overlap detection
        mock_logger.info.assert_called()
        mock_logger.warning.assert_called()  # Should warn about overlap