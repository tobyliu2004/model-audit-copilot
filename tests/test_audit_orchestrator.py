"""Tests for the audit orchestrator module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from copilot.auditor.audit_orchestrator import AuditOrchestrator


class TestAuditOrchestrator:
    """Test cases for AuditOrchestrator class."""
    
    def test_initialization(self, sample_dataframe):
        """Test orchestrator initialization."""
        ref_df = sample_dataframe
        cur_df = sample_dataframe.copy()
        
        orchestrator = AuditOrchestrator(ref_df, cur_df)
        
        assert orchestrator.ref_df is ref_df
        assert orchestrator.cur_df is cur_df
        assert orchestrator.results == {}
        assert orchestrator._drift_report is None
    
    def test_initialization_without_reference(self, sample_dataframe):
        """Test initialization with only current dataset."""
        orchestrator = AuditOrchestrator(None, sample_dataframe)
        
        assert orchestrator.ref_df is None
        assert orchestrator.cur_df is sample_dataframe
    
    def test_initialization_invalid_current(self):
        """Test initialization with invalid current dataset."""
        with pytest.raises(ValueError):
            AuditOrchestrator(pd.DataFrame(), None)
        
        with pytest.raises(ValueError):
            AuditOrchestrator(pd.DataFrame(), pd.DataFrame())  # Empty DataFrame
    
    def test_run_drift_check(self, sample_dataframe, drifted_dataframe, mock_config):
        """Test drift check execution."""
        orchestrator = AuditOrchestrator(sample_dataframe, drifted_dataframe)
        
        result = orchestrator.run_drift_check(method='ks')
        
        assert result is orchestrator  # Method chaining
        assert 'drift' in orchestrator.results
        assert isinstance(orchestrator.results['drift'], pd.DataFrame)
        assert orchestrator._drift_report is not None
    
    def test_drift_check_without_reference(self, sample_dataframe):
        """Test drift check fails without reference data."""
        orchestrator = AuditOrchestrator(None, sample_dataframe)
        
        with pytest.raises(ValueError):
            orchestrator.run_drift_check()
    
    def test_run_fairness_check(self, biased_dataframe, mock_config):
        """Test fairness check execution."""
        df = biased_dataframe
        orchestrator = AuditOrchestrator(None, df)
        
        result = orchestrator.run_fairness_check(
            y_true=df['y_true'],
            y_pred=df['y_pred'],
            group_feature=df['group']
        )
        
        assert result is orchestrator  # Method chaining
        assert 'fairness' in orchestrator.results
        assert isinstance(orchestrator.results['fairness'], pd.DataFrame)
    
    def test_run_leakage_check(self, leakage_dataframe, mock_config):
        """Test leakage check execution."""
        orchestrator = AuditOrchestrator(None, leakage_dataframe)
        
        result = orchestrator.run_leakage_check(target_col='target')
        
        assert result is orchestrator  # Method chaining
        assert 'leakage' in orchestrator.results
        assert 'target_leakage' in orchestrator.results['leakage']
        assert 'duplicate_rows' in orchestrator.results['leakage']
        assert 'id_like_columns' in orchestrator.results['leakage']
    
    def test_run_schema_check(self, sample_dataframe, mock_config):
        """Test schema check execution."""
        ref_df = sample_dataframe
        cur_df = sample_dataframe.copy()
        cur_df['new_column'] = 1  # Add extra column
        cur_df = cur_df.drop(columns=['numeric_col1'])  # Remove column
        
        orchestrator = AuditOrchestrator(ref_df, cur_df)
        result = orchestrator.run_schema_check()
        
        assert result is orchestrator  # Method chaining
        assert 'schema' in orchestrator.results
        assert 'schema_comparison' in orchestrator.results['schema']
        assert 'column_diff' in orchestrator.results['schema']
        
        # Check column differences
        column_diff = orchestrator.results['schema']['column_diff']
        assert 'numeric_col1' in column_diff['missing_in_current']
        assert 'new_column' in column_diff['extra_in_current']
    
    def test_schema_check_without_reference(self, sample_dataframe):
        """Test schema check fails without reference data."""
        orchestrator = AuditOrchestrator(None, sample_dataframe)
        
        with pytest.raises(ValueError):
            orchestrator.run_schema_check()
    
    def test_run_explainability(self, sample_dataframe, sample_model, mock_config):
        """Test explainability check execution."""
        orchestrator = AuditOrchestrator(None, sample_dataframe)
        
        # Prepare feature data
        X = sample_dataframe.select_dtypes(include=[np.number]).iloc[:100]
        
        with patch('copilot.explainability.shap_explainer.compute_shap_summary') as mock_shap:
            mock_fig = Mock()
            mock_shap.return_value = mock_fig
            
            result = orchestrator.run_explainability(sample_model, X)
            
            assert result is orchestrator  # Method chaining
            assert 'shap_summary_plot' in orchestrator.results
            assert orchestrator.results['shap_summary_plot'] is mock_fig
    
    def test_run_all_checks(self, sample_dataframe, biased_dataframe, mock_config):
        """Test running all checks at once."""
        ref_df = sample_dataframe
        cur_df = biased_dataframe
        
        orchestrator = AuditOrchestrator(ref_df, cur_df)
        
        result = orchestrator.run_all_checks(
            drift_method='ks',
            target_col='y_true',
            prediction_col='y_pred',
            group_col='group'
        )
        
        assert result is orchestrator  # Method chaining
        
        # Should have run multiple checks
        assert 'drift' in orchestrator.results
        assert 'schema' in orchestrator.results
        assert 'leakage' in orchestrator.results
        assert 'fairness' in orchestrator.results
    
    def test_continue_on_error(self, sample_dataframe, mock_config):
        """Test continue_on_error configuration."""
        orchestrator = AuditOrchestrator(sample_dataframe, sample_dataframe)
        
        # Mock config to continue on error
        orchestrator.config.continue_on_error = True
        
        # Create a scenario that will fail
        with patch.object(orchestrator, 'run_drift_check', side_effect=Exception("Test error")):
            # Should not raise exception
            orchestrator.run_all_checks()
            
            # Other checks should still run
            assert len(orchestrator.results) > 0
    
    def test_getter_methods(self, sample_dataframe, mock_config):
        """Test all getter methods."""
        orchestrator = AuditOrchestrator(sample_dataframe, sample_dataframe)
        
        # Initially all should return None/empty
        assert orchestrator.get_drift_report() is None
        assert orchestrator.get_fairness_report() is None
        assert orchestrator.get_leakage_report() == {}
        assert orchestrator.get_schema_report() == {}
        assert orchestrator.get_drift_plot() is None
        assert orchestrator.get_shap_plot() is None
        
        # Run some checks
        orchestrator.run_drift_check()
        orchestrator.run_leakage_check()
        
        # Now should return results
        assert orchestrator.get_drift_report() is not None
        assert orchestrator.get_leakage_report() != {}
        assert orchestrator.get_drift_plot() is not None  # Should generate plot
    
    def test_generate_summary(self, sample_dataframe, drifted_dataframe, mock_config):
        """Test summary generation."""
        orchestrator = AuditOrchestrator(sample_dataframe, drifted_dataframe)
        
        # Run various checks
        orchestrator.run_drift_check()
        orchestrator.run_leakage_check()
        
        summary = orchestrator.generate_summary()
        
        assert 'audits_run' in summary
        assert 'issues_found' in summary
        assert 'drift' in summary['audits_run']
        assert 'leakage' in summary['audits_run']
        
        # Check issue counts
        assert 'drift' in summary['issues_found']
        assert summary['issues_found']['drift']['features_with_drift'] > 0
    
    def test_method_chaining(self, sample_dataframe, mock_config):
        """Test that all methods support chaining."""
        orchestrator = AuditOrchestrator(sample_dataframe, sample_dataframe)
        
        # Should be able to chain all methods
        result = (orchestrator
                  .run_drift_check()
                  .run_schema_check()
                  .run_leakage_check())
        
        assert result is orchestrator
        assert len(orchestrator.results) >= 3
    
    @patch('copilot.auditor.audit_orchestrator.logger')
    def test_logging(self, mock_logger, sample_dataframe, mock_config):
        """Test that appropriate logging occurs."""
        orchestrator = AuditOrchestrator(sample_dataframe, sample_dataframe)
        
        # Run various checks
        orchestrator.run_drift_check()
        orchestrator.run_leakage_check()
        
        # Should have logged initialization and operations
        assert mock_logger.info.call_count > 0
        
        # Check for specific log messages
        log_messages = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any('Initialized AuditOrchestrator' in msg for msg in log_messages)
        assert any('drift check' in msg for msg in log_messages)
        assert any('leakage check' in msg for msg in log_messages)
    
    def test_error_handling(self, sample_dataframe, mock_config):
        """Test error handling and propagation."""
        orchestrator = AuditOrchestrator(sample_dataframe, sample_dataframe)
        
        # Test with invalid method
        with pytest.raises(ValueError):
            orchestrator.run_drift_check(method='invalid_method')
        
        # Test with invalid column names
        with pytest.raises(KeyError):
            orchestrator.run_fairness_check(
                y_true=sample_dataframe['nonexistent'],
                y_pred=sample_dataframe['also_nonexistent'],
                group_feature=sample_dataframe['group']
            )