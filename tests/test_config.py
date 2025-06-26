"""Tests for configuration management."""

import pytest
import os
import tempfile
import json
import yaml
from pathlib import Path

from copilot.config import (
    AuditConfig, DriftConfig, FairnessConfig, OutlierConfig,
    LeakageConfig, LoggingConfig, ConfigManager, get_config, load_config
)


class TestConfigDataclasses:
    """Test configuration dataclass structures."""
    
    def test_drift_config_defaults(self):
        """Test DriftConfig default values."""
        config = DriftConfig()
        
        assert config.ks_threshold == 0.05
        assert config.psi_threshold == 0.2
        assert config.psi_buckets == 10
        assert config.min_samples_per_bucket == 5
        assert config.methods == ["ks", "psi"]
    
    def test_fairness_config_defaults(self):
        """Test FairnessConfig default values."""
        config = FairnessConfig()
        
        assert config.min_group_size == 30
        assert config.bias_threshold == 0.1
        assert config.metrics == ["mae", "rmse", "bias"]
        assert config.max_groups == 50
    
    def test_outlier_config_defaults(self):
        """Test OutlierConfig default values."""
        config = OutlierConfig()
        
        assert config.contamination == 0.01
        assert config.n_estimators == 100
        assert config.max_samples == "auto"
        assert config.random_state == 42
    
    def test_audit_config_defaults(self):
        """Test AuditConfig default values."""
        config = AuditConfig()
        
        assert isinstance(config.drift, DriftConfig)
        assert isinstance(config.fairness, FairnessConfig)
        assert isinstance(config.outlier, OutlierConfig)
        assert config.continue_on_error is False
        assert config.parallel_processing is True
        assert config.max_workers == 4


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def test_initialization(self):
        """Test ConfigManager initialization."""
        manager = ConfigManager()
        
        assert isinstance(manager.config, AuditConfig)
        assert manager._config_path is None
    
    def test_load_from_yaml_file(self, tmp_path):
        """Test loading configuration from YAML file."""
        # Create test YAML config
        config_data = {
            'drift': {
                'ks_threshold': 0.01,
                'psi_threshold': 0.15
            },
            'fairness': {
                'min_group_size': 50
            }
        }
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load config
        manager = ConfigManager(str(config_file))
        
        assert manager.config.drift.ks_threshold == 0.01
        assert manager.config.drift.psi_threshold == 0.15
        assert manager.config.fairness.min_group_size == 50
    
    def test_load_from_json_file(self, tmp_path):
        """Test loading configuration from JSON file."""
        config_data = {
            'drift': {
                'psi_buckets': 20
            },
            'continue_on_error': True
        }
        
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        manager = ConfigManager(str(config_file))
        
        assert manager.config.drift.psi_buckets == 20
        assert manager.config.continue_on_error is True
    
    def test_load_from_environment(self, monkeypatch):
        """Test loading configuration from environment variables."""
        # Set environment variables
        monkeypatch.setenv('AUDIT_DRIFT_KS_THRESHOLD', '0.1')
        monkeypatch.setenv('AUDIT_FAIRNESS_BIAS_THRESHOLD', '0.2')
        monkeypatch.setenv('AUDIT_CONTINUE_ON_ERROR', 'true')
        monkeypatch.setenv('AUDIT_MAX_WORKERS', '8')
        
        manager = ConfigManager()
        
        assert manager.config.drift.ks_threshold == 0.1
        assert manager.config.fairness.bias_threshold == 0.2
        assert manager.config.continue_on_error is True
        assert manager.config.max_workers == 8
    
    def test_environment_override(self, tmp_path, monkeypatch):
        """Test that environment variables override file config."""
        # Create config file
        config_data = {'drift': {'ks_threshold': 0.05}}
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Set environment variable
        monkeypatch.setenv('AUDIT_DRIFT_KS_THRESHOLD', '0.15')
        
        manager = ConfigManager(str(config_file))
        
        # Environment should override file
        assert manager.config.drift.ks_threshold == 0.15
    
    def test_validation_drift_config(self):
        """Test validation of drift configuration."""
        manager = ConfigManager()
        
        # Set invalid values
        manager.config.drift.ks_threshold = 1.5  # Should be between 0 and 1
        
        with pytest.raises(ValueError):
            manager._validate_config()
        
        # Reset and test another validation
        manager.config.drift.ks_threshold = 0.05
        manager.config.drift.psi_buckets = 1  # Should be >= 2
        
        with pytest.raises(ValueError):
            manager._validate_config()
    
    def test_validation_fairness_config(self):
        """Test validation of fairness configuration."""
        manager = ConfigManager()
        
        manager.config.fairness.min_group_size = 0  # Should be >= 1
        
        with pytest.raises(ValueError):
            manager._validate_config()
    
    def test_validation_outlier_config(self):
        """Test validation of outlier configuration."""
        manager = ConfigManager()
        
        manager.config.outlier.contamination = 0.6  # Should be < 0.5
        
        with pytest.raises(ValueError):
            manager._validate_config()
    
    def test_get_method(self):
        """Test getting config values with dot notation."""
        manager = ConfigManager()
        
        assert manager.get('drift.ks_threshold') == 0.05
        assert manager.get('fairness.min_group_size') == 30
        assert manager.get('nonexistent.key', 'default') == 'default'
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        manager = ConfigManager()
        config_dict = manager.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'drift' in config_dict
        assert 'fairness' in config_dict
        assert config_dict['drift']['ks_threshold'] == 0.05
    
    def test_save_yaml(self, tmp_path):
        """Test saving configuration to YAML."""
        manager = ConfigManager()
        manager.config.drift.ks_threshold = 0.1
        
        save_path = tmp_path / "saved_config.yaml"
        manager.save(str(save_path))
        
        # Load and verify
        with open(save_path) as f:
            loaded = yaml.safe_load(f)
        
        assert loaded['drift']['ks_threshold'] == 0.1
    
    def test_save_json(self, tmp_path):
        """Test saving configuration to JSON."""
        manager = ConfigManager()
        manager.config.fairness.min_group_size = 100
        
        save_path = tmp_path / "saved_config.json"
        manager.save(str(save_path))
        
        # Load and verify
        with open(save_path) as f:
            loaded = json.load(f)
        
        assert loaded['fairness']['min_group_size'] == 100
    
    def test_missing_config_file(self):
        """Test handling of missing config file."""
        # Should not raise error, just use defaults
        manager = ConfigManager("nonexistent_file.yaml")
        
        assert manager.config.drift.ks_threshold == 0.05  # Default value
    
    def test_invalid_yaml_file(self, tmp_path):
        """Test handling of invalid YAML syntax."""
        config_file = tmp_path / "invalid.yaml"
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: syntax:")
        
        with pytest.raises(Exception):
            ConfigManager(str(config_file))
    
    def test_list_environment_variable(self, monkeypatch):
        """Test parsing list from environment variable."""
        monkeypatch.setenv('AUDIT_DRIFT_METHODS', 'ks,psi,custom')
        
        manager = ConfigManager()
        
        assert manager.config.drift.methods == ['ks', 'psi', 'custom']


class TestGlobalFunctions:
    """Test global configuration functions."""
    
    def test_get_config(self):
        """Test get_config function."""
        config1 = get_config()
        config2 = get_config()
        
        # Should return same instance
        assert config1 is config2
        assert isinstance(config1, AuditConfig)
    
    def test_load_config(self, tmp_path):
        """Test load_config function."""
        # Create config file
        config_data = {'drift': {'ks_threshold': 0.123}}
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load config
        config = load_config(str(config_file))
        
        assert isinstance(config, AuditConfig)
        assert config.drift.ks_threshold == 0.123
        
        # Subsequent calls to get_config should return same instance
        assert get_config() is config