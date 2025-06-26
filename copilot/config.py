"""Configuration management for Model Audit Copilot."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    ks_threshold: float = 0.05
    psi_threshold: float = 0.2
    psi_buckets: int = 10
    min_samples_per_bucket: int = 5
    methods: list = field(default_factory=lambda: ["ks", "psi"])


@dataclass
class FairnessConfig:
    """Configuration for fairness auditing."""
    min_group_size: int = 30
    bias_threshold: float = 0.1
    metrics: list = field(default_factory=lambda: ["mae", "rmse", "bias"])
    max_groups: int = 50


@dataclass
class OutlierConfig:
    """Configuration for outlier detection."""
    contamination: float = 0.01
    n_estimators: int = 100
    max_samples: str = "auto"
    random_state: int = 42


@dataclass
class LeakageConfig:
    """Configuration for data leakage detection."""
    correlation_threshold: float = 0.95
    id_pattern_threshold: float = 0.95
    duplicate_threshold: int = 0
    check_train_test_overlap: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = "audit.log"
    console: bool = True


@dataclass
class AuditConfig:
    """Main configuration class for Model Audit Copilot."""
    drift: DriftConfig = field(default_factory=DriftConfig)
    fairness: FairnessConfig = field(default_factory=FairnessConfig)
    outlier: OutlierConfig = field(default_factory=OutlierConfig)
    leakage: LeakageConfig = field(default_factory=LeakageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # General settings
    continue_on_error: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    cache_enabled: bool = True
    cache_dir: str = ".audit_cache"
    
    # File handling
    max_file_size_mb: int = 1000
    allowed_file_extensions: list = field(default_factory=lambda: [".csv", ".parquet", ".json"])


class ConfigManager:
    """Manages configuration loading and access."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = AuditConfig()
        self._config_path = config_path
        self._load_config()
        self._setup_logging()
    
    def _load_config(self):
        """Load configuration from various sources in priority order."""
        # 1. Load from config file if provided
        if self._config_path:
            self._load_from_file(self._config_path)
        else:
            # Look for default config files
            for config_file in ["audit_config.yaml", "audit_config.json", ".audit_config"]:
                if Path(config_file).exists():
                    logger.info(f"Found config file: {config_file}")
                    self._load_from_file(config_file)
                    break
        
        # 2. Override with environment variables
        self._load_from_env()
        
        # 3. Validate configuration
        self._validate_config()
    
    def _load_from_file(self, path: str):
        """Load configuration from JSON or YAML file."""
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                logger.warning(f"Config file not found: {path}")
                return
            
            with open(path_obj, 'r') as f:
                if path_obj.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            self._update_config(data)
            logger.info(f"Loaded configuration from {path}")
            
        except Exception as e:
            logger.error(f"Error loading config file {path}: {e}")
            raise
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_prefix = "AUDIT_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                self._set_config_from_env(key[len(env_prefix):].lower(), value)
    
    def _set_config_from_env(self, key: str, value: str):
        """Set a configuration value from an environment variable."""
        parts = key.split('_')
        
        try:
            if len(parts) >= 2:
                section = parts[0]
                param = '_'.join(parts[1:])
                
                # Convert value to appropriate type
                if hasattr(self.config, section):
                    section_config = getattr(self.config, section)
                    if hasattr(section_config, param):
                        current_value = getattr(section_config, param)
                        
                        # Type conversion
                        if isinstance(current_value, bool):
                            converted_value = value.lower() in ['true', '1', 'yes']
                        elif isinstance(current_value, int):
                            converted_value = int(value)
                        elif isinstance(current_value, float):
                            converted_value = float(value)
                        elif isinstance(current_value, list):
                            converted_value = value.split(',')
                        else:
                            converted_value = value
                        
                        setattr(section_config, param, converted_value)
                        logger.debug(f"Set {section}.{param} = {converted_value} from env")
            else:
                # Top-level config
                if hasattr(self.config, parts[0]):
                    current_value = getattr(self.config, parts[0])
                    if isinstance(current_value, bool):
                        converted_value = value.lower() in ['true', '1', 'yes']
                    elif isinstance(current_value, int):
                        converted_value = int(value)
                    elif isinstance(current_value, float):
                        converted_value = float(value)
                    else:
                        converted_value = value
                    setattr(self.config, parts[0], converted_value)
        
        except Exception as e:
            logger.warning(f"Failed to set config from env var {key}: {e}")
    
    def _update_config(self, data: Dict[str, Any]):
        """Update configuration from dictionary."""
        for section, values in data.items():
            if hasattr(self.config, section):
                if isinstance(values, dict):
                    section_config = getattr(self.config, section)
                    for key, value in values.items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)
                else:
                    setattr(self.config, section, values)
    
    def _validate_config(self):
        """Validate configuration values."""
        # Validate drift config
        if not 0 < self.config.drift.ks_threshold < 1:
            raise ValueError(f"Invalid KS threshold: {self.config.drift.ks_threshold}")
        if not 0 < self.config.drift.psi_threshold < 10:
            raise ValueError(f"Invalid PSI threshold: {self.config.drift.psi_threshold}")
        if self.config.drift.psi_buckets < 2:
            raise ValueError(f"PSI buckets must be >= 2: {self.config.drift.psi_buckets}")
        
        # Validate fairness config
        if self.config.fairness.min_group_size < 1:
            raise ValueError(f"Min group size must be >= 1: {self.config.fairness.min_group_size}")
        
        # Validate outlier config
        if not 0 < self.config.outlier.contamination < 0.5:
            raise ValueError(f"Invalid contamination rate: {self.config.outlier.contamination}")
        
        # Validate general settings
        if self.config.max_workers < 1:
            raise ValueError(f"Max workers must be >= 1: {self.config.max_workers}")
        if self.config.max_file_size_mb < 1:
            raise ValueError(f"Max file size must be >= 1: {self.config.max_file_size_mb}")
    
    def _setup_logging(self):
        """Configure logging based on config."""
        handlers = []
        
        if self.config.logging.console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(self.config.logging.format))
            handlers.append(console_handler)
        
        if self.config.logging.file:
            file_handler = logging.FileHandler(self.config.logging.file)
            file_handler.setFormatter(logging.Formatter(self.config.logging.format))
            handlers.append(file_handler)
        
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level.upper()),
            handlers=handlers
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        try:
            parts = key.split('.')
            value = self.config
            for part in parts:
                value = getattr(value, part)
            return value
        except AttributeError:
            return default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self.config)
    
    def save(self, path: str):
        """Save current configuration to file."""
        path_obj = Path(path)
        data = self.to_dict()
        
        with open(path_obj, 'w') as f:
            if path_obj.suffix in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)
        
        logger.info(f"Saved configuration to {path}")


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> AuditConfig:
    """Get the global configuration instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.config


def load_config(path: Optional[str] = None) -> AuditConfig:
    """Load configuration from file."""
    global _config_manager
    _config_manager = ConfigManager(path)
    return _config_manager.config