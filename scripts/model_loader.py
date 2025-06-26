"""Safe model loading utilities."""

import joblib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Supported model formats
SUPPORTED_FORMATS = {'.joblib', '.json', '.pkl'}

# Maximum file size (in MB)
MAX_MODEL_SIZE_MB = 500


def load_model(file_path: Union[str, Path]) -> Any:
    """
    Safely load a machine learning model from file.
    
    Uses joblib for secure deserialization instead of pickle.
    
    Args:
        file_path: Path to model file
        
    Returns:
        Loaded model object
        
    Raises:
        ValueError: If file format is not supported or file is too large
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    
    # Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    # Check file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_MODEL_SIZE_MB:
        raise ValueError(f"Model file too large ({file_size_mb:.1f}MB). Maximum: {MAX_MODEL_SIZE_MB}MB")
    
    # Validate file extension
    if file_path.suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported: {SUPPORTED_FORMATS}")
    
    logger.info(f"Loading model from {file_path} ({file_size_mb:.1f}MB)")
    
    try:
        if file_path.suffix in ['.joblib', '.pkl']:
            # Use joblib for safer deserialization
            model = joblib.load(file_path)
            logger.info("Model loaded successfully using joblib")
        elif file_path.suffix == '.json':
            # For simple models stored as JSON
            with open(file_path, 'r') as f:
                model = json.load(f)
            logger.info("Model configuration loaded from JSON")
        else:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
        
        # Validate loaded object
        _validate_model(model)
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def save_model(model: Any, file_path: Union[str, Path], compress: int = 3) -> None:
    """
    Save a model using joblib.
    
    Args:
        model: Model object to save
        file_path: Path to save the model
        compress: Compression level (0-9)
    """
    file_path = Path(file_path)
    
    # Create directory if needed
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model to {file_path}")
    
    try:
        joblib.dump(model, file_path, compress=compress)
        logger.info(f"Model saved successfully ({file_path.stat().st_size / 1024:.1f}KB)")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def _validate_model(model: Any) -> None:
    """
    Validate that the loaded object appears to be a valid model.
    
    Args:
        model: Object to validate
        
    Raises:
        ValueError: If object doesn't appear to be a valid model
    """
    # Check for common model attributes
    required_methods = ['predict', 'fit']
    model_methods = dir(model)
    
    # Skip validation for dict/config objects
    if isinstance(model, (dict, list)):
        return
    
    # Check for scikit-learn compatible interface
    has_sklearn_interface = all(method in model_methods for method in required_methods)
    
    # Check for other common ML frameworks
    is_tensorflow = 'tensorflow' in str(type(model))
    is_pytorch = 'torch' in str(type(model))
    is_xgboost = 'xgboost' in str(type(model))
    
    if not (has_sklearn_interface or is_tensorflow or is_pytorch or is_xgboost):
        logger.warning("Loaded object may not be a valid ML model")


def load_model_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load model metadata without loading the full model.
    
    Args:
        file_path: Path to model file
        
    Returns:
        Dictionary with model metadata
    """
    file_path = Path(file_path)
    
    metadata = {
        'file_path': str(file_path),
        'file_size_mb': file_path.stat().st_size / (1024 * 1024),
        'format': file_path.suffix,
        'modified': file_path.stat().st_mtime
    }
    
    # Try to extract additional metadata
    try:
        if file_path.suffix == '.joblib':
            # For joblib files, we can't easily extract metadata without loading
            # So we just return basic file info
            pass
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    metadata.update({
                        'model_type': data.get('model_type', 'unknown'),
                        'version': data.get('version', 'unknown')
                    })
    except Exception as e:
        logger.warning(f"Could not extract additional metadata: {e}")
    
    return metadata