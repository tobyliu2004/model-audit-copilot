"""Utility functions for secure file handling and validation."""

import os
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import logging

from copilot.config import get_config

logger = logging.getLogger(__name__)


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    Validate and sanitize a file path to prevent path traversal attacks.
    
    Args:
        file_path: The file path to validate
        must_exist: Whether the file must exist
        
    Returns:
        Path: Validated absolute path
        
    Raises:
        ValueError: If path is invalid or contains suspicious patterns
        FileNotFoundError: If file doesn't exist and must_exist=True
    """
    # Convert to Path object
    path = Path(file_path)
    
    # Get absolute path and resolve any .. or . components
    try:
        abs_path = path.resolve()
    except Exception as e:
        raise ValueError(f"Invalid path: {file_path}")
    
    # Check for suspicious patterns
    path_str = str(abs_path)
    suspicious_patterns = ['..', '~', '$', '|', ';', '&', '>', '<', '`']
    for pattern in suspicious_patterns:
        if pattern in str(file_path):
            raise ValueError(f"Path contains suspicious pattern '{pattern}': {file_path}")
    
    # Ensure path doesn't escape current working directory (optional)
    # You can uncomment this for stricter security
    # cwd = Path.cwd()
    # if not str(abs_path).startswith(str(cwd)):
    #     raise ValueError(f"Path escapes current directory: {abs_path}")
    
    # Check if file exists
    if must_exist and not abs_path.exists():
        raise FileNotFoundError(f"File not found: {abs_path}")
    
    # Check if it's a file (not directory) when it exists
    if abs_path.exists() and abs_path.is_dir():
        raise ValueError(f"Path is a directory, not a file: {abs_path}")
    
    logger.debug(f"Validated path: {abs_path}")
    return abs_path


def load_dataframe_safely(
    file_path: Union[str, Path],
    max_size_mb: Optional[int] = None
) -> pd.DataFrame:
    """
    Safely load a DataFrame with size and format validation.
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum file size in MB (uses config default if None)
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        ValueError: If file is too large or unsupported format
    """
    config = get_config()
    path = validate_file_path(file_path)
    
    # Check file size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    max_allowed = max_size_mb or config.max_file_size_mb
    
    if file_size_mb > max_allowed:
        raise ValueError(f"File too large ({file_size_mb:.1f}MB). Maximum: {max_allowed}MB")
    
    # Check file extension
    if path.suffix not in config.allowed_file_extensions:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. "
            f"Allowed: {config.allowed_file_extensions}"
        )
    
    # Load based on file type
    logger.info(f"Loading {path.suffix} file ({file_size_mb:.1f}MB)")
    
    try:
        if path.suffix == '.csv':
            df = pd.read_csv(path)
        elif path.suffix == '.parquet':
            df = pd.read_parquet(path)
        elif path.suffix == '.json':
            df = pd.read_json(path)
        else:
            # This shouldn't happen due to validation above
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        logger.info(f"Successfully loaded data: shape={df.shape}")
        return df
        
    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty: {path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")


def sanitize_column_name(column: str) -> str:
    """
    Sanitize column name to prevent injection attacks.
    
    Args:
        column: Column name to sanitize
        
    Returns:
        str: Sanitized column name
    """
    # Remove any SQL-like patterns
    dangerous_patterns = [';', '--', '/*', '*/', 'DROP', 'DELETE', 'INSERT', 'UPDATE']
    sanitized = column
    
    for pattern in dangerous_patterns:
        sanitized = sanitized.replace(pattern, '')
    
    # Only allow alphanumeric, underscore, and dash
    import re
    sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', sanitized)
    
    return sanitized


def create_safe_directory(dir_path: Union[str, Path]) -> Path:
    """
    Safely create a directory with validation.
    
    Args:
        dir_path: Directory path to create
        
    Returns:
        Path: Created directory path
    """
    path = Path(dir_path)
    
    # Validate path doesn't contain suspicious patterns
    validate_file_path(path, must_exist=False)
    
    # Create directory
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created directory: {path}")
    
    return path