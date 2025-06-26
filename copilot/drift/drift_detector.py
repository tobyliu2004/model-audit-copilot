"""Drift detection module for comparing dataset distributions."""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, List, Tuple
import logging

from copilot.config import get_config

logger = logging.getLogger(__name__)


class DriftReport:
    """
    Container for drift detection results across multiple features.
    
    Attributes:
        results: Dictionary mapping feature names to drift detection results
    """
    
    def __init__(self, results: Dict[str, Dict[str, Any]]):
        """Initialize DriftReport with detection results."""
        self.results = results
        self._validate_results()
        
    def _validate_results(self):
        """Validate the structure of results dictionary."""
        if not isinstance(self.results, dict):
            raise TypeError("results must be a dictionary")
        
        for feature, result in self.results.items():
            if not isinstance(result, dict):
                raise ValueError(f"Result for feature '{feature}' must be a dictionary")
    
    def summary(self) -> pd.DataFrame:
        """
        Generate a summary DataFrame of drift detection results.
        
        Returns:
            pd.DataFrame: Summary with features as rows and metrics as columns
        """
        logger.debug("Generating drift report summary")
        return pd.DataFrame(self.results).T
    
    def get_drifted_features(self) -> List[str]:
        """Get list of features where drift was detected."""
        drifted = [feat for feat, result in self.results.items() 
                   if result.get('drift_detected', False)]
        logger.info(f"Found {len(drifted)} features with drift")
        return drifted
    
    def plot(self, return_fig: bool = False, figsize: Tuple[int, int] = (10, 6)) -> Optional[plt.Figure]:
        """
        Create visualization of drift detection results.
        
        Args:
            return_fig: If True, return figure instead of displaying
            figsize: Figure size as (width, height)
            
        Returns:
            matplotlib.figure.Figure if return_fig=True, else None
        """
        features = list(self.results.keys())
        
        # Extract appropriate metric
        if 'ks_stat' in self.results[features[0]]:
            values = [self.results[f].get('ks_stat', 0) for f in features]
            metric_name = "KS Statistic"
        elif 'psi' in self.results[features[0]]:
            values = [self.results[f].get('psi', 0) for f in features]
            metric_name = "PSI"
        else:
            logger.warning("No recognized drift metric found in results")
            return None
        
        colors = ['crimson' if self.results[f].get('drift_detected', False) 
                  else 'steelblue' for f in features]
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(features, values, color=colors)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center')
        
        ax.set_xlabel(metric_name)
        ax.set_title(f"Feature-wise Drift Detection ({metric_name})")
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if return_fig:
            return fig
        else:
            plt.show()
            return None


def calculate_psi(
    expected: Union[pd.Series, np.ndarray, list], 
    actual: Union[pd.Series, np.ndarray, list], 
    buckets: Optional[int] = None
) -> float:
    """
    Calculate Population Stability Index (PSI) between two distributions.
    
    Args:
        expected: Reference distribution
        actual: Current distribution  
        buckets: Number of buckets for binning (uses config default if None)
        
    Returns:
        PSI value
        
    Raises:
        ValueError: If inputs are invalid
    """
    config = get_config()
    if buckets is None:
        buckets = config.drift.psi_buckets
    
    # Validate inputs
    if buckets < 2:
        raise ValueError(f"buckets must be >= 2, got {buckets}")
    
    # Convert to pandas Series
    expected = pd.Series(expected)
    actual = pd.Series(actual)
    
    # Log null counts
    expected_nulls = expected.isna().sum()
    actual_nulls = actual.isna().sum()
    
    if expected_nulls > 0:
        logger.debug(f"Dropping {expected_nulls} NaN values from expected distribution")
    if actual_nulls > 0:
        logger.debug(f"Dropping {actual_nulls} NaN values from actual distribution")
    
    expected = expected.dropna()
    actual = actual.dropna()
    
    # Check minimum samples
    min_samples = config.drift.min_samples_per_bucket * buckets
    if len(expected) < min_samples:
        logger.warning(
            f"Expected distribution has only {len(expected)} samples. "
            f"Consider reducing buckets to {len(expected) // config.drift.min_samples_per_bucket}"
        )
    
    # Create bins
    if pd.api.types.is_numeric_dtype(expected):
        # For numeric data, use quantile-based bins
        bins = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)
    else:
        # For categorical data
        categories = sorted(set(expected) | set(actual))
        expected_counts = expected.value_counts().reindex(categories, fill_value=0).values
        actual_counts = actual.value_counts().reindex(categories, fill_value=0).values
    
    # Calculate PSI
    expected_pct = (expected_counts + 1) / (expected_counts.sum() + len(expected_counts))
    actual_pct = (actual_counts + 1) / (actual_counts.sum() + len(actual_counts))
    
    psi_values = (expected_pct - actual_pct) * np.log(expected_pct / actual_pct)
    psi = np.sum(psi_values)
    
    logger.debug(f"PSI calculation: {psi:.4f}")
    return round(psi, 4)


def compare_datasets(
    reference_df: pd.DataFrame, 
    current_df: pd.DataFrame, 
    method: str = 'ks',
    columns: Optional[List[str]] = None
) -> DriftReport:
    """
    Compare two datasets for distribution drift.
    
    Args:
        reference_df: Reference dataset (e.g., training data)
        current_df: Current dataset (e.g., production data)
        method: Drift detection method ('ks' or 'psi')
        columns: Specific columns to compare (None = all common columns)
        
    Returns:
        DriftReport object containing results
        
    Raises:
        ValueError: If method is invalid or datasets are incompatible
    """
    config = get_config()
    
    logger.info(f"Starting dataset comparison using method: {method}")
    logger.debug(f"Reference shape: {reference_df.shape}, Current shape: {current_df.shape}")
    
    # Validate method
    if method not in config.drift.methods:
        raise ValueError(f"Invalid method '{method}'. Allowed: {config.drift.methods}")
    
    # Determine columns to compare
    if columns:
        common_columns = [col for col in columns if col in reference_df.columns and col in current_df.columns]
        missing_ref = [col for col in columns if col not in reference_df.columns]
        missing_cur = [col for col in columns if col not in current_df.columns]
        
        if missing_ref:
            logger.warning(f"Columns not in reference dataset: {missing_ref}")
        if missing_cur:
            logger.warning(f"Columns not in current dataset: {missing_cur}")
    else:
        common_columns = [col for col in reference_df.columns if col in current_df.columns]
    
    logger.info(f"Comparing {len(common_columns)} common columns")
    
    drift_results = {}
    errors_count = 0
    
    for col in common_columns:
        try:
            logger.debug(f"Processing column: {col}")
            ref_series = reference_df[col].dropna()
            cur_series = current_df[col].dropna()
            
            # Check if column has enough data
            if len(ref_series) < 10 or len(cur_series) < 10:
                logger.warning(f"Column '{col}' has too few samples for reliable drift detection")
            
            is_numeric = pd.api.types.is_numeric_dtype(ref_series)
            
            if method == 'ks' and is_numeric:
                # Kolmogorov-Smirnov test for numeric data
                ks_stat, p_value = ks_2samp(ref_series, cur_series)
                drift_detected = p_value < config.drift.ks_threshold
                
                drift_results[col] = {
                    'ks_stat': round(ks_stat, 4),
                    'p_value': round(p_value, 4),
                    'drift_detected': drift_detected,
                    'method': 'ks',
                    'samples': {'reference': len(ref_series), 'current': len(cur_series)}
                }
                
                if drift_detected:
                    logger.warning(
                        f"KS test detected drift in '{col}': "
                        f"ks_stat={ks_stat:.4f}, p_value={p_value:.4f}"
                    )
            
            elif method == 'psi':
                # Population Stability Index for any data type
                psi_value = calculate_psi(ref_series, cur_series)
                drift_detected = psi_value > config.drift.psi_threshold
                
                drift_results[col] = {
                    'psi': psi_value,
                    'drift_detected': drift_detected,
                    'method': 'psi',
                    'samples': {'reference': len(ref_series), 'current': len(cur_series)}
                }
                
                if drift_detected:
                    logger.warning(f"PSI detected drift in '{col}': PSI={psi_value:.4f}")
            
            else:
                # Skip non-numeric columns for KS test
                if method == 'ks' and not is_numeric:
                    logger.debug(f"Skipping non-numeric column '{col}' for KS test")
                    drift_results[col] = {
                        'error': 'KS test not applicable for non-numeric data',
                        'drift_detected': False
                    }
                    
        except Exception as e:
            errors_count += 1
            logger.error(f"Error processing column '{col}': {str(e)}", exc_info=True)
            drift_results[col] = {'error': str(e), 'drift_detected': False}
    
    logger.info(
        f"Drift comparison completed. Processed {len(drift_results)} columns "
        f"with {errors_count} errors"
    )
    
    # Log summary statistics
    drift_count = sum(1 for r in drift_results.values() if r.get('drift_detected', False))
    logger.info(f"Drift detected in {drift_count}/{len(drift_results)} features")
    
    return DriftReport(drift_results)