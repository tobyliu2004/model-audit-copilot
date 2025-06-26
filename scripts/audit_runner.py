"""Command-line interface for running model audits."""

import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

from copilot.config import get_config, load_config
from copilot.auditor.audit_orchestrator import AuditOrchestrator
from copilot.outliers.outlier_detector import detect_outliers
from scripts.reports_utils import generate_audit_markdown, save_report
from scripts.utils import validate_file_path, load_dataframe_safely

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AuditError(Exception):
    """Base exception for audit-related errors."""
    pass


class DataLoadError(AuditError):
    """Error loading data files."""
    pass


def load_dataset(file_path: str, dataset_name: str) -> pd.DataFrame:
    """
    Load a dataset with validation and error handling.
    
    Args:
        file_path: Path to the data file
        dataset_name: Name of the dataset for logging
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        DataLoadError: If data cannot be loaded
    """
    try:
        # Validate path
        safe_path = validate_file_path(file_path)
        logger.info(f"Loading {dataset_name} from {safe_path}")
        
        # Load data
        df = load_dataframe_safely(safe_path)
        
        # Validate loaded data
        if df.empty:
            raise DataLoadError(f"{dataset_name} is empty")
        
        logger.info(f"Successfully loaded {dataset_name}: shape={df.shape}")
        return df
        
    except Exception as e:
        if isinstance(e, DataLoadError):
            raise
        raise DataLoadError(f"Error loading {dataset_name}: {str(e)}")


def main(args):
    """Main function for audit runner."""
    exit_code = 0
    
    try:
        # Load configuration if specified
        if args.config:
            config = load_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            config = get_config()
        
        # Set log level from args
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.info("Starting audit process")
        logger.debug(f"Arguments: {vars(args)}")
        
        # Load datasets
        try:
            ref_df = load_dataset(args.reference, "Reference dataset") if args.reference else None
            cur_df = load_dataset(args.current, "Current dataset")
        except DataLoadError as e:
            logger.error(str(e))
            return 1
        
        # Initialize auditor
        try:
            auditor = AuditOrchestrator(ref_df, cur_df)
        except Exception as e:
            logger.error(f"Failed to initialize auditor: {e}")
            return 2
        
        # Run drift check
        if ref_df is not None:
            try:
                logger.info(f"Running drift check with method: {args.method}")
                auditor.run_drift_check(method=args.method)
            except Exception as e:
                logger.error(f"Drift check failed: {e}", exc_info=True)
                if not config.continue_on_error:
                    raise AuditError(f"Drift check failed: {e}")
        
        # Run fairness and leakage checks if target column specified
        if args.target:
            # Validate required columns exist
            required_cols = [args.target]
            if args.pred:
                required_cols.append(args.pred)
            if args.group:
                required_cols.append(args.group)
            
            missing_cols = [col for col in required_cols if col not in cur_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Run fairness check
            if args.pred and args.group:
                try:
                    logger.info("Running fairness check")
                    auditor.run_fairness_check(
                        y_true=cur_df[args.target],
                        y_pred=cur_df[args.pred],
                        group_feature=cur_df[args.group]
                    )
                except Exception as e:
                    logger.error(f"Fairness check failed: {e}", exc_info=True)
                    if not config.continue_on_error:
                        raise AuditError(f"Fairness check failed: {e}")
            
            # Run leakage check
            try:
                logger.info("Running leakage check")
                auditor.run_leakage_check(dataset=cur_df, target_col=args.target)
            except Exception as e:
                logger.error(f"Leakage check failed: {e}", exc_info=True)
                if not config.continue_on_error:
                    raise AuditError(f"Leakage check failed: {e}")
        
        # Detect outliers
        outliers = None
        if args.outliers > 0:
            try:
                logger.info(f"Detecting top {args.outliers} outliers")
                outliers = detect_outliers(cur_df, n_outliers=args.outliers)
            except Exception as e:
                logger.error(f"Outlier detection failed: {e}", exc_info=True)
                if not config.continue_on_error:
                    raise AuditError(f"Outlier detection failed: {e}")
        
        # Generate and save report
        try:
            logger.info("Generating audit report")
            markdown = generate_audit_markdown(
                drift=auditor.get_results().get("drift"),
                fairness=auditor.get_results().get("fairness"),
                leakage=auditor.get_results().get("leakage"),
                outliers=outliers
            )
            
            # Validate output path
            output_path = validate_file_path(args.output, must_exist=False)
            save_report(markdown, str(output_path))
            
            logger.info(f"Audit complete. Report saved to {output_path}")
            print(f"\n✅ Audit completed successfully! Report saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}", exc_info=True)
            raise AuditError(f"Report generation failed: {e}")
            
    except AuditError as e:
        logger.error(f"Audit failed: {e}")
        exit_code = 3
    except KeyboardInterrupt:
        logger.info("Audit interrupted by user")
        exit_code = 130
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        exit_code = 255
    
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run comprehensive model audit checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic drift check
  python audit_runner.py --current data/test.csv --reference data/train.csv
  
  # Full audit with fairness and leakage checks
  python audit_runner.py --current data/test.csv --reference data/train.csv \\
    --target actual_price --pred predicted_price --group customer_segment
  
  # Use custom configuration
  python audit_runner.py --current data/test.csv --config my_config.yaml
        """
    )
    
    # Data inputs
    parser.add_argument("--reference", help="Reference CSV (e.g., training data)")
    parser.add_argument("--current", required=True, help="Current CSV (e.g., test or production data)")
    
    # Audit options
    parser.add_argument("--method", default="ks", choices=["ks", "psi"], 
                       help="Drift detection method (default: ks)")
    parser.add_argument("--target", help="Target column (required for fairness/leakage)")
    parser.add_argument("--pred", help="Prediction column (required for fairness)")
    parser.add_argument("--group", help="Sensitive group column (required for fairness)")
    parser.add_argument("--outliers", type=int, default=10, 
                       help="Number of top outliers to report (default: 10, 0 to disable)")
    
    # Output options
    parser.add_argument("--output", default="audit_report.md", 
                       help="Output path for markdown report (default: audit_report.md)")
    
    # Configuration
    parser.add_argument("--config", help="Path to configuration file (YAML or JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    sys.exit(main(args))