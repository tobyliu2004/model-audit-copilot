"""Set up SQLite database with sample data."""

import sqlite3
import pandas as pd
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_database(csv_path: str, db_path: str = "data/hospital_audit.db", table_name: str = "hospital_costs"):
    """
    Create SQLite database from CSV file.
    
    Args:
        csv_path: Path to CSV file to import
        db_path: Path for SQLite database file
        table_name: Name of the table to create
    """
    # Validate paths
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Connect to SQLite DB (creates file if not exists)
    logger.info(f"Creating database at {db_path}")
    conn = sqlite3.connect(str(db_path))
    
    # Write table
    df.to_sql(table_name, conn, index=False, if_exists="replace")
    logger.info(f"Created table '{table_name}' with {len(df)} records")
    
    # Verify
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    logger.info(f"Verified: {count} records in database")
    
    conn.close()
    logger.info("Database setup complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up SQLite database from CSV")
    parser.add_argument("csv_file", help="Path to CSV file to import")
    parser.add_argument("--db", default="data/hospital_audit.db", help="Path for SQLite database")
    parser.add_argument("--table", default="hospital_costs", help="Table name to create")
    
    args = parser.parse_args()
    
    try:
        setup_database(args.csv_file, args.db, args.table)
    except Exception as e:
        logger.error(f"Failed to set up database: {e}")
        exit(1)