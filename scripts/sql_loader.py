"""SQL data loader with security improvements."""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Predefined safe queries
SAFE_QUERIES = {
    "hospital_costs": "SELECT age, income, race, true_cost, predicted_cost FROM hospital_costs",
    "all_hospital": "SELECT * FROM hospital_costs",
    "demographics": "SELECT age, race, income FROM hospital_costs",
    "predictions": "SELECT true_cost, predicted_cost FROM hospital_costs"
}


def load_audit_data_from_sql(
    db_path: str = "data/hospital_audit.db",
    table: str = "hospital_costs",
    columns: Optional[list] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Load data from SQL database with security protections.
    
    Args:
        db_path: Path to SQLite database file
        table: Table name (validated against whitelist)
        columns: List of column names to select (optional)
        limit: Maximum number of rows to return (optional)
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        ValueError: If inputs are invalid
        FileNotFoundError: If database file doesn't exist
    """
    # Validate database path
    db_path_obj = Path(db_path)
    if not db_path_obj.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    # Validate table name (whitelist approach)
    allowed_tables = ["hospital_costs", "audit_results", "model_predictions"]
    if table not in allowed_tables:
        raise ValueError(f"Table '{table}' not allowed. Allowed tables: {allowed_tables}")
    
    try:
        conn = sqlite3.connect(str(db_path_obj))
        
        # Build query safely
        if columns:
            # Validate column names
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table})")
            valid_columns = {row[1] for row in cursor.fetchall()}
            
            for col in columns:
                if col not in valid_columns:
                    raise ValueError(f"Invalid column '{col}' for table '{table}'")
            
            columns_str = ", ".join(columns)
            query = f"SELECT {columns_str} FROM {table}"
        else:
            query = f"SELECT * FROM {table}"
        
        # Add limit if specified
        if limit and isinstance(limit, int) and limit > 0:
            query += f" LIMIT {limit}"
        
        logger.info(f"Executing query: {query}")
        df = pd.read_sql_query(query, conn)
        
        logger.info(f"Loaded {len(df)} rows from {table}")
        return df
        
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()


def load_predefined_query(
    db_path: str = "data/hospital_audit.db",
    query_name: str = "hospital_costs"
) -> pd.DataFrame:
    """
    Load data using a predefined safe query.
    
    Args:
        db_path: Path to SQLite database file
        query_name: Name of predefined query
        
    Returns:
        pd.DataFrame: Query results
    """
    if query_name not in SAFE_QUERIES:
        raise ValueError(f"Unknown query '{query_name}'. Available: {list(SAFE_QUERIES.keys())}")
    
    db_path_obj = Path(db_path)
    if not db_path_obj.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    try:
        conn = sqlite3.connect(str(db_path_obj))
        query = SAFE_QUERIES[query_name]
        
        logger.info(f"Executing predefined query '{query_name}'")
        df = pd.read_sql_query(query, conn)
        
        logger.info(f"Loaded {len(df)} rows")
        return df
        
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()