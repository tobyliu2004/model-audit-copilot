# copilot/leakage/leakage_detector.py

import pandas as pd
import hashlib

def detect_target_leakage(df: pd.DataFrame, target_col: str, threshold: float = 0.9):
    """
    Detect features that are highly correlated with the target — possible leakage.
    """
    results = []

    for col in df.columns:
        if col == target_col:
            continue
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                corr = df[col].corr(df[target_col])
                if abs(corr) > threshold:
                    results.append({
                        "feature": col,
                        "correlation": round(corr, 4),
                        "leakage_flag": True
                    })
        except Exception:
            continue

    return pd.DataFrame(results)

def detect_duplicate_and_id_leakage(df: pd.DataFrame):
    """
    Check for duplicate rows and high-cardinality ID-like columns.
    """
    results = {
        "duplicate_rows": int(df.duplicated().sum()),
        "id_like_columns": []
    }

    for col in df.columns:
        if df[col].nunique() == len(df):
            results["id_like_columns"].append(col)

    return results

def detect_train_test_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame, subset_cols=None):
    """
    Detect overlapping rows between train and test sets.
    Uses row hashing for fast comparison.
    """
    if subset_cols is not None:
        train_df = train_df[subset_cols]
        test_df = test_df[subset_cols]

    def hash_row(row):
        return hashlib.sha256(str(row.values).encode()).hexdigest()

    train_hashes = set(train_df.apply(hash_row, axis=1))
    test_hashes = set(test_df.apply(hash_row, axis=1))

    overlap = train_hashes & test_hashes
    overlap_count = len(overlap)

    return {
        "train_size": len(train_df),
        "test_size": len(test_df),
        "overlap_count": overlap_count,
        "percent_overlap": round(100 * overlap_count / len(test_df), 2)
    }
