# copilot/schema/schema_validator.py

import pandas as pd

def compare_schemas(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Compare schema of two DataFrames: column presence and data types.
    """
    schema1 = df1.dtypes.apply(lambda x: x.name).to_dict()
    schema2 = df2.dtypes.apply(lambda x: x.name).to_dict()

    all_keys = set(schema1.keys()).union(schema2.keys())
    report = []

    for col in sorted(all_keys):
        type1 = schema1.get(col, "MISSING")
        type2 = schema2.get(col, "MISSING")
        report.append({
            "column": col,
            "reference_type": type1,
            "current_type": type2,
            "mismatch": type1 != type2
        })

    return pd.DataFrame(report)

def find_missing_or_extra_columns(df1: pd.DataFrame, df2: pd.DataFrame):
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    return {
        "missing_in_current": list(cols1 - cols2),
        "extra_in_current": list(cols2 - cols1)
    }
