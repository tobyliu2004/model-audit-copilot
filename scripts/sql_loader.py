# scripts/sql_loader.py

import sqlite3
import pandas as pd

def load_audit_data_from_sql(db_path="data/hospital_audit.db", query=None):
    if query is None:
        query = "SELECT age, income, race, true_cost, predicted_cost FROM hospital_costs;"
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df