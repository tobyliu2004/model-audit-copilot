import sqlite3
import pandas as pd

# Load your existing cur_df as mock data
df = pd.read_csv("/Users/tobyliu/model_audit_copilot/data/cur_df.csv")

# Connect to SQLite DB (creates file if not exists)
conn = sqlite3.connect("data/hospital_audit.db")

# Write table
df.to_sql("hospital_costs", conn, index=False, if_exists="replace")

conn.close()
