import pandas as pd
from sqlalchemy import create_engine
import os

# Create a local SQLite database engine
db_path = "sqlite:///claims_database.db"
engine = create_engine(db_path)

def load_csv_to_sqlite(csv_file, table_name):
    print(f"Loading {csv_file} into table '{table_name}'...")
    df = pd.read_csv(f"./data/csv/{csv_file}")
    # Write to SQLite, replacing the table if it exists
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"Success! {len(df)} rows loaded.")

if __name__ == "__main__":
    # Ensure the directory exists
    if not os.path.exists("./data/csv"):
        print("Please place your CSVs in the ./data/csv directory.")
    else:
        load_csv_to_sqlite("policy_db.csv", "policies")
        load_csv_to_sqlite("claims_db.csv", "claims")
        load_csv_to_sqlite("history_db.csv", "claim_history")
        print("\nStructured database created successfully at claims_database.db")