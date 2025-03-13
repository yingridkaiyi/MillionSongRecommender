import sqlite3
import pandas as pd

def explore_db(db_path:str):

    conn = sqlite3.connect(db_path)

    tables_query = """
    SELECT name FROM sqlite_master WHERE type='table';
    """

    tables = pd.read_sql_query(tables_query, conn)
    print("\nTables in the database:")
    print(tables)

    for table_name in tables['name']:
        print(f"\nStructure of table '{table_name}':")
        schema_query = f"PRAGMA table_info({table_name});"
        schema = pd.read_sql_query(schema_query, conn)
        print(schema)
    
            # Get a sample of data
        print(f"\nSample data from '{table_name}':")
        sample_query = f"SELECT * FROM {table_name} LIMIT 5;"
        sample_data = pd.read_sql_query(sample_query, conn)
        print(sample_data)
     
    conn.close()

if __name__ == "__main__":
    db_path =  "../../data/extracted.db" 
    explore_db(db_path)
    