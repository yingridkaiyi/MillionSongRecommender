import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Get the project root directory
            project_root = Path(__file__).resolve().parent.parent.parent
            db_path = str(project_root / "data" / "extracted.db")
        self.db_path = db_path
        self._create_indices()
    
    def _create_indices(self):
        """Create indices for frequently queried columns"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Create indices for the most commonly filtered columns
            indices = [
                "CREATE INDEX IF NOT EXISTS idx_danceability ON extracted(danceability)",
                "CREATE INDEX IF NOT EXISTS idx_energy ON extracted(energy)",
                "CREATE INDEX IF NOT EXISTS idx_valence ON extracted(valence)",
                "CREATE INDEX IF NOT EXISTS idx_acousticness ON extracted(acousticness)",
                "CREATE INDEX IF NOT EXISTS idx_instrumentalness ON extracted(instrumentalness)",
                "CREATE INDEX IF NOT EXISTS idx_speechiness ON extracted(speechiness)"
            ]
            for index_sql in indices:
                cursor.execute(index_sql)
            conn.commit()
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def get_table_info(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            table_info = {}
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                table_info[table_name] = [col[1] for col in columns]
        return table_info
    
    def get_song_features(self, limit:int = None):
        # Only select the columns we actually use
        query = """
        SELECT 
            track_name,
            artist_name,
            danceability,
            energy,
            valence,
            tempo,
            acousticness,
            instrumentalness,
            speechiness
        FROM extracted
        WHERE 
            danceability IS NOT NULL
            AND energy IS NOT NULL
            AND valence IS NOT NULL
            AND acousticness IS NOT NULL
            AND instrumentalness IS NOT NULL
            AND speechiness IS NOT NULL
        """
        if limit:
            query += f" LIMIT {limit}"
            
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)