import sqlite3
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent / 'data.db'

@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def read_table(table: str):
    import pandas as pd
    with get_connection() as conn:
        df = pd.read_sql_query(f'SELECT * FROM {table}', conn)
    return df
