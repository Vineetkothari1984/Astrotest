"""Migration script to populate SQLite database from Excel files."""
import pandas as pd
from pathlib import Path
from database import get_connection

EXCEL_TABLES = {
    'doc.xlsx': 'stocks',
    'numerology.xlsx': 'numerology',
    'nifty.xlsx': 'nifty',
    'banknifty.xlsx': 'banknifty',
    'moon.xlsx': 'moon',
    'mercury.xlsx': 'mercury',
    'mercurycom.xlsx': 'mercurycom',
    'panchak.xlsx': 'panchak',
}

DATE_COLUMNS = {
    'stocks': ['NSE LISTING DATE', 'BSE LISTING DATE', 'DATE OF INCORPORATION'],
    'numerology': ['date'],
    'moon': ['Date'],
    'mercury': ['Date'],
    'mercurycom': ['Start Date', 'End Date'],
    'panchak': ['Start Date', 'End Date'],
    'nifty': ['Date'],
    'banknifty': ['Date'],
}


def migrate():
    with get_connection() as conn:
        for file, table in EXCEL_TABLES.items():
            path = Path(file)
            if not path.exists():
                continue
            if table in ['nifty', 'banknifty']:
                df = pd.read_excel(path, index_col=0)
                df.index = pd.to_datetime(df.index)
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'Date'}, inplace=True)
            else:
                df = pd.read_excel(path)
            for col in DATE_COLUMNS.get(table, []):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
            df.to_sql(table, conn, if_exists='replace', index=False)

if __name__ == '__main__':
    migrate()
