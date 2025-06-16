import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def load_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(BASE_DIR / path)
