import pandas as pd
from config import DATA_FILE, INDEX_COLUMN


def load_data() -> pd.DataFrame:
    """Загружает датасет German Credit."""
    df = pd.read_csv(DATA_FILE)
    if INDEX_COLUMN in df.columns:
        df.drop(columns=[INDEX_COLUMN], axis=1, inplace=True)
    return df
