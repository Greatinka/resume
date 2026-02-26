from typing import List

import pandas as pd
from sqlalchemy import create_engine, text

from config import DB_CONNECTION_STRING, DEFAULT_CHUNKSIZE


def create_db_engine():
    """Create database connection engine."""
    return create_engine(DB_CONNECTION_STRING)


def batch_load_sql(query: str, chunksize: int = DEFAULT_CHUNKSIZE) -> pd.DataFrame:
    """Batch load large volumes of data from PostgreSQL """
    
    engine = create_db_engine()
    conn = engine.connect().execution_options(stream_results=True)
    
    chunks: List[pd.DataFrame] = []
    for chunk_dataframe in pd.read_sql(text(query), conn, chunksize=chunksize):
        chunks.append(chunk_dataframe)
    
    conn.close()
    engine.dispose()
    
    return pd.concat(chunks, ignore_index=True)


def load_likes_data() -> pd.DataFrame:
    """Load user likes from database"""
    
    from config import LIKE_QUERY
    
    x1 = batch_load_sql(LIKE_QUERY)
    x1.loc[x1['target'].isnull(), 'target'] = 1
    return x1


def load_views_data() -> pd.DataFrame:
    """Load user views from database"""
    
    from config import VIEW_QUERY
    
    return batch_load_sql(VIEW_QUERY)


def load_user_data() -> pd.DataFrame:
    """Load user demographic data"""
    
    from config import USER_QUERY
    
    return batch_load_sql(USER_QUERY)


def load_post_data() -> pd.DataFrame:
    """Load post content data"""
    
    from config import POST_QUERY
    
    return batch_load_sql(POST_QUERY)
