from typing import List

import pandas as pd
from sqlalchemy import create_engine, text

from config import DB_CONNECTION_STRING, DEFAULT_CHUNKSIZE


def create_db_engine():
    """Create database connection engine."""
    return create_engine(DB_CONNECTION_STRING)


def batch_load_sql(query: str, chunksize: int = DEFAULT_CHUNKSIZE) -> pd.DataFrame:
    """
    Batch load large volumes of data from PostgreSQL.
    
    Splits the query into chunks to save memory and returns a concatenated DataFrame.
    
    Args:
        query: SQL query to execute
        chunksize: Number of rows to load per chunk
        
    Returns:
        DataFrame containing all query results
    """
    engine = create_db_engine()
    conn = engine.connect().execution_options(stream_results=True)
    
    chunks: List[pd.DataFrame] = []
    for chunk_dataframe in pd.read_sql(text(query), conn, chunksize=chunksize):
        chunks.append(chunk_dataframe)
    
    conn.close()
    engine.dispose()
    
    return pd.concat(chunks, ignore_index=True)


def load_likes_data() -> pd.DataFrame:
    """
    Load user likes from database.
    
    Returns:
        DataFrame with likes data (target=1)
    """
    from config import LIKE_QUERY
    
    x1 = batch_load_sql(LIKE_QUERY)
    x1.loc[x1['target'].isnull(), 'target'] = 1
    return x1


def load_views_data() -> pd.DataFrame:
    """
    Load user views from database.
    
    Returns:
        DataFrame with views data
    """
    from config import VIEW_QUERY
    
    return batch_load_sql(VIEW_QUERY)


def load_user_data() -> pd.DataFrame:
    """
    Load user demographic data.
    
    Returns:
        DataFrame with user data (age, exp_group)
    """
    from config import USER_QUERY
    
    return batch_load_sql(USER_QUERY)


def load_post_data() -> pd.DataFrame:
    """
    Load post content data.
    
    Returns:
        DataFrame with post data (text, topic)
    """
    from config import POST_QUERY
    
    return batch_load_sql(POST_QUERY)
