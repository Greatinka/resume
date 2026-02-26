from typing import Tuple

import pandas as pd

from lib.data.loader import (
    load_likes_data, load_views_data,
    load_user_data, load_post_data
)


def combine_interactions(likes_df: pd.DataFrame, views_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine likes and views into a single dataset with priority for likes.
    
    Args:
        likes_df: DataFrame with likes (target=1)
        views_df: DataFrame with views (target=0/null)
        
    Returns:
        Combined DataFrame with duplicates removed (likes prioritized)
    """
    combined = pd.concat([likes_df, views_df], axis=0, ignore_index=True)
    combined = combined.sort_values('target', ascending=False)
    combined = combined.drop_duplicates(subset=['user_id', 'post_id'], keep='first')
    
    return combined


def load_and_preprocess_features() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess all necessary features for the recommendation system.
    
    Returns:
        Tuple containing (interactions_df, user_df, post_df)
    """
    likes_df = load_likes_data()
    views_df = load_views_data()
    user_df = load_user_data()
    post_df = load_post_data()
    
    interactions_df = combine_interactions(likes_df, views_df)
    
    return interactions_df, user_df, post_df
