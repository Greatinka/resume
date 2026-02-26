from typing import Tuple, Optional

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

from lib.utils.path_utils import get_model_path
from config import MODEL_FILENAME, CATEGORICAL_COLUMNS


def load_model(model_path: str = MODEL_FILENAME) -> Optional[CatBoostClassifier]:
    """Load the trained CatBoost model"""
    
    full_path = get_model_path(model_path)
    model = CatBoostClassifier()
    model.load_model(full_path)
    return model


def get_user_data(user_id: int, user_df: pd.DataFrame) -> pd.Series:
    """Get data for a specific user"""
    
    return user_df[user_df['user_id'] == user_id].iloc[0]


def get_liked_posts(user_id: int, interactions_df: pd.DataFrame) -> np.ndarray:
    """Get posts that a user has liked"""
    
    return interactions_df[
        (interactions_df['user_id'] == user_id) & 
        (interactions_df['target'] == 1)
    ]['post_id'].unique()


def prepare_features_for_user(
    user_id: int,
    post_df: pd.DataFrame,
    user_df: pd.DataFrame,
    interactions_df: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Prepare features for a specific user for prediction"""
    
    user_data = get_user_data(user_id, user_df)
    liked_posts = get_liked_posts(user_id, interactions_df)
    
    features_df = post_df[~post_df['post_id'].isin(liked_posts)].copy()
    
    features_df['age'] = user_data['age']
    features_df['exp_group'] = user_data['exp_group']
    
    features_df[CATEGORICAL_COLUMNS] = features_df[CATEGORICAL_COLUMNS].astype('object')
    
    return features_df, liked_posts


def get_predictions(
    model: CatBoostClassifier, 
    features_df: pd.DataFrame
) -> pd.DataFrame:
    """Get predictions from the model for prepared features"""
    
    pred_features = features_df.drop(['text'], axis=1)
    predictions = model.predict_proba(pred_features)[:, 1]
    
    result_df = features_df.copy()
    result_df['prediction_score'] = predictions
    
    return result_df


def get_top_recommendations(
    predictions_df: pd.DataFrame,
    limit: int = 10
) -> pd.Series:
    """Get top N recommendations sorted by prediction score"""
    
    return predictions_df.sort_values(
        'prediction_score', ascending=False
    )['post_id'].head(limit).reset_index(drop=True)


def format_recommendations(
    post_ids: pd.Series,
    post_df: pd.DataFrame
) -> list:
    """Format recommendations for API response"""
    
    recommendations = []
    
    for post_id in post_ids:
        post_data = post_df[post_df['post_id'] == post_id].iloc[0]
        recommendations.append({
            'id': int(post_id),
            'text': post_data['text'],
            'topic': post_data['topic']
        })
    
    return recommendations
