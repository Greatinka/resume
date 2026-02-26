from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException
import uvicorn

from lib.data.preprocessor import load_and_preprocess_features
from lib.models.recommender import (
    load_model,
    prepare_features_for_user,
    get_predictions,
    get_top_recommendations,
    format_recommendations
)
from lib.schemas.post import PostGet
from config import DEFAULT_RECOMMENDATION_LIMIT, API_HOST, API_PORT

# Global variables for loaded data and model
interactions_df = None
user_df = None
post_df = None
model = None

interactions_df, user_df, post_df = load_and_preprocess_features()

model = load_model()
if model is None:
    print("WARNING: Model could not be loaded. API will return errors.")
else:
    print("Model loaded successfully.")

# Initialize FastAPI application
app = FastAPI(
    title="Post Recommendation API",
    description="API for personalized post recommendations",
    version="1.0.0"
)


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
    id: int, 
    time: datetime, 
    limit: int = DEFAULT_RECOMMENDATION_LIMIT
) -> List[PostGet]:
    """
    Get personalized post recommendations for a user.
    
    Args:
        id: User ID
        time: Request timestamp (for logging/analysis)
        limit: Maximum number of recommendations to return
        
    Returns:
        List of recommended posts with id, text, and topic
        
    Raises:
        HTTPException: If model is not loaded or user not found
    """

    if model is None:
        raise HTTPException(status_code=500, detail="Model not found")
    
    if id not in user_df['user_id'].values:
        raise HTTPException(status_code=404, detail=f"User {id} not found")

    features_df, liked_posts = prepare_features_for_user(
        user_id=id,
        post_df=post_df,
        user_df=user_df,
        interactions_df=interactions_df
    )
    
    if features_df.empty:
        return []
    
    predictions_df = get_predictions(model, features_df)
    
    top_post_ids = get_top_recommendations(predictions_df, limit)
  
    recommendations = format_recommendations(top_post_ids, post_df)
    
    return [PostGet(**rec) for rec in recommendations]


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": all(df is not None for df in [interactions_df, user_df, post_df])
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False
    )
