import os
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from typing import List
from datetime import datetime
from catboost import CatBoostClassifier
import pandas as pd
from sqlalchemy import create_engine

# Pydantic model for post data validation and serialization
# Used to format the API response with post recommendations
class PostGet(BaseModel):
    id: int
    text: str
    topic: str
    
    class Config:
        from_attributes = True

# Function for batch loading large volumes of data from PostgreSQL
# Splits the query into chunks of 200000 rows to save memory
# Returns a concatenated DataFrame with all the data
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        ################################
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

# Function to determine the model file path based on the environment
# Adapts paths for both local environment and work enviroment
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # If running in LMS
        MODEL_PATH = '/workdir/user_input/model'
    else:  # If running locally
        MODEL_PATH = path
    return MODEL_PATH

# Function to load the trained CatBoost model
# Contains error handling for potential model loading issues
def load_models():
    model_path = get_model_path("catboost_hitrate_model.cbm")
    try:
        from_file = CatBoostClassifier()
        model = from_file.load_model(model_path)
        return model
    except Exception as e:
        print(f": {e}")
        return None

# Function to load and preprocess all necessary features
# Loads user interaction history, user data, and post texts
# Merges likes and views, removes duplicates, prepares data for the model
def load_features():
    # Load user likes (target variable = 1)
    x1 = batch_load_sql("SELECT DISTINCT post_id, user_id, target FROM public.feed_data WHERE action='like' LIMIT 1000000")
    x1.loc[(x1['target'].isnull()), 'target'] = 1
    
    # Load user views
    x2 = batch_load_sql("SELECT DISTINCT post_id, user_id, target FROM public.feed_data WHERE action='view' LIMIT 1000000")
    
    # Merge likes and views
    x = pd.concat([x1, x2], axis=0)

    # Sort so that likes (target=1) appear above views (target=0)
    x = x.sort_values('target', ascending=False)  # 1 > 0 > NaN
    # Remove duplicates, keeping the first occurrence (like has priority over view)
    x = x.drop_duplicates(subset=['user_id', 'post_id'], keep='first')

    # Load user data (age, experimental group)
    data_user = batch_load_sql("SELECT age, exp_group, user_id FROM user_data")
    
    # Load post data (text, topic)
    post_text_df = batch_load_sql("SELECT * FROM post_text_df")

    return x, data_user, post_text_df

# Load all features at application startup (once)
x, data_user, post_text_df = load_features()

# Determine path and load the model at application startup
model_path = get_model_path("catboost_hitrate_model.cbm")
from_file = CatBoostClassifier()
model = from_file.load_model(model_path)

# Initialize FastAPI application
app = FastAPI()

# Endpoint for getting personalized post recommendations for a user
# Accepts user ID, request time, and recommendation limit
# Returns a list of recommended posts with their ID, text, and topic
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
    id: int, 
    time: datetime, 
    limit: int = 10
) -> List[PostGet]:
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found")
    
    # Get specific user data
    user_data = data_user[data_user['user_id'] == id]
    
    # Get posts that the user has already liked (exclude from recommendations)
    liked_posts = x[(x['user_id'] == id) & (x['target'] == 1)]['post_id'].unique()
    
    # Take all posts, excluding already liked ones
    X = post_text_df
    X = X[~X['post_id'].isin(liked_posts)]
    
    # Add user features to each post
    X['age'] = user_data.iloc[0, 0]
    X['exp_group'] = user_data.iloc[0, 1]
    
    # Convert categorical features to the required type
    columns_to_convert = ['post_id', 'age', 'exp_group', 'topic']
    X[columns_to_convert] = X[columns_to_convert].astype('object')
    
    # Remove text field before passing to the model
    pred_X = X.drop(['text'], axis=1)
    
    # Get like probability predictions for each post
    predictions = model.predict_proba(pred_X)[:, 1]

    # Create final DataFrame with predictions
    df_final = X.copy()
    df_final = df_final[~df_final['post_id'].isin(liked_posts)]
    df_final['target'] = predictions

    # Sort by probability and take top-limit posts
    post_list = df_final.sort_values(['target'], ascending=[False])['post_id'].head(limit).reset_index(drop=True)
    
    # Format the response in PostGet format
    recommended_posts = [
        PostGet(
            id=int(post_id),
            text=df_final[df_final['post_id'] == post_id]['text'].iloc[0],
            topic=df_final[df_final['post_id'] == post_id]['topic'].iloc[0]
        )
        for post_id in post_list
    ]
    
    return recommended_posts
