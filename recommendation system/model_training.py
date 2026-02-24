import pandas as pd
from sqlalchemy import text
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
import numpy as np
from sqlalchemy import create_engine

# Set display options for better DataFrame visualization in console
pd.options.display.max_rows = 100
pd.options.display.max_columns = 50

# Create database connection engine
engine = create_engine(
            ###############################
        )
    
# Establish database connection and load all necessary data
with engine.connect() as conn:
    # Load user likes (positive interactions)
    x1 = pd.read_sql(text("SELECT DISTINCT post_id, user_id, timestamp, target FROM public.feed_data WHERE action='like' LIMIT 600000"), conn)
    # Set target=1 for likes (if null values exist)
    x1.loc[(x1['target'].isnull()), 'target'] = 1
    
    # Load user views (neutral interactions)
    x2 = pd.read_sql(text("SELECT DISTINCT post_id, user_id, timestamp, target FROM public.feed_data WHERE action='view' LIMIT 1400000"), conn)
    
    # Combine likes and views into single dataset
    x = pd.concat([x1, x2], axis=0)
    
    # Load user demographic data
    data_user = pd.read_sql(text("SELECT age, exp_group, user_id FROM user_data"), conn)
    
    # Load post content data (text and topic)
    post_text_df = pd.read_sql(text("SELECT * FROM post_text_df"), conn)

# Close database connection to free resources
engine.dispose()

# Merge all data sources: user interactions + user features + post features
merged_data = x.merge(data_user, on='user_id', how='left')
merged_data = merged_data.merge(post_text_df, on='post_id', how='left')

# Prepare features (X) and target variable (y)
# Drop columns that shouldn't be used as features
X = merged_data.drop(columns=['target', 'text', 'user_id', 'timestamp'])
y = merged_data['target']

# Convert categorical columns to 'object' type for CatBoost
columns_to_convert = ['post_id', 'age', 'exp_group', 'topic']
X[columns_to_convert] = X[columns_to_convert].astype('object')

# Display first rows and data types for verification
print(X.head())
print(X.dtypes)

# Split data into training and testing sets with stratification
# to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Identify categorical features for CatBoost
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Create CatBoost Pool objects for efficient data handling
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
test_pool = Pool(X_test, y_test, cat_features=categorical_features)

# Custom metric class for HitRate@5 calculation
# Used for evaluating recommendation quality
class HitRateAt5Metric(object):
    def get_final_error(self, error, weight):
        """Calculate final error value (average hitrate)"""
        return error / weight if weight != 0 else 0

    def is_max_optimal(self):
        """Indicates that higher metric values are better"""
        return True

    def evaluate(self, approxes, target, weight):
        """
        Evaluate HitRate@5 for a single data batch
        approxes: list of predictions for each class
        target: true labels
        weight: sample weights
        """
        # Validate input for binary classification
        if len(approxes) != 1:
            raise ValueError("Expected one approx array for binary classification")
        
        approx = approxes[0]
        
        # Sort predictions in descending order and take top 5
        sorted_indices = np.argsort(approx)[::-1]
        top_5_indices = sorted_indices[:5]
        
        # Check if any of top 5 predictions is a like (target=1)
        hit = 0
        for idx in top_5_indices:
            if target[idx] == 1:
                hit = 1
                break
        
        return hit, 1  # Return hit value and weight

# Initialize CatBoost classifier with optimized parameters
model = CatBoostClassifier(
    iterations=300,           # Reduced from 1000 for faster training
    learning_rate=0.1,        # Step size for gradient descent
    depth=5,                   # Tree depth (controls model complexity)
    loss_function='Logloss',   # Binary cross-entropy loss
    eval_metric='AUC',         # Area Under ROC Curve for monitoring
    random_seed=42,            # For reproducibility
    verbose=100,               # Print progress every 100 iterations
    
    # Regularization parameters to reduce overfitting and model size
    l2_leaf_reg=3,             # L2 regularization coefficient
    random_strength=1,         # Amount of randomness for split selection
    bagging_temperature=0.5,   # Bayesian bagging temperature
    leaf_estimation_iterations=1,  # Reduce computation time
    grow_policy='Lossguide',   # More efficient tree growth strategy
    max_leaves=64,             # Limit tree complexity
    
    # Early stopping to prevent overfitting
    early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
    use_best_model=True        # Use best model from training
)

# Train the model with validation set
model.fit(
    train_pool,
    eval_set=test_pool,
    early_stopping_rounds=50,  # Stop early if validation metric doesn't improve
    verbose=100                # Print progress
)

# Get prediction probabilities for test set
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Function to calculate Hitrate@5 metric
# Measures if at least one relevant item appears in top 5 recommendations
def calculate_hitrate_at5(y_true, y_pred_proba, user_ids=None):
    """
    Calculate Hitrate@5 metric
    
    Parameters:
    y_true: true labels (1 for likes, 0 for views)
    y_pred_proba: predicted probabilities
    user_ids: user identifiers for grouping (optional)
    
    Returns:
    Hitrate@5 value (0 to 1)
    """
    if user_ids is None:
        # Global hitrate calculation (all users combined)
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        top_5_indices = sorted_indices[:5]
        hit = np.any(y_true.iloc[top_5_indices] == 1)
        return hit
    else:
        # Per-user hitrate calculation and averaging
        df = pd.DataFrame({
            'user_id': user_ids,
            'y_true': y_true,
            'y_pred_proba': y_pred_proba
        })
        
        hitrates = []
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id]
            # Only consider users with at least 5 posts
            if len(user_data) >= 5:
                # Take top 5 predictions for this user
                top_5 = user_data.nlargest(5, 'y_pred_proba')
                # Check if any of top 5 is a like
                hit = np.any(top_5['y_true'] == 1)
                hitrates.append(hit)
        
        # Return average hitrate across users
        return np.mean(hitrates) if hitrates else 0

# Calculate and display Hitrate@5 for test set
hitrate_at5 = calculate_hitrate_at5(y_test, y_pred_proba)
print(f"Hitrate@5: {hitrate_at5:.4f}")

# Get feature importance to understand model decisions
feature_importance = model.get_feature_importance()
feature_names = X_train.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Display top 10 most important features
print("\nTop-10 most important features:")
print(importance_df.head(10))
