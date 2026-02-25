import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# Load the German Credit dataset
df = pd.read_csv('german_credit_data.csv')
# Drop the unnecessary index column
df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

# Handle missing values in account-related columns
# Replace NaN with 'unknown' to preserve information about missing data
df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')

# Define categorical columns for one-hot encoding
categorical_columns = ['Sex', 'Housing', 'Checking account', 'Purpose']
# Define ordinal order for 'Saving accounts' column
order = [['unknown', 'little', 'moderate', 'quite rich', 'rich']]

# Initialize encoders
ohe = OneHotEncoder(handle_unknown='ignore')  # Will ignore unknown categories in test data
oe = OrdinalEncoder(categories=order)  # Ordinal encoding with predefined order

# Apply ordinal encoding to 'Saving accounts' column
df['Saving accounts'] = oe.fit_transform(df[['Saving accounts']])

# Apply one-hot encoding to categorical columns
encoded_data = ohe.fit_transform(df[categorical_columns]).toarray()

# Create DataFrame with encoded features
encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(categorical_columns))

# Concatenate original dataframe with encoded features and drop original categorical columns
df = pd.concat([df, encoded_df], axis=1).drop(categorical_columns, axis=1)

# Save encoders for future use (e.g., in production)
joblib.dump(ohe, 'one_hot_encoder.joblib')
joblib.dump(oe, 'ordinal_encoder.joblib')

# Replace infinite values with NaN to avoid numerical issues
df = df.replace([np.inf, -np.inf], np.nan)

# Visualize age distribution
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Visualize credit amount distribution
sns.histplot(df['Credit amount'], kde=True)
plt.title('Credit Amount Distribution')
plt.show()

# Create correlation matrix for numerical features
# Exclude one-hot encoded columns to avoid clutter
df_corr = df.drop(columns=ohe.get_feature_names_out(categorical_columns))
plt.figure(figsize=(10, 8))
sns.heatmap(df_corr.corr(), annot=True, cmap='BuPu')
plt.title('Feature Correlation Matrix')
plt.show()

# Prepare features and target variable
# Target: whether credit amount is above median (binary classification)
X = df.drop(columns=['Credit amount'])
y = df['Credit amount'] > df['Credit amount'].median()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and apply standard scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler for future use
joblib.dump(scaler, 'scaler.joblib')

# Define hyperparameters for Random Forest grid search
estimators = [10, 50, 100, 300, 400, 500, 600]  # Number of trees
min_samples_leaf = list(np.arange(1, 20, 5))  # Minimum samples in leaf nodes
max_depth = list(np.arange(1, 5, 1))  # Maximum tree depth

# Create grid search parameters dictionary
grid_values = {'n_estimators': estimators, 'min_samples_leaf': min_samples_leaf, 'max_depth': max_depth}

# Perform grid search with 5-fold cross-validation, optimizing for ROC-AUC
clf = GridSearchCV(RandomForestClassifier(), grid_values, scoring='roc_auc', cv=5)
clf.fit(X_train, y_train)

# Extract best parameters and score
best_n_estimators_value = clf.best_params_['n_estimators']
best_min_samples_leaf = clf.best_params_['min_samples_leaf']
best_max_depth = clf.best_params_['max_depth']
best_score = clf.best_score_

print('Optimal number of trees:', best_n_estimators_value)
print('Optimal minimum samples per leaf:', best_min_samples_leaf)
print('Optimal max depth:', best_max_depth)
print('AUC-ROC score:', best_score)

# Train final Random Forest model with best parameters
rand_clf = RandomForestClassifier(
    n_estimators=best_n_estimators_value,
    max_depth=best_max_depth,
    min_samples_leaf=best_min_samples_leaf,
    random_state=42
)
rand_clf.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = rand_clf.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC-AUC: {roc_auc}')

# Analyze feature importance
importances = rand_clf.feature_importances_
features = pd.DataFrame({"feature": X.columns, 'importance': importances})
features.sort_values(by='importance', ascending=False, inplace=True)

# Save Random Forest model
joblib.dump(rand_clf, 'credit_risk_random_forest_model.joblib')

# Calculate and display detailed metrics for Random Forest
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print('-----------------')
print("Confusion Matrix:")
print(conf_matrix)
print('-----------------')
print("Classification Report:")
print(class_report)

# Define hyperparameters for Gradient Boosting grid search
estimators = [10, 50, 100, 300, 400, 500, 600]  # Number of boosting stages
min_samples_leaf = list(np.arange(1, 20, 5))  # Minimum samples in leaf nodes
max_depth = list(np.arange(1, 10, 3))  # Maximum tree depth (wider range for boosting)

# Create grid search parameters dictionary
grid_values = {'n_estimators': estimators, 'min_samples_leaf': min_samples_leaf, 'max_depth': max_depth}

# Perform grid search for Gradient Boosting
clf = GridSearchCV(GradientBoostingClassifier(), grid_values, scoring='roc_auc', cv=5)
clf.fit(X_train, y_train)

# Extract best parameters and score for Gradient Boosting
best_n_estimators_value_boost = clf.best_params_['n_estimators']
best_min_samples_leaf_boost = clf.best_params_['min_samples_leaf']
best_max_depth_boost = clf.best_params_['max_depth']
best_score_boost = clf.best_score_

print('Optimal number of trees:', best_n_estimators_value_boost)
print('Optimal minimum samples per leaf:', best_min_samples_leaf_boost)
print('Optimal max depth:', best_max_depth_boost)
print('AUC-ROC score:', best_score_boost)

# Train final Gradient Boosting model with best parameters
gb_clf = GradientBoostingClassifier(
    n_estimators=best_n_estimators_value_boost,
    learning_rate=0.1,  # Standard learning rate for boosting
    min_samples_leaf=best_min_samples_leaf_boost,
    max_depth=best_max_depth_boost,
    random_state=42
)
gb_clf.fit(X_train, y_train)

# Make predictions and evaluate Gradient Boosting
y_pred = gb_clf.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC-AUC: {roc_auc}')

# Calculate and display detailed metrics for Gradient Boosting
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print('-----------------')
print("Confusion Matrix:")
print(conf_matrix)
print('-----------------')
print("Classification Report:")
print(class_report)

# Save the best Gradient Boosting model (overall best performer)
joblib.dump(gb_clf, 'best_credit_risk_model.joblib')

# Analyze predictions on test set
# Create a DataFrame with test data, predictions, and actual values
df_test = df.iloc[y_test.index].copy()
df_test['Prediction'] = y_pred
df_test['Actual'] = y_test.values

# Identify correctly classified instances
true_negatives = df_test[(df_test['Actual'] == 0) & (df_test['Prediction'] == 0)]  # Correctly predicted low credit
true_positives = df_test[(df_test['Actual'] == 1) & (df_test['Prediction'] == 1)]  # Correctly predicted high credit

print("True Negatives (correctly predicted low credit amount):")
display(true_negatives)

print("\nTrue Positives (correctly predicted high credit amount):")
display(true_positives)
