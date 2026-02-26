from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import joblib

from config import (
    ESTIMATORS, MIN_SAMPLES_LEAF, RF_MAX_DEPTH, GB_MAX_DEPTH,
    RANDOM_STATE, LEARNING_RATE, RF_MODEL_PATH, GB_MODEL_PATH
)


def train_random_forest(X_train, y_train):
    """Обучает Random Forest с подбором гиперпараметров."""
    grid_values = {
        'n_estimators': ESTIMATORS,
        'min_samples_leaf': MIN_SAMPLES_LEAF,
        'max_depth': RF_MAX_DEPTH
    }
    
    clf = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        grid_values, scoring='roc_auc', cv=5
    )
    clf.fit(X_train, y_train)
    
    print('Оптимальное количество деревьев:', clf.best_params_['n_estimators'])
    print('Оптимальное количество листьев:', clf.best_params_['min_samples_leaf'])
    print('Оптимальная глубина:', clf.best_params_['max_depth'])
    print('AUC-ROC:', clf.best_score_)
    
    model = RandomForestClassifier(
        n_estimators=clf.best_params_['n_estimators'],
        max_depth=clf.best_params_['max_depth'],
        min_samples_leaf=clf.best_params_['min_samples_leaf'],
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    
    joblib.dump(model, RF_MODEL_PATH)
    
    return model


def train_gradient_boosting(X_train, y_train):
    """Обучает Gradient Boosting с подбором гиперпараметров."""
    grid_values = {
        'n_estimators': ESTIMATORS,
        'min_samples_leaf': MIN_SAMPLES_LEAF,
        'max_depth': GB_MAX_DEPTH
    }
    
    clf = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        grid_values, scoring='roc_auc', cv=5
    )
    clf.fit(X_train, y_train)
    
    print('Оптимальное количество деревьев:', clf.best_params_['n_estimators'])
    print('Оптимальное количество листьев:', clf.best_params_['min_samples_leaf'])
    print('Оптимальная глубина:', clf.best_params_['max_depth'])
    print('AUC-ROC:', clf.best_score_)
    
    model = GradientBoostingClassifier(
        n_estimators=clf.best_params_['n_estimators'],
        learning_rate=LEARNING_RATE,
        min_samples_leaf=clf.best_params_['min_samples_leaf'],
        max_depth=clf.best_params_['max_depth'],
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    
    joblib.dump(model, GB_MODEL_PATH)
    
    return model
