from sklearn.model_selection import train_test_split

from lib import (
    load_data, preprocess_data, prepare_features, scale_features,
    visualize_data, train_random_forest, train_gradient_boosting,
    evaluate_model, get_feature_importance, analyze_predictions
)
from config import TEST_SIZE, RANDOM_STATE


def main():
    """Основная функция выполнения."""
    df = load_data()
    
    df = preprocess_data(df)
    
    visualize_data(df)
    
    X, y = prepare_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    print("\n" + "="*50)
    print("ОБУЧЕНИЕ RANDOM FOREST")
    print("="*50)
    rf_model = train_random_forest(X_train_scaled, y_train)
    rf_pred = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    get_feature_importance(rf_model, X.columns)
    
    print("\n" + "="*50)
    print("ОБУЧЕНИЕ GRADIENT BOOSTING")
    print("="*50)
    gb_model = train_gradient_boosting(X_train_scaled, y_train)
    gb_pred = evaluate_model(gb_model, X_test_scaled, y_test, "Gradient Boosting")
    
    analyze_predictions(df, y_test, gb_pred)


if __name__ == "__main__":
    main()
