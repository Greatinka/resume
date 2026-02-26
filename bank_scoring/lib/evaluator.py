import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, roc_auc_score
)


def evaluate_model(model, X_test, y_test, model_name="Модель"):
    """Оценивает модель и выводит метрики."""
    y_pred = model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"\nРезультаты {model_name}:")
    print(f'ROC-AUC: {roc_auc:.4f}')
    print(f"Точность: {accuracy:.2f}")
    print('-----------------')
    print("Матрица ошибок:")
    print(conf_matrix)
    print('-----------------')
    print("Отчет по классификации:")
    print(class_report)
    
    return y_pred


def get_feature_importance(model, feature_names):
    """Получает и отображает важность признаков."""
    importances = model.feature_importances_
    features = pd.DataFrame({
        "признак": feature_names,
        'важность': importances
    }).sort_values('важность', ascending=False)
    
    print("\nТоп-10 наиболее важных признаков:")
    print(features.head(10))
    
    return features


def analyze_predictions(df, y_test, y_pred):
    """Анализирует предсказания на тестовой выборке."""
    df_test = df.iloc[y_test.index].copy()
    df_test['Предсказание'] = y_pred
    df_test['Факт'] = y_test.values
    
    true_negatives = df_test[(df_test['Факт'] == 0) & (df_test['Предсказание'] == 0)]
    true_positives = df_test[(df_test['Факт'] == 1) & (df_test['Предсказание'] == 1)]
    
    print("\nИстинно отрицательные (верно предсказаны низкие кредиты):")
    print(true_negatives)
    
    print("\nИстинно положительные (верно предсказаны высокие кредиты):")
    print(true_positives)
    
    return df_test
