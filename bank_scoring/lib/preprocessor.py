import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import joblib

from config import (
    CATEGORICAL_COLUMNS, ORDINAL_COLUMN, TARGET_COLUMN,
    SAVINGS_ORDER, OHE_ENCODER_PATH, ORDINAL_ENCODER_PATH, SCALER_PATH
)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Обрабатывает пропущенные значения в колонках счетов."""
    df[ORDINAL_COLUMN] = df[ORDINAL_COLUMN].fillna('unknown')
    df['Checking account'] = df['Checking account'].fillna('unknown')
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Применяет порядковое и one-hot кодирование к категориальным признакам."""
    oe = OrdinalEncoder(categories=SAVINGS_ORDER)
    df[ORDINAL_COLUMN] = oe.fit_transform(df[[ORDINAL_COLUMN]])
    joblib.dump(oe, ORDINAL_ENCODER_PATH)
    
    ohe = OneHotEncoder(handle_unknown='ignore')
    encoded_data = ohe.fit_transform(df[CATEGORICAL_COLUMNS]).toarray()
    encoded_df = pd.DataFrame(
        encoded_data, 
        columns=ohe.get_feature_names_out(CATEGORICAL_COLUMNS)
    )
    joblib.dump(ohe, OHE_ENCODER_PATH)
    
    df = pd.concat([df, encoded_df], axis=1).drop(CATEGORICAL_COLUMNS, axis=1)
    
    return df


def prepare_features(df: pd.DataFrame):
    """Подготавливает признаки и целевую переменную для обучения."""
    df = df.replace([np.inf, -np.inf], np.nan)
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN] > df[TARGET_COLUMN].median()
    
    return X, y


def scale_features(X_train, X_test):
    """Применяет стандартное масштабирование к признакам."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    return X_train_scaled, X_test_scaled


def visualize_data(df: pd.DataFrame):
    """Создает визуализации данных."""
    sns.histplot(df['Age'], kde=True)
    plt.title('Распределение возрастов')
    plt.show()
    
    sns.histplot(df['Credit amount'], kde=True)
    plt.title('Распределение суммы кредита')
    plt.show()
    
    ohe = joblib.load(OHE_ENCODER_PATH)
    df_corr = df.drop(columns=ohe.get_feature_names_out(CATEGORICAL_COLUMNS))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr.corr(), annot=True, cmap='BuPu')
    plt.title('Матрица корреляций признаков')
    plt.show()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Полный пайплайн предобработки данных."""
    df = handle_missing_values(df)
    df = encode_features(df)
    return df
