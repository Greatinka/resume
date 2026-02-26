from typing import Final, List

DATA_FILE: Final[str] = 'german_credit_data.csv'

RF_MODEL_PATH: Final[str] = 'credit_risk_random_forest_model.joblib'
GB_MODEL_PATH: Final[str] = 'best_credit_risk_model.joblib'
OHE_ENCODER_PATH: Final[str] = 'one_hot_encoder.joblib'
ORDINAL_ENCODER_PATH: Final[str] = 'ordinal_encoder.joblib'
SCALER_PATH: Final[str] = 'scaler.joblib'

CATEGORICAL_COLUMNS: Final[List[str]] = ['Sex', 'Housing', 'Checking account', 'Purpose']
ORDINAL_COLUMN: Final[str] = 'Saving accounts'
TARGET_COLUMN: Final[str] = 'Credit amount'
INDEX_COLUMN: Final[str] = 'Unnamed: 0'

SAVINGS_ORDER: Final[List[List[str]]] = [['unknown', 'little', 'moderate', 'quite rich', 'rich']]

TEST_SIZE: Final[float] = 0.2
RANDOM_STATE: Final[int] = 42

ESTIMATORS: Final[List[int]] = [10, 50, 100, 300, 400, 500, 600]
MIN_SAMPLES_LEAF: Final[List[int]] = [1, 6, 11, 16]
RF_MAX_DEPTH: Final[List[int]] = [1, 2, 3, 4]
GB_MAX_DEPTH: Final[List[int]] = [1, 4, 7]

# Скорость обучения
LEARNING_RATE: Final[float] = 0.1
