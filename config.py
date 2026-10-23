# config.py
import os
from pathlib import Path

class Config:
    # --- 1. Reproducibility & Data Splitting ---
    RANDOM_STATE = 42
    TEST_SIZE = 0.15        # Final test set ratio
    VAL_SIZE = 0.15         # Internal validation set ratio
    CV_FOLDS = 5            # Number of cross-validation folds
    N_JOBS = -1             # Use all CPU cores to speed up training

    # --- 2. Preprocessing Strategies ---
    NUMERIC_IMPUTE_STRATEGY = 'median'
    CATEGORICAL_IMPUTE_STRATEGY = 'most_frequent'
    UNSEEN_CATEGORY_VALUE = -1  # To handle new categorical values in Val/Test

    # --- 3. Model Parameters ---
    LOGISTIC_REGRESSION_MAX_ITER = 1000

    # --- 3. Directories & Paths (Automatically create directories) ---
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT_DIR / "data"
    MODEL_DIR = ROOT_DIR / "models"
    EXPERIMENT_DIR = ROOT_DIR / "experiments"

    # Ensure directories exist if not already present
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    # --- 4. MLflow Settings ---
    MLFLOW_EXPERIMENT_NAME = "AutoML_Pipeline_Experiment"
    # A local path for storing mlruns files can be specified
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

    # --- 5. Hyperparameter Search Spaces (Classification) ---
    CLASSIFICATION_MODELS = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "max_iter": [1000]
        },
        "Random Forest Classifier": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        },
        "Support Vector Machine": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"]
        }
    }

    # --- 6. Hyperparameter Search Spaces (Regression) ---
    REGRESSION_MODELS = {
        "Linear Regression": {},
        "Random Forest Regressor": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        },
        "Support Vector Regressor": {
            "C": [0.1, 1.0, 10.0],
            "epsilon": [0.01, 0.1]
        }
    }
