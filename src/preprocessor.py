# src/preprocessor.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from config import Config

def preprocess_data(df, target_column):
    """
    Handles data preprocessing with a strict 3-way split (Train, Validation, Test)
    using parameters defined in src/config.py (Config class).
    """
    try:
        data = df.copy()

        # 1. Separate features (X) and target (y)
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        X = data.drop(columns=[target_column])
        y = data[target_column]

        target_encoder = None
        if not pd.api.types.is_numeric_dtype(y) or y.nunique() < 20: # أو إذا كان تصنيفياً
            if not pd.api.types.is_numeric_dtype(y):
                target_encoder = LabelEncoder()
                y = pd.Series(target_encoder.fit_transform(y), index=y.index)

        # 2. Strict 3-Way Split using Config class values
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
        )

        relative_val_size = Config.VAL_SIZE / (1.0 - Config.TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=relative_val_size, random_state=Config.RANDOM_STATE
        )

        num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns

        # 3. Imputation using config strategies
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy=Config.NUMERIC_IMPUTE_STRATEGY)
            X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
            X_val[num_cols] = num_imputer.transform(X_val[num_cols])
            X_test[num_cols] = num_imputer.transform(X_test[num_cols])

        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy=Config.CATEGORICAL_IMPUTE_STRATEGY)
            X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
            X_val[cat_cols] = cat_imputer.transform(X_val[cat_cols])
            X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

            # 4. Encoding
            for col in cat_cols:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))

                train_classes = set(le.classes_)
                X_val[col] = X_val[col].astype(str).map(
                    lambda s: le.transform([s])[0] if s in train_classes else Config.UNSEEN_CATEGORY_VALUE
                )
                X_test[col] = X_test[col].astype(str).map(
                    lambda s: le.transform([s])[0] if s in train_classes else Config.UNSEEN_CATEGORY_VALUE
                )


        # 5. Feature Scaling
        if len(num_cols) > 0:
            scaler = StandardScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_val[num_cols] = scaler.transform(X_val[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])

        return X_train, X_val, X_test, y_train, y_val, y_test

    except Exception as e:
        raise Exception(f"An error occurred during preprocessing: {e}")
