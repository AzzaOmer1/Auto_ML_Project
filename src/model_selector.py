# src/model_selector.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from config import Config
from src.evaluator import evaluate_predictions   # Import evaluation module

import mlflow
from config import Config

# Configure MLflow tracking URI and experiment from configuration
mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)

# Safely attempt to import advanced models
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def get_models(problem_type):
    """
    Returns the selected production-grade models based on the problem type.
    """
    if problem_type == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=Config.LOGISTIC_REGRESSION_MAX_ITER, random_state=Config.RANDOM_STATE),
            "Random Forest Classifier": RandomForestClassifier(random_state=Config.RANDOM_STATE)
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(random_state=Config.RANDOM_STATE, eval_metric='logloss')
        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = LGBMClassifier(random_state=Config.RANDOM_STATE, verbose=-1)

        return models

    elif problem_type == "Regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=Config.RANDOM_STATE),
            "Random Forest Regressor": RandomForestRegressor(random_state=Config.RANDOM_STATE)
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost Regressor"] = XGBRegressor(random_state=Config.RANDOM_STATE)
        if LIGHTGBM_AVAILABLE:
            models["LightGBM Regressor"] = LGBMRegressor(random_state=Config.RANDOM_STATE, verbose=-1)

        return models

    return {}


def train_and_evaluate_models(X_train, X_val, y_train, y_val, selected_model_names, problem_type):
    """
    Trains selected models and delegates performance evaluation to src/evaluator.py,
    while tracking everything with MLflow.
    """
    models_dict = get_models(problem_type)
    results = []
    trained_models = {}

    # Configure MLflow tracking uri and experiment from Config
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)

    sort_metric = ""
    ascending_flag = False

    for name in selected_model_names:
        if name in models_dict:
            model = models_dict[name]

            with mlflow.start_run(run_name=name):
                try:
                    # 1. Train the model
                    model.fit(X_train, y_train)

                    # 2. Predict on Validation Set
                    y_pred = model.predict(X_val)

                    # 3. Evaluate using the dedicated Evaluator module
                    metrics, sort_metric, ascending_flag = evaluate_predictions(y_val, y_pred, problem_type)

                    # Log metrics to MLflow
                    for metric_name, metric_val in metrics.items():
                        mlflow.log_metric(metric_name.lower().replace(" ", "_").replace("(%)", "pct"), metric_val)

                    # Append results for DataFrame summary
                    res_row = {"Model": name}
                    res_row.update(metrics)
                    results.append(res_row)

                    # Log the trained model artifact to MLflow
                    mlflow.sklearn.log_model(model, artifact_path=f"models/{name.replace(' ', '_')}")

                    trained_models[name] = model

                except Exception as e:
                    print(f"Error training {name}: {e}")

    results_df = pd.DataFrame(results)
    if not results_df.empty and sort_metric in results_df.columns:
        results_df = results_df.sort_values(by=sort_metric, ascending=ascending_flag).reset_index(drop=True)

    return results_df, trained_models


def tune_and_evaluate_best_model(X_train, X_val, X_test, y_train, y_val, y_test, best_model_name, problem_type):
    """
    Safely performs Hyperparameter Tuning for the selected model, then evaluates it finally on the Test Set.
    """
    models_dict = get_models(problem_type)
    if best_model_name not in models_dict:
        return None, {}

    base_model = models_dict[best_model_name]

    # Safely retrieve search spaces from Config
    search_spaces = {}
    if problem_type == "Regression" and hasattr(Config, "REGRESSION_MODELS"):
        search_spaces = Config.REGRESSION_MODELS.get(best_model_name, {})
    elif problem_type == "Classification" and hasattr(Config, "CLASSIFICATION_MODELS"):
        search_spaces = Config.CLASSIFICATION_MODELS.get(best_model_name, {})

    # If search spaces are available and the model supports tuning, perform RandomizedSearchCV, otherwise train directly
    try:
        if search_spaces and len(search_spaces) > 0:
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=search_spaces,
                n_iter=3,
                cv=3,
                scoring='r2' if problem_type == 'Regression' else 'f1_weighted',
                random_state=Config.RANDOM_STATE,
                n_jobs=Config.N_JOBS
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
        else:
            best_model = base_model
            best_model.fit(X_train, y_train)
    except Exception as e:
        # Safety fallback: If tuning fails, train the model directly without freezing the system
        base_model.fit(X_train, y_train)
        best_model = base_model

    # Final evaluation on the true Test Set
    y_test_pred = best_model.predict(X_test)
    test_metrics, _, _ = evaluate_predictions(y_test, y_test_pred, problem_type)

    return best_model, test_metrics
