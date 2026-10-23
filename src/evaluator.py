# src/evaluator.py
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)

def evaluate_predictions(y_true, y_pred, problem_type):
    """
    Computes performance metrics based on the problem type (Classification vs Regression).
    """
    try:
        if problem_type == "Classification":
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            metrics = {
                "Accuracy (%)": round(acc * 100, 2),
                "Precision (%)": round(precision * 100, 2),
                "Recall (%)": round(recall * 100, 2),
                "F1-Score (%)": round(f1 * 100, 2)
            }
            sort_metric = "F1-Score (%)"
            ascending_flag = False

        else:  # Regression
            mse = mean_squared_error(y_true, y_pred)
            #rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)

            metrics = {
                "MSE": round(mse, 4),
                "R2-Score (%)": round(r2 * 100, 2)
            }
            sort_metric = "R2-Score (%)"
            ascending_flag = False

        return metrics, sort_metric, ascending_flag

    except Exception as e:
        raise Exception(f"Error during evaluation: {e}")
