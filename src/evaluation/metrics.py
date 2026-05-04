"""Evaluation metrics for demand forecasting models."""
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.exception import CustomException
from src.logger import logging


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        MAPE as a percentage.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> Dict[str, float]:
    """
    Compute MAE, RMSE, R², and MAPE.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        model_name: Optional model name for logging.

    Returns:
        Dict of metric name -> value.
    """
    try:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        metrics = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "MAPE": round(mape, 4),
        }

        if model_name:
            logging.info(f"Metrics for {model_name}: {metrics}")

        return metrics
    except Exception as e:
        raise CustomException(e, sys)


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a comparison DataFrame of model metrics.

    Args:
        results: Dict of model_name -> metrics_dict.

    Returns:
        Sorted comparison DataFrame.
    """
    try:
        df = pd.DataFrame(results).T
        df = df.sort_values("RMSE", ascending=True)
        return df
    except Exception as e:
        raise CustomException(e, sys)
