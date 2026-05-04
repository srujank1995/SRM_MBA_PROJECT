"""Prediction pipeline for demand forecasting."""
import sys
import os
import pandas as pd
import numpy as np
from typing import Union, Optional, List
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    """
    Prediction pipeline supporting both legacy artifacts and new models.
    """

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the legacy model and preprocessor.

        Args:
            features: Input DataFrame with feature columns.

        Returns:
            Array of predictions.
        """
        try:
            model = load_object("artifacts/model.pkl")
            preprocessor = load_object("artifacts/preprocessor.pkl")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

    def predict_advanced(
        self,
        features: pd.DataFrame,
        model_name: str = "best",
    ) -> np.ndarray:
        """
        Generate predictions using the advanced models.

        Args:
            features: Feature DataFrame (must match training columns).
            model_name: Model to use ('best', 'randomforest', 'xgboost', etc.)

        Returns:
            Array of predictions.
        """
        try:
            feature_columns: List[str] = load_object("models/feature_columns.pkl")
            scaler = load_object("models/scaler.pkl")

            # Align features
            for col in feature_columns:
                if col not in features.columns:
                    features[col] = 0
            X = features[feature_columns]
            X_scaled = scaler.transform(X)

            if model_name == "best":
                # Try XGBoost as default best
                model_path = "models/xgboost_model.pkl"
                if not os.path.exists(model_path):
                    model_path = "models/randomforest_model.pkl"
            else:
                model_path = f"models/{model_name.lower()}_model.pkl"

            model = load_object(model_path)
            return model.predict(X_scaled)

        except Exception as e:
            raise CustomException(e, sys)