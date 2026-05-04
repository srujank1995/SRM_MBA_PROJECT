"""Traditional ML models for demand forecasting."""
import sys
import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from src.models.base_model import BaseModel
from src.exception import CustomException
from src.logger import logging


class LinearRegressionModel(BaseModel):
    """Linear Regression forecasting model."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__("LinearRegression", params)
        self.build()

    def build(self) -> None:
        self.model = LinearRegression(**self.params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            logging.info(f"Training {self.name}")
            self.model.fit(X_train, y_train)
            self.is_fitted = True
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, X: np.ndarray) -> np.ndarray:
        try:
            return self.model.predict(X)
        except Exception as e:
            raise CustomException(e, sys)


class RandomForestModel(BaseModel):
    """Random Forest forecasting model."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {"n_estimators": 100, "max_depth": 10, "random_state": 42, "n_jobs": -1}
        if params:
            default_params.update(params)
        super().__init__("RandomForest", default_params)
        self.build()

    def build(self) -> None:
        self.model = RandomForestRegressor(**self.params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            logging.info(f"Training {self.name}")
            self.model.fit(X_train, y_train)
            self.is_fitted = True
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, X: np.ndarray) -> np.ndarray:
        try:
            return self.model.predict(X)
        except Exception as e:
            raise CustomException(e, sys)

    def feature_importances(self) -> np.ndarray:
        """Return feature importance scores."""
        if self.is_fitted:
            return self.model.feature_importances_
        return np.array([])


class XGBoostModel(BaseModel):
    """XGBoost forecasting model."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "random_state": 42,
            "n_jobs": -1,
        }
        if params:
            default_params.update(params)
        super().__init__("XGBoost", default_params)
        self.build()

    def build(self) -> None:
        self.model = xgb.XGBRegressor(**self.params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            logging.info(f"Training {self.name}")
            self.model.fit(X_train, y_train, verbose=False)
            self.is_fitted = True
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, X: np.ndarray) -> np.ndarray:
        try:
            return self.model.predict(X)
        except Exception as e:
            raise CustomException(e, sys)


class LightGBMModel(BaseModel):
    """LightGBM forecasting model."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        if params:
            default_params.update(params)
        super().__init__("LightGBM", default_params)
        self.build()

    def build(self) -> None:
        self.model = lgb.LGBMRegressor(**self.params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            logging.info(f"Training {self.name}")
            self.model.fit(X_train, y_train)
            self.is_fitted = True
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, X: np.ndarray) -> np.ndarray:
        try:
            return self.model.predict(X)
        except Exception as e:
            raise CustomException(e, sys)


class CatBoostModel(BaseModel):
    """CatBoost forecasting model."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "iterations": 100,
            "learning_rate": 0.1,
            "depth": 6,
            "verbose": 0,
            "random_seed": 42,
        }
        if params:
            default_params.update(params)
        super().__init__("CatBoost", default_params)
        self.build()

    def build(self) -> None:
        self.model = CatBoostRegressor(**self.params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            logging.info(f"Training {self.name}")
            self.model.fit(X_train, y_train, verbose=False)
            self.is_fitted = True
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, X: np.ndarray) -> np.ndarray:
        try:
            return self.model.predict(X)
        except Exception as e:
            raise CustomException(e, sys)
