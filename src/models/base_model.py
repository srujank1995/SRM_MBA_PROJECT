"""Base model class for demand forecasting models."""
import sys
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from src.exception import CustomException
from src.logger import logging


class BaseModel(ABC):
    """Abstract base class for all forecasting models."""

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize base model.

        Args:
            name: Model name identifier.
            params: Model hyperparameters.
        """
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def build(self) -> None:
        """Build/initialize the underlying model."""
        pass

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
