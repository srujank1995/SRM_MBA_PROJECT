"""Ensemble model combining multiple forecasting models."""
import sys
import numpy as np
from typing import Dict, List, Optional, Any
from src.models.base_model import BaseModel
from src.exception import CustomException
from src.logger import logging


class EnsembleModel:
    """
    Weighted ensemble of multiple demand forecasting models.
    Supports simple averaging and performance-weighted combination.
    """

    def __init__(
        self,
        models: Dict[str, BaseModel],
        weights: Optional[Dict[str, float]] = None,
        method: str = "weighted_average",
    ):
        """
        Initialize ensemble.

        Args:
            models: Dict mapping model name to trained model instance.
            weights: Dict mapping model name to weight. If None, equal weights.
            method: Combination method ('weighted_average' or 'simple_average').
        """
        self.models = models
        self.method = method
        self.weights = weights or {name: 1.0 / len(models) for name in models}
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def set_weights_from_scores(self, scores: Dict[str, float]) -> None:
        """
        Set weights proportional to model R² scores.

        Args:
            scores: Dict of model_name -> R² score.
        """
        # Only use positive scores; use 0 for non-positive
        adjusted = {k: max(v, 0.0) for k, v in scores.items()}
        total = sum(adjusted.values())
        if total == 0:
            self.weights = {k: 1.0 / len(self.models) for k in self.models}
        else:
            self.weights = {k: adjusted.get(k, 0.0) / total for k in self.models}
        logging.info(f"Ensemble weights set: {self.weights}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate ensemble predictions.

        Args:
            X: Feature matrix.

        Returns:
            Weighted average of model predictions.
        """
        try:
            predictions = np.zeros(len(X))
            for name, model in self.models.items():
                preds = model.predict(X)
                weight = self.weights.get(name, 0.0)
                predictions += weight * preds
            logging.info("Generated ensemble predictions")
            return predictions
        except Exception as e:
            raise CustomException(e, sys)

    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Return predictions from each individual model."""
        try:
            return {name: model.predict(X) for name, model in self.models.items()}
        except Exception as e:
            raise CustomException(e, sys)
