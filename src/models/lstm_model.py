"""LSTM deep learning model for time-series demand forecasting."""
import sys
import numpy as np
from typing import Dict, Any, Optional, Tuple
from src.models.base_model import BaseModel
from src.exception import CustomException
from src.logger import logging


class LSTMModel(BaseModel):
    """
    LSTM-based sequential model for demand forecasting.
    Uses TensorFlow/Keras (imported lazily to avoid startup overhead).
    """

    def __init__(
        self,
        sequence_length: int = 30,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LSTM model.

        Args:
            sequence_length: Number of time steps to look back.
            params: Hyperparameters (units, dropout, epochs, batch_size).
        """
        default_params = {
            "units": 64,
            "dropout": 0.2,
            "epochs": 50,
            "batch_size": 32,
        }
        if params:
            default_params.update(params)
        super().__init__("LSTM", default_params)
        self.sequence_length = sequence_length
        self.history = None
        self.n_features = 1

    def build(self) -> None:
        """Build LSTM model architecture."""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers

            self.model = keras.Sequential([
                layers.LSTM(
                    self.params["units"],
                    return_sequences=True,
                    input_shape=(self.sequence_length, self.n_features),
                ),
                layers.Dropout(self.params["dropout"]),
                layers.LSTM(self.params["units"] // 2, return_sequences=False),
                layers.Dropout(self.params["dropout"]),
                layers.Dense(32, activation="relu"),
                layers.Dense(1),
            ])
            self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
            logging.info("Built LSTM model architecture")
        except Exception as e:
            raise CustomException(e, sys)

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create (X, y) sequences for LSTM training.

        Args:
            data: 1D or 2D array of values.

        Returns:
            Tuple of (X sequences, y targets).
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i : i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train LSTM model on sequence data.

        Args:
            X_train: Shaped (samples, sequence_length, n_features).
            y_train: Target values.
        """
        try:
            if len(X_train.shape) == 2:
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            self.n_features = X_train.shape[2]

            if self.model is None:
                self.build()

            from tensorflow.keras.callbacks import EarlyStopping
            early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

            logging.info(f"Training LSTM: epochs={self.params['epochs']}, batch_size={self.params['batch_size']}")
            self.history = self.model.fit(
                X_train,
                y_train,
                epochs=self.params["epochs"],
                batch_size=self.params["batch_size"],
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=0,
            )
            self.is_fitted = True
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        try:
            if len(X.shape) == 2:
                X = X.reshape(X.shape[0], X.shape[1], 1)
            preds = self.model.predict(X, verbose=0)
            return preds.flatten()
        except Exception as e:
            raise CustomException(e, sys)
