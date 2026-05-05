import os
import sys
import pytest
import numpy as np
import scipy.sparse
from unittest.mock import patch, MagicMock, call
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.exception import CustomException


def _make_data(n=100, n_features=5):
    rng = np.random.RandomState(42)
    X = rng.randn(n, n_features)
    y = X[:, 0] * 3 + rng.randn(n) * 0.5
    split = int(n * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


def _mock_catboost_rf():
    """Return mock classes for CatBoostRegressor and RandomForestRegressor that accept any kwargs."""
    catboost_mock = MagicMock()
    catboost_mock.return_value = MagicMock()
    rf_mock = MagicMock()
    rf_mock.return_value = MagicMock()
    return catboost_mock, rf_mock


class TestModelTrainerConfig:
    def test_model_file_path_in_artifacts(self):
        config = ModelTrainerConfig()
        assert "artifacts" in config.trained_model_file_path

    def test_model_file_path_ends_with_pkl(self):
        config = ModelTrainerConfig()
        assert config.trained_model_file_path.endswith(".pkl")


class TestModelTrainer:
    def test_initiate_model_trainer_returns_float(self, tmp_path):
        X_train, X_test, y_train, y_test = _make_data()
        catboost_cls, rf_cls = _mock_catboost_rf()

        with patch("src.components.model_trainer.ModelTrainerConfig") as mock_cfg_cls, \
             patch("src.components.model_trainer.save_object"), \
             patch("src.components.model_trainer.CatBoostRegressor", catboost_cls), \
             patch("src.components.model_trainer.RandomForestRegressor", rf_cls), \
             patch("src.components.model_trainer.evaluate_models") as mock_eval:
            mock_cfg = MagicMock()
            mock_cfg.trained_model_file_path = str(tmp_path / "model.pkl")
            mock_cfg_cls.return_value = mock_cfg
            mock_eval.return_value = {
                "RandomForest": 0.75,
                "LinearRegression": 0.88,
                "CatBoost": 0.70,
            }

            trainer = ModelTrainer()
            score = trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)

        assert isinstance(score, float)

    def test_best_model_selected_by_highest_score(self, tmp_path):
        X_train, X_test, y_train, y_test = _make_data()
        catboost_cls, rf_cls = _mock_catboost_rf()

        with patch("src.components.model_trainer.ModelTrainerConfig") as mock_cfg_cls, \
             patch("src.components.model_trainer.save_object") as mock_save, \
             patch("src.components.model_trainer.CatBoostRegressor", catboost_cls), \
             patch("src.components.model_trainer.RandomForestRegressor", rf_cls), \
             patch("src.components.model_trainer.evaluate_models") as mock_eval:
            mock_cfg = MagicMock()
            mock_cfg.trained_model_file_path = str(tmp_path / "model.pkl")
            mock_cfg_cls.return_value = mock_cfg
            mock_eval.return_value = {
                "RandomForest": 0.95,
                "LinearRegression": 0.10,
                "CatBoost": 0.50,
            }

            trainer = ModelTrainer()
            score = trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)

        assert score == pytest.approx(0.95)

    def test_save_object_called_once(self, tmp_path):
        X_train, X_test, y_train, y_test = _make_data()
        catboost_cls, rf_cls = _mock_catboost_rf()

        with patch("src.components.model_trainer.ModelTrainerConfig") as mock_cfg_cls, \
             patch("src.components.model_trainer.save_object") as mock_save, \
             patch("src.components.model_trainer.CatBoostRegressor", catboost_cls), \
             patch("src.components.model_trainer.RandomForestRegressor", rf_cls), \
             patch("src.components.model_trainer.evaluate_models") as mock_eval:
            mock_cfg = MagicMock()
            mock_cfg.trained_model_file_path = str(tmp_path / "model.pkl")
            mock_cfg_cls.return_value = mock_cfg
            mock_eval.return_value = {
                "RandomForest": 0.9,
                "LinearRegression": 0.5,
                "CatBoost": 0.3,
            }

            trainer = ModelTrainer()
            trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)

        mock_save.assert_called_once()

    def test_raises_custom_exception_on_evaluate_error(self, tmp_path):
        X_train, X_test, y_train, y_test = _make_data()
        catboost_cls, rf_cls = _mock_catboost_rf()

        with patch("src.components.model_trainer.ModelTrainerConfig") as mock_cfg_cls, \
             patch("src.components.model_trainer.save_object"), \
             patch("src.components.model_trainer.CatBoostRegressor", catboost_cls), \
             patch("src.components.model_trainer.RandomForestRegressor", rf_cls), \
             patch("src.components.model_trainer.evaluate_models",
                   side_effect=ValueError("eval failed")):
            mock_cfg = MagicMock()
            mock_cfg.trained_model_file_path = str(tmp_path / "model.pkl")
            mock_cfg_cls.return_value = mock_cfg

            trainer = ModelTrainer()
            with pytest.raises(CustomException):
                trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)

    def test_score_is_best_test_score_from_report(self, tmp_path):
        X_train, X_test, y_train, y_test = _make_data()
        catboost_cls, rf_cls = _mock_catboost_rf()

        with patch("src.components.model_trainer.ModelTrainerConfig") as mock_cfg_cls, \
             patch("src.components.model_trainer.save_object"), \
             patch("src.components.model_trainer.CatBoostRegressor", catboost_cls), \
             patch("src.components.model_trainer.RandomForestRegressor", rf_cls), \
             patch("src.components.model_trainer.evaluate_models") as mock_eval:
            mock_cfg = MagicMock()
            mock_cfg.trained_model_file_path = str(tmp_path / "model.pkl")
            mock_cfg_cls.return_value = mock_cfg
            mock_eval.return_value = {
                "RandomForest": 0.70,
                "LinearRegression": 0.85,
                "CatBoost": 0.60,
            }

            trainer = ModelTrainer()
            score = trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)

        assert score == pytest.approx(0.85)

