import os
import sys
import pytest
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from src.utils import save_object, load_object, evaluate_models
from src.exception import CustomException


class TestSaveObject:
    def test_saves_file_to_disk(self, tmp_path):
        file_path = str(tmp_path / "model.pkl")
        obj = {"key": "value"}
        save_object(file_path, obj)
        assert os.path.exists(file_path)

    def test_creates_intermediate_directories(self, tmp_path):
        file_path = str(tmp_path / "nested" / "deep" / "model.pkl")
        save_object(file_path, [1, 2, 3])
        assert os.path.exists(file_path)

    def test_saved_object_is_loadable(self, tmp_path):
        file_path = str(tmp_path / "obj.pkl")
        original = {"a": 1, "b": [2, 3]}
        save_object(file_path, original)
        loaded = load_object(file_path)
        assert loaded == original

    def test_raises_custom_exception_on_invalid_path(self):
        with pytest.raises(CustomException):
            save_object("/nonexistent_root_dir_that_cannot_be_created/x/y/z.pkl", {})

    def test_saves_sklearn_model(self, tmp_path):
        file_path = str(tmp_path / "lr.pkl")
        model = LinearRegression()
        save_object(file_path, model)
        assert os.path.exists(file_path)


class TestLoadObject:
    def test_loads_dict(self, tmp_path):
        file_path = str(tmp_path / "data.pkl")
        data = {"hello": "world"}
        save_object(file_path, data)
        result = load_object(file_path)
        assert result == data

    def test_loads_list(self, tmp_path):
        file_path = str(tmp_path / "lst.pkl")
        save_object(file_path, [10, 20, 30])
        result = load_object(file_path)
        assert result == [10, 20, 30]

    def test_loads_sklearn_model(self, tmp_path):
        file_path = str(tmp_path / "model.pkl")
        model = LinearRegression()
        save_object(file_path, model)
        loaded = load_object(file_path)
        assert isinstance(loaded, LinearRegression)

    def test_raises_custom_exception_for_missing_file(self, tmp_path):
        with pytest.raises(CustomException):
            load_object(str(tmp_path / "does_not_exist.pkl"))

    def test_roundtrip_numpy_array(self, tmp_path):
        file_path = str(tmp_path / "arr.pkl")
        arr = np.array([[1, 2], [3, 4]])
        save_object(file_path, arr)
        loaded = load_object(file_path)
        np.testing.assert_array_equal(loaded, arr)


class TestEvaluateModels:
    def _make_data(self, n=100, n_features=3):
        rng = np.random.RandomState(0)
        X = rng.randn(n, n_features)
        y = X[:, 0] * 2 + rng.randn(n) * 0.1
        split = int(n * 0.8)
        return X[:split], X[split:], y[:split], y[split:]

    def test_returns_dict_with_model_names(self):
        X_train, X_test, y_train, y_test = self._make_data()
        models = {
            "LinearRegression": LinearRegression(),
            "Dummy": DummyRegressor(),
        }
        report = evaluate_models(X_train, y_train, X_test, y_test, models)
        assert set(report.keys()) == {"LinearRegression", "Dummy"}

    def test_scores_are_floats(self):
        X_train, X_test, y_train, y_test = self._make_data()
        models = {"LR": LinearRegression()}
        report = evaluate_models(X_train, y_train, X_test, y_test, models)
        assert isinstance(report["LR"], float)

    def test_linear_regression_scores_high_on_linear_data(self):
        X_train, X_test, y_train, y_test = self._make_data()
        models = {"LR": LinearRegression()}
        report = evaluate_models(X_train, y_train, X_test, y_test, models)
        assert report["LR"] > 0.9

    def test_multiple_models_evaluated(self):
        X_train, X_test, y_train, y_test = self._make_data()
        models = {
            "LR": LinearRegression(),
            "Dummy": DummyRegressor(strategy="mean"),
        }
        report = evaluate_models(X_train, y_train, X_test, y_test, models)
        assert len(report) == 2

    def test_empty_models_dict_returns_empty_report(self):
        X_train, X_test, y_train, y_test = self._make_data()
        report = evaluate_models(X_train, y_train, X_test, y_test, {})
        assert report == {}
