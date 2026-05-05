import os
import sys
import pytest
import numpy as np
import pandas as pd
import scipy.sparse
from unittest.mock import patch, MagicMock
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.exception import CustomException


def _make_csv_files(tmp_path):
    """Write minimal train/test CSVs and return their paths."""
    countries = ["United Kingdom", "Germany", "France", "Australia", "USA"]
    train_df = pd.DataFrame({
        "Country": countries * 4,
        "Quantity": list(range(20)),
    })
    test_df = pd.DataFrame({
        "Country": countries,
        "Quantity": list(range(5)),
    })
    train_path = str(tmp_path / "train.csv")
    test_path = str(tmp_path / "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


class TestDataTransformationConfig:
    def test_preprocessor_path_in_artifacts(self):
        config = DataTransformationConfig()
        assert "artifacts" in config.preprocessor_obj_file_path

    def test_preprocessor_path_ends_with_pkl(self):
        config = DataTransformationConfig()
        assert config.preprocessor_obj_file_path.endswith(".pkl")


class TestDataTransformation:
    def test_get_data_transformer_object_returns_column_transformer(self):
        dt = DataTransformation()
        preprocessor = dt.get_data_transformer_object()
        assert isinstance(preprocessor, ColumnTransformer)

    def test_transformer_has_onehot_encoder(self):
        dt = DataTransformation()
        preprocessor = dt.get_data_transformer_object()
        # transformers is a list of (name, transformer, columns) tuples
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        transformer_map = {name: t for name, t, _ in preprocessor.transformers}
        assert "OneHotEncoder" in transformer_names
        assert isinstance(transformer_map["OneHotEncoder"], OneHotEncoder)

    def test_initiate_data_transformation_returns_four_items(self, tmp_path):
        train_path, test_path = _make_csv_files(tmp_path)
        preprocessor_path = str(tmp_path / "preprocessor.pkl")

        with patch("src.components.data_transformation.DataTransformationConfig") as mock_cfg_cls, \
             patch("src.components.data_transformation.save_object") as mock_save:
            mock_cfg = MagicMock()
            mock_cfg.preprocessor_obj_file_path = preprocessor_path
            mock_cfg_cls.return_value = mock_cfg

            dt = DataTransformation()
            result = dt.initiate_data_transformation(train_path, test_path)

        assert len(result) == 4

    def test_x_train_and_x_test_are_arrays(self, tmp_path):
        train_path, test_path = _make_csv_files(tmp_path)

        with patch("src.components.data_transformation.DataTransformationConfig") as mock_cfg_cls, \
             patch("src.components.data_transformation.save_object"):
            mock_cfg = MagicMock()
            mock_cfg.preprocessor_obj_file_path = str(tmp_path / "preprocessor.pkl")
            mock_cfg_cls.return_value = mock_cfg

            dt = DataTransformation()
            X_train, X_test, y_train, y_test = dt.initiate_data_transformation(
                train_path, test_path
            )

        # ColumnTransformer may return a sparse or dense array
        assert hasattr(X_train, "shape")
        assert hasattr(X_test, "shape")

    def test_y_train_y_test_are_series(self, tmp_path):
        train_path, test_path = _make_csv_files(tmp_path)

        with patch("src.components.data_transformation.DataTransformationConfig") as mock_cfg_cls, \
             patch("src.components.data_transformation.save_object"):
            mock_cfg = MagicMock()
            mock_cfg.preprocessor_obj_file_path = str(tmp_path / "preprocessor.pkl")
            mock_cfg_cls.return_value = mock_cfg

            dt = DataTransformation()
            _, _, y_train, y_test = dt.initiate_data_transformation(train_path, test_path)

        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_save_object_called_with_preprocessor(self, tmp_path):
        train_path, test_path = _make_csv_files(tmp_path)
        preprocessor_path = str(tmp_path / "preprocessor.pkl")

        with patch("src.components.data_transformation.DataTransformationConfig") as mock_cfg_cls, \
             patch("src.components.data_transformation.save_object") as mock_save:
            mock_cfg = MagicMock()
            mock_cfg.preprocessor_obj_file_path = preprocessor_path
            mock_cfg_cls.return_value = mock_cfg

            dt = DataTransformation()
            dt.initiate_data_transformation(train_path, test_path)

        mock_save.assert_called_once()
        call_kwargs = mock_save.call_args
        assert call_kwargs[1]["file_path"] == preprocessor_path or \
               (len(call_kwargs[0]) > 0 and call_kwargs[0][0] == preprocessor_path)

    def test_raises_custom_exception_on_missing_file(self):
        dt = DataTransformation()
        with pytest.raises(CustomException):
            dt.initiate_data_transformation(
                "/nonexistent/train.csv",
                "/nonexistent/test.csv"
            )

    def test_x_train_rows_match_train_csv(self, tmp_path):
        train_path, test_path = _make_csv_files(tmp_path)

        with patch("src.components.data_transformation.DataTransformationConfig") as mock_cfg_cls, \
             patch("src.components.data_transformation.save_object"):
            mock_cfg = MagicMock()
            mock_cfg.preprocessor_obj_file_path = str(tmp_path / "preprocessor.pkl")
            mock_cfg_cls.return_value = mock_cfg

            dt = DataTransformation()
            X_train, X_test, y_train, y_test = dt.initiate_data_transformation(
                train_path, test_path
            )

        train_df = pd.read_csv(train_path)
        assert X_train.shape[0] == len(train_df)
