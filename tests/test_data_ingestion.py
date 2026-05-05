import os
import sys
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.exception import CustomException


class TestDataIngestionConfig:
    def test_train_data_path_contains_artifacts(self):
        config = DataIngestionConfig()
        assert "artifacts" in config.train_data_path

    def test_test_data_path_contains_artifacts(self):
        config = DataIngestionConfig()
        assert "artifacts" in config.test_data_path

    def test_raw_data_path_contains_artifacts(self):
        config = DataIngestionConfig()
        assert "artifacts" in config.raw_data_path

    def test_train_and_test_paths_are_different(self):
        config = DataIngestionConfig()
        assert config.train_data_path != config.test_data_path

    def test_filenames_are_csv(self):
        config = DataIngestionConfig()
        assert config.train_data_path.endswith(".csv")
        assert config.test_data_path.endswith(".csv")
        assert config.raw_data_path.endswith(".csv")


class TestDataIngestion:
    def _make_sample_df(self):
        return pd.DataFrame({
            "Country": ["United Kingdom", "Germany", "France"] * 10,
            "Quantity": [6, 3, 8] * 10,
            "InvoiceNo": range(30),
        })

    def test_initiate_data_ingestion_returns_two_paths(self, tmp_path):
        sample_df = self._make_sample_df()
        with patch("src.components.data_ingestion.pd.read_csv", return_value=sample_df), \
             patch("src.components.data_ingestion.DataIngestionConfig") as mock_cfg_cls:
            mock_cfg = MagicMock()
            mock_cfg.train_data_path = str(tmp_path / "train.csv")
            mock_cfg.test_data_path = str(tmp_path / "test.csv")
            mock_cfg.raw_data_path = str(tmp_path / "data.csv")
            mock_cfg_cls.return_value = mock_cfg

            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

        assert train_path == str(tmp_path / "train.csv")
        assert test_path == str(tmp_path / "test.csv")

    def test_initiate_data_ingestion_creates_output_files(self, tmp_path):
        sample_df = self._make_sample_df()
        with patch("src.components.data_ingestion.pd.read_csv", return_value=sample_df), \
             patch("src.components.data_ingestion.DataIngestionConfig") as mock_cfg_cls:
            mock_cfg = MagicMock()
            mock_cfg.train_data_path = str(tmp_path / "train.csv")
            mock_cfg.test_data_path = str(tmp_path / "test.csv")
            mock_cfg.raw_data_path = str(tmp_path / "data.csv")
            mock_cfg_cls.return_value = mock_cfg

            ingestion = DataIngestion()
            ingestion.initiate_data_ingestion()

        assert os.path.exists(str(tmp_path / "train.csv"))
        assert os.path.exists(str(tmp_path / "test.csv"))
        assert os.path.exists(str(tmp_path / "data.csv"))

    def test_train_set_larger_than_test_set(self, tmp_path):
        sample_df = self._make_sample_df()
        with patch("src.components.data_ingestion.pd.read_csv", return_value=sample_df), \
             patch("src.components.data_ingestion.DataIngestionConfig") as mock_cfg_cls:
            mock_cfg = MagicMock()
            mock_cfg.train_data_path = str(tmp_path / "train.csv")
            mock_cfg.test_data_path = str(tmp_path / "test.csv")
            mock_cfg.raw_data_path = str(tmp_path / "data.csv")
            mock_cfg_cls.return_value = mock_cfg

            ingestion = DataIngestion()
            ingestion.initiate_data_ingestion()

        train_df = pd.read_csv(str(tmp_path / "train.csv"))
        test_df = pd.read_csv(str(tmp_path / "test.csv"))
        assert len(train_df) > len(test_df)

    def test_raises_custom_exception_on_missing_csv(self):
        with patch("src.components.data_ingestion.pd.read_csv",
                   side_effect=FileNotFoundError("not found")):
            ingestion = DataIngestion()
            with pytest.raises(CustomException):
                ingestion.initiate_data_ingestion()

    def test_total_rows_preserved_across_splits(self, tmp_path):
        sample_df = self._make_sample_df()
        with patch("src.components.data_ingestion.pd.read_csv", return_value=sample_df), \
             patch("src.components.data_ingestion.DataIngestionConfig") as mock_cfg_cls:
            mock_cfg = MagicMock()
            mock_cfg.train_data_path = str(tmp_path / "train.csv")
            mock_cfg.test_data_path = str(tmp_path / "test.csv")
            mock_cfg.raw_data_path = str(tmp_path / "data.csv")
            mock_cfg_cls.return_value = mock_cfg

            ingestion = DataIngestion()
            ingestion.initiate_data_ingestion()

        train_df = pd.read_csv(str(tmp_path / "train.csv"))
        test_df = pd.read_csv(str(tmp_path / "test.csv"))
        assert len(train_df) + len(test_df) == len(sample_df)
