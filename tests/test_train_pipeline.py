import pytest
from unittest.mock import patch, MagicMock, call
from src.pipeline.train_pipeline import TrainPipeline


class TestTrainPipeline:
    def _mock_components(self):
        mock_ingestion = MagicMock()
        mock_ingestion.initiate_data_ingestion.return_value = (
            "artifacts/train.csv",
            "artifacts/test.csv",
        )

        mock_transformation = MagicMock()
        mock_transformation.initiate_data_transformation.return_value = (
            "X_train", "X_test", "y_train", "y_test"
        )

        mock_trainer = MagicMock()
        mock_trainer.initiate_model_trainer.return_value = 0.85

        return mock_ingestion, mock_transformation, mock_trainer

    def test_run_pipeline_calls_data_ingestion(self):
        mock_ingestion, mock_transformation, mock_trainer = self._mock_components()

        with patch("src.pipeline.train_pipeline.DataIngestion",
                   return_value=mock_ingestion), \
             patch("src.pipeline.train_pipeline.DataTransformation",
                   return_value=mock_transformation), \
             patch("src.pipeline.train_pipeline.ModelTrainer",
                   return_value=mock_trainer):
            pipeline = TrainPipeline()
            pipeline.run_pipeline()

        mock_ingestion.initiate_data_ingestion.assert_called_once()

    def test_run_pipeline_calls_data_transformation(self):
        mock_ingestion, mock_transformation, mock_trainer = self._mock_components()

        with patch("src.pipeline.train_pipeline.DataIngestion",
                   return_value=mock_ingestion), \
             patch("src.pipeline.train_pipeline.DataTransformation",
                   return_value=mock_transformation), \
             patch("src.pipeline.train_pipeline.ModelTrainer",
                   return_value=mock_trainer):
            pipeline = TrainPipeline()
            pipeline.run_pipeline()

        mock_transformation.initiate_data_transformation.assert_called_once_with(
            "artifacts/train.csv", "artifacts/test.csv"
        )

    def test_run_pipeline_calls_model_trainer(self):
        mock_ingestion, mock_transformation, mock_trainer = self._mock_components()

        with patch("src.pipeline.train_pipeline.DataIngestion",
                   return_value=mock_ingestion), \
             patch("src.pipeline.train_pipeline.DataTransformation",
                   return_value=mock_transformation), \
             patch("src.pipeline.train_pipeline.ModelTrainer",
                   return_value=mock_trainer):
            pipeline = TrainPipeline()
            pipeline.run_pipeline()

        mock_trainer.initiate_model_trainer.assert_called_once_with(
            "X_train", "X_test", "y_train", "y_test"
        )

    def test_run_pipeline_executes_in_correct_order(self):
        call_order = []
        mock_ingestion, mock_transformation, mock_trainer = self._mock_components()

        mock_ingestion.initiate_data_ingestion.side_effect = \
            lambda: call_order.append("ingestion") or ("train.csv", "test.csv")
        mock_transformation.initiate_data_transformation.side_effect = \
            lambda *a: call_order.append("transformation") or ("X_tr", "X_te", "y_tr", "y_te")
        mock_trainer.initiate_model_trainer.side_effect = \
            lambda *a: call_order.append("trainer") or 0.9

        with patch("src.pipeline.train_pipeline.DataIngestion",
                   return_value=mock_ingestion), \
             patch("src.pipeline.train_pipeline.DataTransformation",
                   return_value=mock_transformation), \
             patch("src.pipeline.train_pipeline.ModelTrainer",
                   return_value=mock_trainer):
            TrainPipeline().run_pipeline()

        assert call_order == ["ingestion", "transformation", "trainer"]

    def test_run_pipeline_passes_ingestion_paths_to_transformation(self):
        mock_ingestion, mock_transformation, mock_trainer = self._mock_components()
        mock_ingestion.initiate_data_ingestion.return_value = (
            "custom/train.csv", "custom/test.csv"
        )

        with patch("src.pipeline.train_pipeline.DataIngestion",
                   return_value=mock_ingestion), \
             patch("src.pipeline.train_pipeline.DataTransformation",
                   return_value=mock_transformation), \
             patch("src.pipeline.train_pipeline.ModelTrainer",
                   return_value=mock_trainer):
            TrainPipeline().run_pipeline()

        mock_transformation.initiate_data_transformation.assert_called_once_with(
            "custom/train.csv", "custom/test.csv"
        )
