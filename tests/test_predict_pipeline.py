import pytest
import numpy as np
import pandas as pd
import scipy.sparse
from unittest.mock import patch, MagicMock
from src.pipeline.predict_pipeline import PredictPipeline


class TestPredictPipeline:
    def _make_pipeline(self, countries=None):
        """Return a PredictPipeline with mocked model and preprocessor."""
        if countries is None:
            countries = ["United Kingdom"]

        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = np.array([[1, 0, 0]])

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([42.0])

        return mock_preprocessor, mock_model

    def test_predict_returns_array(self):
        mock_preprocessor, mock_model = self._make_pipeline()

        with patch("src.pipeline.predict_pipeline.load_object") as mock_load:
            mock_load.side_effect = [mock_model, mock_preprocessor]
            pipeline = PredictPipeline()
            features = pd.DataFrame({"Country": ["United Kingdom"]})
            result = pipeline.predict(features)

        assert result is not None

    def test_predict_calls_preprocessor_transform(self):
        mock_preprocessor, mock_model = self._make_pipeline()

        with patch("src.pipeline.predict_pipeline.load_object") as mock_load:
            mock_load.side_effect = [mock_model, mock_preprocessor]
            pipeline = PredictPipeline()
            features = pd.DataFrame({"Country": ["Germany"]})
            pipeline.predict(features)

        mock_preprocessor.transform.assert_called_once()

    def test_predict_calls_model_predict(self):
        mock_preprocessor, mock_model = self._make_pipeline()

        with patch("src.pipeline.predict_pipeline.load_object") as mock_load:
            mock_load.side_effect = [mock_model, mock_preprocessor]
            pipeline = PredictPipeline()
            features = pd.DataFrame({"Country": ["France"]})
            pipeline.predict(features)

        mock_model.predict.assert_called_once()

    def test_predict_returns_correct_value(self):
        mock_preprocessor, mock_model = self._make_pipeline()
        mock_model.predict.return_value = np.array([99.5])

        with patch("src.pipeline.predict_pipeline.load_object") as mock_load:
            mock_load.side_effect = [mock_model, mock_preprocessor]
            pipeline = PredictPipeline()
            features = pd.DataFrame({"Country": ["Australia"]})
            result = pipeline.predict(features)

        assert result[0] == pytest.approx(99.5)

    def test_predict_loads_model_and_preprocessor(self):
        mock_preprocessor, mock_model = self._make_pipeline()

        with patch("src.pipeline.predict_pipeline.load_object") as mock_load:
            mock_load.side_effect = [mock_model, mock_preprocessor]
            pipeline = PredictPipeline()
            features = pd.DataFrame({"Country": ["United Kingdom"]})
            pipeline.predict(features)

        assert mock_load.call_count == 2

    def test_predict_passes_transformed_features_to_model(self):
        transformed = np.array([[0, 1, 0, 0, 0]])
        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = transformed
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([7.0])

        with patch("src.pipeline.predict_pipeline.load_object") as mock_load:
            mock_load.side_effect = [mock_model, mock_preprocessor]
            pipeline = PredictPipeline()
            features = pd.DataFrame({"Country": ["Germany"]})
            pipeline.predict(features)

        np.testing.assert_array_equal(
            mock_model.predict.call_args[0][0], transformed
        )
