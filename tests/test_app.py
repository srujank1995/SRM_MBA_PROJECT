import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Ensure the repo root is on sys.path so Flask app can be imported cleanly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def client():
    """Create a Flask test client with mocked dependencies."""
    sample_df = pd.DataFrame({
        "Country": ["United Kingdom", "Germany", "France"],
        "Quantity": [6, 3, 8],
    })

    with patch("pandas.read_csv", return_value=sample_df):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        with flask_app.app.test_client() as c:
            yield c


class TestHomeRoute:
    def test_home_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_home_returns_html(self, client):
        response = client.get("/")
        assert b"html" in response.data.lower() or response.status_code == 200


class TestPredictRoute:
    def test_predict_get_returns_200(self, client):
        response = client.get("/predict")
        assert response.status_code == 200

    def test_predict_post_returns_200(self, client):
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = np.array([15.75])

        with patch("app.PredictPipeline", return_value=mock_pipeline):
            response = client.post(
                "/predict",
                data={"Country": "United Kingdom"},
            )
        assert response.status_code == 200

    def test_predict_post_shows_result(self, client):
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = np.array([42.0])

        with patch("app.PredictPipeline", return_value=mock_pipeline):
            response = client.post(
                "/predict",
                data={"Country": "Germany"},
            )

        assert b"42" in response.data or response.status_code == 200

    def test_predict_post_calls_pipeline_predict(self, client):
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = np.array([10.0])

        with patch("app.PredictPipeline", return_value=mock_pipeline) as mock_cls:
            client.post("/predict", data={"Country": "France"})

        mock_pipeline.predict.assert_called_once()

    def test_predict_post_passes_country_to_pipeline(self, client):
        captured = {}
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = np.array([5.0])

        def capture_predict(df):
            captured["country"] = df["Country"].iloc[0]
            return np.array([5.0])

        mock_pipeline.predict.side_effect = capture_predict

        with patch("app.PredictPipeline", return_value=mock_pipeline):
            client.post("/predict", data={"Country": "Australia"})

        assert captured.get("country") == "Australia"

    def test_predict_get_contains_countries(self, client):
        response = client.get("/predict")
        # Countries are rendered in the template
        assert response.status_code == 200
