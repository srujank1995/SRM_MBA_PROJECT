"""Training pipeline for the demand forecasting system."""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


class TrainPipeline:
    """
    End-to-end training pipeline.
    Orchestrates data ingestion, preprocessing, feature engineering,
    model training, and evaluation.
    """

    def __init__(self, data_path: str = "notebook/data/data.csv"):
        """
        Initialize TrainPipeline.

        Args:
            data_path: Path to the raw data CSV file.
        """
        self.data_path = data_path

    def run_pipeline(self) -> None:
        """Run the legacy training pipeline (backward-compatible)."""
        try:
            from src.components.data_ingestion import DataIngestion
            from src.components.data_transformation import DataTransformation
            from src.components.model_trainer import ModelTrainer

            logging.info("Starting training pipeline")

            data_ingestion = DataIngestion()
            train_data, test_data = data_ingestion.initiate_data_ingestion()

            data_transformation = DataTransformation()
            X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(
                train_data, test_data
            )

            model_trainer = ModelTrainer()
            score = model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)

            logging.info(f"Training pipeline complete. Best R² score: {score:.4f}")
            return score

        except Exception as e:
            raise CustomException(e, sys)

    def run_advanced_pipeline(self) -> Dict[str, Any]:
        """
        Run advanced training pipeline with multiple models, feature engineering,
        and comprehensive evaluation.

        Returns:
            Dict with training results and model scores.
        """
        try:
            from src.data.preprocessing import DataPreprocessor
            from src.data.feature_engineering import FeatureEngineer
            from src.evaluation.metrics import evaluate_model, compare_models
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            import xgboost as xgb
            import lightgbm as lgb
            from catboost import CatBoostRegressor

            logging.info("Starting advanced training pipeline")

            # Load and preprocess data
            df = pd.read_csv(self.data_path, encoding="latin1")
            preprocessor = DataPreprocessor()
            df_clean = preprocessor.preprocess(df)

            # Feature engineering
            engineer = FeatureEngineer()
            df_features = engineer.create_time_series_features(df_clean)

            # Prepare features and target
            drop_cols = [
                preprocessor.target_column,
                preprocessor.date_column,
                "InvoiceNo", "StockCode", "Description",
                "Date", "Revenue",
            ]
            drop_cols = [c for c in drop_cols if c in df_features.columns]

            X = df_features.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
            y = df_features[preprocessor.target_column]

            # Temporal train/test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Define models
            models = {
                "LinearRegression": LinearRegression(),
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
                "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
                "CatBoost": CatBoostRegressor(iterations=100, random_seed=42, verbose=0),
            }

            results = {}
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                metrics = evaluate_model(y_test, y_pred, name)
                results[name] = metrics

                # Save model
                os.makedirs("models", exist_ok=True)
                save_object(f"models/{name.lower()}_model.pkl", model)

            # Save scaler and feature columns
            save_object("models/scaler.pkl", scaler)
            save_object("models/feature_columns.pkl", list(X.columns))

            comparison = compare_models(results)
            logging.info(f"\nModel Comparison:\n{comparison}")

            return {
                "results": results,
                "comparison": comparison,
                "feature_columns": list(X.columns),
                "best_model": comparison.index[0],
            }

        except Exception as e:
            raise CustomException(e, sys)