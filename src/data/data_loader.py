"""Data loading utilities for the demand forecasting system."""
import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from src.exception import CustomException
from src.logger import logging


class DataLoader:
    """Handles loading and basic validation of datasets."""

    def __init__(self, data_path: str):
        """
        Initialize DataLoader.

        Args:
            data_path: Path to the data file or directory.
        """
        self.data_path = data_path

    def load_csv(self, file_path: Optional[str] = None, encoding: str = "latin1") -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame.

        Args:
            file_path: Path to CSV file. Uses self.data_path if None.
            encoding: File encoding.

        Returns:
            Loaded DataFrame.
        """
        try:
            path = file_path or self.data_path
            logging.info(f"Loading data from: {path}")
            df = pd.read_csv(path, encoding=encoding)
            logging.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def validate_schema(self, df: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate that required columns are present.

        Args:
            df: DataFrame to validate.
            required_columns: List of required column names.

        Returns:
            True if valid, raises exception otherwise.
        """
        try:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            logging.info("Schema validation passed")
            return True
        except Exception as e:
            raise CustomException(e, sys)

    def generate_sample_data(self, n_rows: int = 5000) -> pd.DataFrame:
        """
        Generate synthetic e-commerce retail data for demo/testing.

        Args:
            n_rows: Number of rows to generate.

        Returns:
            Synthetic DataFrame mimicking the Online Retail dataset.
        """
        try:
            np.random.seed(42)
            dates = pd.date_range(start="2020-01-01", periods=n_rows, freq="H")
            countries = ["United Kingdom", "Germany", "France", "Netherlands",
                         "Australia", "Spain", "Belgium", "Sweden", "India", "USA"]
            stock_codes = [f"SC{i:04d}" for i in range(50)]
            descriptions = [
                "WHITE HANGING HEART T-LIGHT HOLDER", "WHITE METAL LANTERN",
                "CREAM CUPID HEARTS COAT HANGER", "KNITTED UNION FLAG HOT WATER BOTTLE",
                "RED WOOLLY HOTTIE WHITE HEART", "SET 7 BABUSHKA NESTING BOXES",
                "GLASS STAR FROSTED T-LIGHT HOLDER", "HAND WARMER UNION JACK",
                "HAND WARMER RED POLKA DOT", "ASSORTED COLOUR BIRD ORNAMENT",
            ]

            df = pd.DataFrame({
                "InvoiceNo": [f"INV{i:06d}" for i in range(n_rows)],
                "StockCode": np.random.choice(stock_codes, n_rows),
                "Description": np.random.choice(descriptions, n_rows),
                "Quantity": np.random.randint(1, 50, n_rows),
                "InvoiceDate": dates.strftime("%m/%d/%Y %H:%M"),
                "UnitPrice": np.round(np.random.uniform(0.5, 50.0, n_rows), 2),
                "CustomerID": np.random.randint(10000, 20000, n_rows),
                "Country": np.random.choice(countries, n_rows),
            })

            return df
        except Exception as e:
            raise CustomException(e, sys)
