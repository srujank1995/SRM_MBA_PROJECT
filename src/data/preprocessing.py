"""Data preprocessing module for demand forecasting."""
import sys
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging


class DataPreprocessor:
    """
    Comprehensive data preprocessing for e-commerce demand forecasting.
    Handles missing values, outliers, encoding, and scaling.
    """

    def __init__(
        self,
        target_column: str = "Quantity",
        date_column: str = "InvoiceDate",
    ):
        """
        Initialize preprocessor.

        Args:
            target_column: Name of the target variable.
            date_column: Name of the date column.
        """
        self.target_column = target_column
        self.date_column = date_column
        self.scaler = StandardScaler()
        self.label_encoders: dict = {}
        self.imputer = SimpleImputer(strategy="median")

    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date column to datetime."""
        try:
            df = df.copy()
            df[self.date_column] = pd.to_datetime(df[self.date_column], infer_datetime_format=True)
            logging.info(f"Parsed {self.date_column} as datetime")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values:
        - Drop rows where target is missing
        - Impute numerical with median
        - Fill categorical with 'Unknown'
        """
        try:
            df = df.copy()
            initial_rows = len(df)

            # Drop rows with missing target
            df = df.dropna(subset=[self.target_column])

            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column in numerical_cols:
                numerical_cols.remove(self.target_column)

            categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
            if self.date_column in categorical_cols:
                categorical_cols.remove(self.date_column)

            if numerical_cols:
                df[numerical_cols] = self.imputer.fit_transform(df[numerical_cols])

            for col in categorical_cols:
                df[col] = df[col].fillna("Unknown")

            logging.info(
                f"Missing value handling: {initial_rows} -> {len(df)} rows"
            )
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def remove_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove outliers using IQR method.

        Args:
            df: Input DataFrame.
            columns: Columns to check. Defaults to target column.
        """
        try:
            df = df.copy()
            if columns is None:
                columns = [self.target_column]

            initial_rows = len(df)
            for col in columns:
                if col not in df.columns:
                    continue
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 3 * IQR
                upper = Q3 + 3 * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]

            logging.info(
                f"Outlier removal: {initial_rows} -> {len(df)} rows"
            )
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def filter_valid_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out returns, cancellations, and zero-quantity rows."""
        try:
            df = df.copy()
            initial_rows = len(df)

            # Remove negative quantities (returns)
            df = df[df[self.target_column] > 0]

            # Remove zero unit prices if column exists
            if "UnitPrice" in df.columns:
                df = df[df["UnitPrice"] > 0]

            # Remove cancelled invoices (starting with C)
            if "InvoiceNo" in df.columns:
                df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

            logging.info(
                f"Transaction filtering: {initial_rows} -> {len(df)} rows"
            )
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract year, month, day, weekday, hour from the date column."""
        try:
            df = df.copy()
            dt = df[self.date_column]
            df["Year"] = dt.dt.year
            df["Month"] = dt.dt.month
            df["Day"] = dt.dt.day
            df["DayOfWeek"] = dt.dt.dayofweek
            df["Hour"] = dt.dt.hour
            df["Quarter"] = dt.dt.quarter
            df["WeekOfYear"] = dt.dt.isocalendar().week.astype(int)
            df["IsWeekend"] = (dt.dt.dayofweek >= 5).astype(int)
            logging.info("Extracted date features")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def encode_categoricals(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Label-encode categorical columns."""
        try:
            df = df.copy()
            if columns is None:
                columns = df.select_dtypes(include=["object"]).columns.tolist()
                if self.date_column in columns:
                    columns.remove(self.date_column)

            for col in columns:
                if col not in df.columns:
                    continue
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

            logging.info(f"Encoded categorical columns: {columns}")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def create_revenue_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Revenue = Quantity * UnitPrice feature if possible."""
        try:
            df = df.copy()
            if "UnitPrice" in df.columns and self.target_column in df.columns:
                df["Revenue"] = df[self.target_column] * df["UnitPrice"]
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full preprocessing pipeline.

        Args:
            df: Raw input DataFrame.

        Returns:
            Preprocessed DataFrame.
        """
        try:
            logging.info("Starting full preprocessing pipeline")
            df = self.parse_dates(df)
            df = self.filter_valid_transactions(df)
            df = self.handle_missing_values(df)
            df = self.remove_outliers(df)
            df = self.extract_date_features(df)
            df = self.create_revenue_feature(df)
            logging.info(f"Preprocessing complete. Final shape: {df.shape}")
            return df
        except Exception as e:
            raise CustomException(e, sys)
