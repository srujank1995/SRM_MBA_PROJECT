"""Feature engineering for time-series demand forecasting."""
import sys
import pandas as pd
import numpy as np
from typing import List, Optional
from src.exception import CustomException
from src.logger import logging


class FeatureEngineer:
    """
    Creates time-series features: lags, rolling statistics,
    seasonal indicators, and interaction terms.
    """

    def __init__(
        self,
        target_column: str = "Quantity",
        date_column: str = "InvoiceDate",
        lag_periods: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
    ):
        """
        Initialize FeatureEngineer.

        Args:
            target_column: Target variable name.
            date_column: Date column name.
            lag_periods: Lag periods (days) for lag features.
            rolling_windows: Window sizes for rolling statistics.
        """
        self.target_column = target_column
        self.date_column = date_column
        self.lag_periods = lag_periods or [1, 7, 14, 30]
        self.rolling_windows = rolling_windows or [7, 14, 30]

    def _aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate to daily total quantity."""
        df = df.copy()
        if self.date_column in df.columns and not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column], infer_datetime_format=True)
        df["Date"] = df[self.date_column].dt.date
        daily = df.groupby("Date")[self.target_column].sum().reset_index()
        daily.columns = ["Date", self.target_column]
        daily["Date"] = pd.to_datetime(daily["Date"])
        daily = daily.sort_values("Date").reset_index(drop=True)
        return daily

    def add_lag_features(self, df: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
        """
        Add lag features for the target variable.

        Args:
            df: DataFrame with target column.
            group_col: If provided, compute lags within each group.
        """
        try:
            df = df.copy()
            for lag in self.lag_periods:
                col_name = f"{self.target_column}_lag_{lag}"
                if group_col and group_col in df.columns:
                    df[col_name] = df.groupby(group_col)[self.target_column].shift(lag)
                else:
                    df[col_name] = df[self.target_column].shift(lag)
            logging.info(f"Added lag features: {self.lag_periods}")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def add_rolling_features(self, df: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
        """
        Add rolling mean, std, min, max for the target.

        Args:
            df: DataFrame with target column.
            group_col: If provided, compute rolling stats within each group.
        """
        try:
            df = df.copy()
            for window in self.rolling_windows:
                if group_col and group_col in df.columns:
                    grp = df.groupby(group_col)[self.target_column]
                    df[f"{self.target_column}_roll_mean_{window}"] = grp.transform(
                        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                    )
                    df[f"{self.target_column}_roll_std_{window}"] = grp.transform(
                        lambda x: x.shift(1).rolling(window, min_periods=1).std()
                    )
                else:
                    shifted = df[self.target_column].shift(1)
                    df[f"{self.target_column}_roll_mean_{window}"] = shifted.rolling(window, min_periods=1).mean()
                    df[f"{self.target_column}_roll_std_{window}"] = shifted.rolling(window, min_periods=1).std()
            logging.info(f"Added rolling features: windows={self.rolling_windows}")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def add_seasonal_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add seasonal/calendar features.

        Requires date-related columns (Year, Month, Day, DayOfWeek, etc.)
        or a datetime date_column.
        """
        try:
            df = df.copy()
            date_col = None
            if self.date_column in df.columns and pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
                date_col = df[self.date_column]
            elif "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
                date_col = df["Date"]

            if date_col is not None:
                if "Month" not in df.columns:
                    df["Month"] = date_col.dt.month
                if "DayOfWeek" not in df.columns:
                    df["DayOfWeek"] = date_col.dt.dayofweek
                if "Quarter" not in df.columns:
                    df["Quarter"] = date_col.dt.quarter
                if "WeekOfYear" not in df.columns:
                    df["WeekOfYear"] = date_col.dt.isocalendar().week.astype(int)

            if "Month" in df.columns:
                df["IsHolidaySeason"] = df["Month"].isin([11, 12]).astype(int)
                df["IsSummer"] = df["Month"].isin([6, 7, 8]).astype(int)
                # Sine/cosine encoding for month cyclicality
                df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
                df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

            if "DayOfWeek" in df.columns:
                df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
                df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
                df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

            logging.info("Added seasonal indicator features")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature engineering pipeline for time-series data.

        Args:
            df: Preprocessed DataFrame.

        Returns:
            Feature-enriched DataFrame.
        """
        try:
            logging.info("Starting feature engineering pipeline")
            df = self.add_lag_features(df)
            df = self.add_rolling_features(df)
            df = self.add_seasonal_indicators(df)
            # Drop rows with NaN from lag features
            df = df.dropna().reset_index(drop=True)
            logging.info(f"Feature engineering complete. Shape: {df.shape}")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def prepare_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate to daily totals, then add all time-series features.

        Args:
            df: Raw preprocessed DataFrame.

        Returns:
            Daily feature DataFrame.
        """
        try:
            daily = self._aggregate_daily(df)
            daily = self.add_lag_features(daily)
            daily = self.add_rolling_features(daily)
            daily = self.add_seasonal_indicators(daily)
            daily = daily.dropna().reset_index(drop=True)
            return daily
        except Exception as e:
            raise CustomException(e, sys)
