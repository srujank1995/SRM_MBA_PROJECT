"""
Data Ingestion Module
This module handles loading, validating, and splitting the retail dataset
Dataset: Online Retail Transaction Data
Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.exceptions import CustomException
from src.logger import logging
from src.component.data_transformation import DataTransformation
from src.component.data_transformation import DataTransformationConfig
from src.component.model_trainer import ModelTrainerConfig
from src.component.model_trainer import ModelTrainer

# ============================================================================
# DATA INGESTION CONFIGURATION
# ============================================================================

@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion paths.
    
    Attributes:
        train_data_path: Path to store training dataset
        test_data_path: Path to store test dataset
        raw_data_path: Path to store raw dataset copy
    """
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")


# ============================================================================
# DATA INGESTION CLASS
# ============================================================================

class DataIngestion:
    """
    Main Data Ingestion Class
    Handles the complete data ingestion pipeline:
    1. Load data from CSV
    2. Explore dataset structure
    3. Validate data quality
    4. Handle missing values and duplicates
    5. Split into train-test sets
    6. Save processed datasets
    """
    
    def __init__(self):
        """Initialize data ingestion configuration."""
        self.ingestion_config = DataIngestionConfig()
        self.raw_data_path = r'notebook\Data\data.csv'

    def load_data(self):
        """
        STEP 1: Load the dataset from CSV file
        
        Returns:
            pd.DataFrame: Raw dataset as pandas DataFrame
            
        Raises:
            CustomException: If file not found or loading fails
        """
        logging.info("=" * 80)
        logging.info("STEP 1: LOADING DATA FROM CSV FILE")
        logging.info("=" * 80)
        try:
            df = pd.read_csv(self.raw_data_path)
            logging.info(f"✓ Successfully loaded dataset from: {self.raw_data_path}")
            logging.info(f"  Dataset shape: {df.shape} (rows, columns)")
            return df
        except FileNotFoundError as e:
            logging.error(f"✗ File not found: {self.raw_data_path}")
            raise CustomException(e, sys)
        except Exception as e:
            logging.error(f"✗ Error loading data: {str(e)}")
            raise CustomException(e, sys)

    def explore_dataset(self, df):
        """
        STEP 2: Explore dataset structure and basic statistics
        
        Args:
            df (pd.DataFrame): Dataset to explore
            
        Returns:
            pd.DataFrame: Same dataset (unchanged)
        """
        logging.info("\n" + "=" * 80)
        logging.info("STEP 2: DATASET EXPLORATION")
        logging.info("=" * 80)
        
        try:
            # Dataset Info
            logging.info(f"\n📊 DATASET OVERVIEW:")
            logging.info(f"  Total Records: {len(df)}")
            logging.info(f"  Total Features: {len(df.columns)}")
            logging.info(f"  Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Column Information
            logging.info(f"\n📋 COLUMN INFORMATION:")
            logging.info(f"  Columns: {', '.join(df.columns.tolist())}")
            logging.info(f"\n  Data Types:")
            for col, dtype in df.dtypes.items():
                logging.info(f"    - {col}: {dtype}")
            
            # Statistical Summary
            logging.info(f"\n📈 NUMERICAL COLUMNS STATISTICS:")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                logging.info(f"  {col}:")
                logging.info(f"    - Mean: {df[col].mean():.2f}")
                logging.info(f"    - Median: {df[col].median():.2f}")
                logging.info(f"    - Min: {df[col].min():.2f}")
                logging.info(f"    - Max: {df[col].max():.2f}")
                logging.info(f"    - Std Dev: {df[col].std():.2f}")
            
            # Sample Records
            logging.info(f"\n🔍 FIRST 5 RECORDS:")
            for idx, row in df.head().iterrows():
                logging.info(f"  Record {idx+1}: {dict(row)}")
            
            return df
        except Exception as e:
            logging.error(f"✗ Error during exploration: {str(e)}")
            raise CustomException(e, sys)

    def check_data_quality(self, df):
        """
        STEP 3: Check data quality (missing values, duplicates, outliers)
        
        Args:
            df (pd.DataFrame): Dataset to validate
            
        Returns:
            pd.DataFrame: Same dataset (unchanged)
        """
        logging.info("\n" + "=" * 80)
        logging.info("STEP 3: DATA QUALITY CHECK")
        logging.info("=" * 80)
        
        try:
            # Missing Values
            logging.info(f"\n❌ MISSING VALUES:")
            missing_data = df.isnull().sum()
            if missing_data.sum() == 0:
                logging.info(f"  ✓ No missing values found")
            else:
                for col in missing_data[missing_data > 0].index:
                    missing_percent = (missing_data[col] / len(df)) * 100
                    logging.info(f"  {col}: {missing_data[col]} ({missing_percent:.2f}%)")
            
            # Duplicate Records
            logging.info(f"\n🔄 DUPLICATE RECORDS:")
            duplicates = df.duplicated().sum()
            logging.info(f"  Total Duplicates: {duplicates}")
            if duplicates > 0:
                logging.info(f"  ⚠️  {(duplicates/len(df)*100):.2f}% of data is duplicated")
            else:
                logging.info(f"  ✓ No duplicate records found")
            
            # Outliers Detection (for numerical columns)
            logging.info(f"\n⚡ OUTLIERS DETECTION (IQR Method):")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    logging.info(f"  {col}: {outliers} outliers ({(outliers/len(df)*100):.2f}%)")
                else:
                    logging.info(f"  {col}: ✓ No outliers")
            
            # Value Counts for Categorical Columns
            logging.info(f"\n🏷️  CATEGORICAL COLUMNS - UNIQUE VALUES:")
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                logging.info(f"  {col}: {df[col].nunique()} unique values")
                logging.info(f"    Top 3 values: {df[col].value_counts().head(3).to_dict()}")
            
            return df
        except Exception as e:
            logging.error(f"✗ Error during quality check: {str(e)}")
            raise CustomException(e, sys)

    def preprocess_data(self, df):
        """
        STEP 4: Preprocess data (handle missing values, duplicates)
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        logging.info("\n" + "=" * 80)
        logging.info("STEP 4: DATA PREPROCESSING")
        logging.info("=" * 80)
        
        try:
            initial_rows = len(df)
            
            # Remove duplicate records
            logging.info(f"\n🔄 Removing Duplicates...")
            df = df.drop_duplicates().reset_index(drop=True)
            removed_duplicates = initial_rows - len(df)
            if removed_duplicates > 0:
                logging.info(f"  ✓ Removed {removed_duplicates} duplicate records")
            else:
                logging.info(f"  ✓ No duplicates to remove")
            
            # Handle missing values
            logging.info(f"\n❌ Handling Missing Values...")
            missing_before = df.isnull().sum().sum()
            
            # Remove rows with missing values (optional - based on your strategy)
            df = df.dropna()
            missing_after = df.isnull().sum().sum()
            
            if missing_before > 0:
                rows_removed = initial_rows - len(df)
                logging.info(f"  ✓ Removed {missing_before} missing values")
                logging.info(f"  ✓ {rows_removed} rows removed due to missing values")
            else:
                logging.info(f"  ✓ No missing values to handle")
            
            logging.info(f"\n✓ Preprocessing Complete")
            logging.info(f"  Initial rows: {initial_rows}")
            logging.info(f"  Final rows: {len(df)}")
            logging.info(f"  Rows removed: {initial_rows - len(df)}")
            
            return df
        except Exception as e:
            logging.error(f"✗ Error during preprocessing: {str(e)}")
            raise CustomException(e, sys)

    def save_raw_data(self, df):
        """
        STEP 5: Save raw/cleaned data to artifacts folder
        
        Args:
            df (pd.DataFrame): Cleaned dataset
            
        Returns:
            str: Path to saved raw data
        """
        logging.info("\n" + "=" * 80)
        logging.info("STEP 5: SAVING RAW DATA")
        logging.info("=" * 80)
        
        try:
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"✓ Raw data saved to: {self.ingestion_config.raw_data_path}")
            logging.info(f"  Total records: {len(df)}")
            return self.ingestion_config.raw_data_path
        except Exception as e:
            logging.error(f"✗ Error saving raw data: {str(e)}")
            raise CustomException(e, sys)

    def split_train_test(self, df):
        """
        STEP 6: Split data into training and testing sets
        
        Args:
            df (pd.DataFrame): Cleaned dataset
            
        Returns:
            tuple: (train_data_path, test_data_path)
        """
        logging.info("\n" + "=" * 80)
        logging.info("STEP 6: TRAIN-TEST SPLIT")
        logging.info("=" * 80)
        
        try:
            logging.info(f"\n✓ Splitting data into train-test sets...")
            logging.info(f"  Test size: 20%")
            logging.info(f"  Train size: 80%")
            logging.info(f"  Random state: 42")
            
            # Split with test_size=0.2 (80-20 split)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            logging.info(f"\n✓ Split Complete:")
            logging.info(f"  Training set: {len(train_set)} records ({(len(train_set)/len(df)*100):.2f}%)")
            logging.info(f"  Testing set: {len(test_set)} records ({(len(test_set)/len(df)*100):.2f}%)")
            
            # Save training data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info(f"\n✓ Training data saved to: {self.ingestion_config.train_data_path}")
            
            # Save testing data
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"✓ Testing data saved to: {self.ingestion_config.test_data_path}")
            
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            logging.error(f"✗ Error during train-test split: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        """
        Main orchestration method that runs the entire data ingestion pipeline
        
        Returns:
            tuple: (train_data_path, test_data_path)
            
        Raises:
            CustomException: If any step fails
        """
        logging.info("\n")
        logging.info("╔" + "=" * 78 + "╗")
        logging.info("║" + " " * 78 + "║")
        logging.info("║" + "INITIATING DATA INGESTION PIPELINE".center(78) + "║")
        logging.info("║" + " " * 78 + "║")
        logging.info("╚" + "=" * 78 + "╝")
        
        try:
            # Step 1: Load data
            df = self.load_data()
            
            # Step 2: Explore dataset
            df = self.explore_dataset(df)
            
            # Step 3: Check data quality
            df = self.check_data_quality(df)
            
            # Step 4: Preprocess data
            df = self.preprocess_data(df)
            
            # Step 5: Save raw data
            self.save_raw_data(df)
            
            # Step 6: Split and save train-test data
            train_data_path, test_data_path = self.split_train_test(df)
            
            logging.info("\n")
            logging.info("╔" + "=" * 78 + "╗")
            logging.info("║" + " " * 78 + "║")
            logging.info("║" + "✓ DATA INGESTION COMPLETED SUCCESSFULLY".center(78) + "║")
            logging.info("║" + " " * 78 + "║")
            logging.info("╚" + "=" * 78 + "╝")
            
            return train_data_path, test_data_path
            
        except Exception as e:
            logging.error(f"\n✗ DATA INGESTION FAILED: {str(e)}")
            raise CustomException(e, sys)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution block
    Execute the complete ML pipeline:
    1. Data Ingestion
    2. Data Transformation
    3. Model Training
    """
    
    try:
        # Step 1: Data Ingestion
        print("\n" + "=" * 80)
        print("STARTING DATA INGESTION PIPELINE")
        print("=" * 80)
        
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        
        print(f"\n✓ Train data path: {train_data}")
        print(f"✓ Test data path: {test_data}")
        
        # Step 2: Data Transformation
        print("\n" + "=" * 80)
        print("STARTING DATA TRANSFORMATION")
        print("=" * 80)
        
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
        
        print(f"✓ Data transformation completed")
        print(f"  Train array shape: {train_arr.shape}")
        print(f"  Test array shape: {test_arr.shape}")
        
        # Step 3: Model Training
        print("\n" + "=" * 80)
        print("STARTING MODEL TRAINING")
        print("=" * 80)
        
        modeltrainer = ModelTrainer()
        model_result = modeltrainer.initiate_model_trainer(train_arr, test_arr)
        
        print(f"\n✓ Model training completed")
        print(f"  Result: {model_result}")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {str(e)}")
        raise



