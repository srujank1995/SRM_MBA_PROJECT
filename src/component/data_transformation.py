"""
Data Transformation Module
This module handles feature engineering, encoding, and scaling for the retail dataset
Dataset: Online Retail Transaction Data
Target Variable: Total Transaction Amount (for prediction/forecasting)
"""

import sys
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object

# ============================================================================
# DATA TRANSFORMATION CONFIGURATION
# ============================================================================

@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation paths.
    
    Attributes:
        preprocessor_obj_file_path: Path to save the preprocessor pipeline
    """
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


# ============================================================================
# DATA TRANSFORMATION CLASS
# ============================================================================

class DataTransformation:
    """
    Main Data Transformation Class
    Handles the complete feature engineering and preprocessing pipeline:
    1. Load train and test data
    2. Create new features from raw data
    3. Separate features and target variable
    4. Build transformation pipelines (numerical and categorical)
    5. Apply transformations
    6. Save preprocessor object
    """
    
    def __init__(self):
        """Initialize data transformation configuration."""
        self.data_transformation_config = DataTransformationConfig()

    def feature_engineering(self, df):
        """
        STEP 1: Create new features from raw data
        Engineer meaningful features from the retail transaction data
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        logging.info("\n" + "=" * 80)
        logging.info("STEP 1: FEATURE ENGINEERING")
        logging.info("=" * 80)
        
        try:
            df = df.copy()
            initial_cols = len(df.columns)
            
            logging.info(f"\n🔧 Creating New Features...")
            
            # Convert InvoiceDate to datetime
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            logging.info(f"  ✓ Converted InvoiceDate to datetime format")
            
            # Extract temporal features
            df['Month'] = df['InvoiceDate'].dt.month
            df['Day'] = df['InvoiceDate'].dt.day
            df['Hour'] = df['InvoiceDate'].dt.hour
            df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek  # 0=Monday, 6=Sunday
            df['Quarter'] = df['InvoiceDate'].dt.quarter
            logging.info(f"  ✓ Extracted temporal features (Month, Day, Hour, DayOfWeek, Quarter)")
            
            # Create transaction amount feature (Total Sale Value)
            df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
            logging.info(f"  ✓ Created TotalAmount = Quantity × UnitPrice")
            
            # Create average price per unit feature
            df['IsHighPrice'] = (df['UnitPrice'] > df['UnitPrice'].median()).astype(int)
            logging.info(f"  ✓ Created IsHighPrice feature (binary)")
            
            # Quantity categories
            df['QuantityCategory'] = pd.cut(df['Quantity'], 
                                            bins=[0, 10, 50, df['Quantity'].max()], 
                                            labels=['Low', 'Medium', 'High'])
            logging.info(f"  ✓ Created QuantityCategory feature")
            
            # Invoice-level features (invoices often have multiple items)
            df['ItemsPerInvoice'] = df.groupby('InvoiceNo')['StockCode'].transform('count')
            logging.info(f"  ✓ Created ItemsPerInvoice feature")
            
            # Customer purchase frequency (in dataset)
            df['CustomerFrequency'] = df.groupby('CustomerID')['InvoiceNo'].transform('nunique')
            logging.info(f"  ✓ Created CustomerFrequency feature")
            
            # Remove the original InvoiceDate (temporal features extracted)
            df = df.drop(columns=['InvoiceDate', 'InvoiceNo', 'Description', 'StockCode'])
            logging.info(f"  ✓ Dropped redundant columns (InvoiceDate, InvoiceNo, Description, StockCode)")
            
            final_cols = len(df.columns)
            logging.info(f"\n✓ Feature Engineering Complete")
            logging.info(f"  Initial columns: {initial_cols}")
            logging.info(f"  Final columns: {final_cols}")
            logging.info(f"  New features created: {final_cols - initial_cols + 4}")
            
            return df
            
        except Exception as e:
            logging.error(f"✗ Error during feature engineering: {str(e)}")
            raise CustomException(e, sys)

    def identify_feature_types(self, df):
        """
        STEP 2: Identify and separate numerical and categorical features
        
        Args:
            df (pd.DataFrame): Transformed dataframe
            
        Returns:
            tuple: (numerical_columns, categorical_columns)
        """
        logging.info("\n" + "=" * 80)
        logging.info("STEP 2: IDENTIFY FEATURE TYPES")
        logging.info("=" * 80)
        
        try:
            # Separate features
            numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            logging.info(f"\n📊 NUMERICAL FEATURES ({len(numerical_columns)}):")
            for col in numerical_columns:
                logging.info(f"  - {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
            
            logging.info(f"\n🏷️  CATEGORICAL FEATURES ({len(categorical_columns)}):")
            for col in categorical_columns:
                unique_count = df[col].nunique()
                logging.info(f"  - {col}: {unique_count} unique values")
                if unique_count <= 10:
                    logging.info(f"    Values: {df[col].unique().tolist()}")
                else:
                    logging.info(f"    Top 5: {df[col].value_counts().head().to_dict()}")
            
            return numerical_columns, categorical_columns
            
        except Exception as e:
            logging.error(f"✗ Error identifying feature types: {str(e)}")
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        """
        STEP 3: Create preprocessing pipelines for numerical and categorical features
        Build sklearn transformation pipelines for different feature types
        
        Returns:
            ColumnTransformer: Combined preprocessor object
            
        Raises:
            CustomException: If pipeline creation fails
        """
        logging.info("\n" + "=" * 80)
        logging.info("STEP 3: BUILD PREPROCESSING PIPELINES")
        logging.info("=" * 80)
        
        try:
            # Define feature columns
            numerical_columns = [
                'Quantity', 'UnitPrice', 'CustomerID', 'Month', 'Day', 'Hour',
                'DayOfWeek', 'Quarter', 'TotalAmount', 'IsHighPrice', 
                'ItemsPerInvoice', 'CustomerFrequency'
            ]
            
            categorical_columns = ['Country', 'QuantityCategory']
            
            logging.info(f"\n🔧 Numerical Columns ({len(numerical_columns)}):")
            for col in numerical_columns:
                logging.info(f"  - {col}")
            
            logging.info(f"\n🏷️  Categorical Columns ({len(categorical_columns)}):")
            for col in categorical_columns:
                logging.info(f"  - {col}")
            
            # ================================================================
            # NUMERICAL PIPELINE
            # ================================================================
            logging.info(f"\n📈 Building Numerical Pipeline...")
            logging.info(f"  Step 1: Missing Value Imputation (strategy='median')")
            logging.info(f"  Step 2: Feature Scaling (StandardScaler - z-score normalization)")
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            logging.info(f"  ✓ Numerical pipeline created successfully")
            
            # ================================================================
            # CATEGORICAL PIPELINE
            # ================================================================
            logging.info(f"\n🏷️  Building Categorical Pipeline...")
            logging.info(f"  Step 1: Missing Value Imputation (strategy='most_frequent')")
            logging.info(f"  Step 2: One-Hot Encoding (convert categories to binary columns)")
            logging.info(f"  Step 3: Feature Scaling (StandardScaler with mean=False)")
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"  ✓ Categorical pipeline created successfully")
            
            # ================================================================
            # COLUMN TRANSFORMER (Combine both pipelines)
            # ================================================================
            logging.info(f"\n⚙️  Combining Pipelines with ColumnTransformer...")
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ],
                remainder='drop'  # Drop columns not specified
            )
            
            logging.info(f"✓ Preprocessor object created successfully")
            logging.info(f"  - Numerical features: {len(numerical_columns)}")
            logging.info(f"  - Categorical features: {len(categorical_columns)}")
            
            return preprocessor, numerical_columns, categorical_columns
            
        except Exception as e:
            logging.error(f"✗ Error creating preprocessor: {str(e)}")
            raise CustomException(e, sys)

    def prepare_features_target(self, df, target_col='TotalAmount'):
        """
        STEP 4: Separate features (X) and target variable (y)
        
        Args:
            df (pd.DataFrame): Transformed dataframe
            target_col (str): Name of target column
            
        Returns:
            tuple: (X, y) where X is features and y is target
        """
        logging.info("\n" + "=" * 80)
        logging.info("STEP 4: SEPARATE FEATURES AND TARGET")
        logging.info("=" * 80)
        
        try:
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataframe")
            
            logging.info(f"\n✓ Target Variable: {target_col}")
            logging.info(f"  Target Statistics:")
            logging.info(f"    - Mean: {df[target_col].mean():.2f}")
            logging.info(f"    - Median: {df[target_col].median():.2f}")
            logging.info(f"    - Std Dev: {df[target_col].std():.2f}")
            logging.info(f"    - Min: {df[target_col].min():.2f}")
            logging.info(f"    - Max: {df[target_col].max():.2f}")
            
            X = df.drop(columns=[target_col], axis=1)
            y = df[target_col]
            
            logging.info(f"\n✓ Feature-Target Separation Complete")
            logging.info(f"  X (Features) shape: {X.shape}")
            logging.info(f"  y (Target) shape: {y.shape}")
            
            return X, y
            
        except Exception as e:
            logging.error(f"✗ Error separating features and target: {str(e)}")
            raise CustomException(e, sys)

    def apply_transformations(self, X_train, y_train, X_test, y_test, preprocessor):
        """
        STEP 5: Apply transformations to training and testing data
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_test (pd.DataFrame): Testing features
            y_test (pd.Series): Testing target
            preprocessor: ColumnTransformer object
            
        Returns:
            tuple: (train_arr, test_arr) as numpy arrays
        """
        logging.info("\n" + "=" * 80)
        logging.info("STEP 5: APPLY TRANSFORMATIONS")
        logging.info("=" * 80)
        
        try:
            logging.info(f"\n🔄 Fitting preprocessor on training data...")
            X_train_transformed = preprocessor.fit_transform(X_train)
            logging.info(f"  ✓ Training data transformed")
            logging.info(f"    Input shape: {X_train.shape}")
            logging.info(f"    Output shape: {X_train_transformed.shape}")
            
            logging.info(f"\n🔄 Applying preprocessor to testing data...")
            X_test_transformed = preprocessor.transform(X_test)
            logging.info(f"  ✓ Testing data transformed")
            logging.info(f"    Input shape: {X_test.shape}")
            logging.info(f"    Output shape: {X_test_transformed.shape}")
            
            # Combine features and target into single arrays
            logging.info(f"\n⚡ Combining features with target variable...")
            
            y_train_array = np.array(y_train).reshape(-1, 1)
            y_test_array = np.array(y_test).reshape(-1, 1)
            
            train_arr = np.c_[X_train_transformed, y_train_array]
            test_arr = np.c_[X_test_transformed, y_test_array]
            
            logging.info(f"  ✓ Combined arrays created")
            logging.info(f"    Train array shape: {train_arr.shape}")
            logging.info(f"    Test array shape: {test_arr.shape}")
            
            return train_arr, test_arr
            
        except Exception as e:
            logging.error(f"✗ Error applying transformations: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        MAIN ORCHESTRATION METHOD
        Execute the complete data transformation pipeline
        
        Args:
            train_path (str): Path to training data CSV
            test_path (str): Path to testing data CSV
            
        Returns:
            tuple: (train_arr, test_arr, preprocessor_obj_file_path)
            
        Raises:
            CustomException: If any transformation step fails
        """
        logging.info("\n")
        logging.info("╔" + "=" * 78 + "╗")
        logging.info("║" + " " * 78 + "║")
        logging.info("║" + "INITIATING DATA TRANSFORMATION PIPELINE".center(78) + "║")
        logging.info("║" + " " * 78 + "║")
        logging.info("╚" + "=" * 78 + "╝")
        
        try:
            # Load data
            logging.info(f"\n📂 Loading data files...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"  ✓ Training data loaded: {train_df.shape}")
            logging.info(f"  ✓ Testing data loaded: {test_df.shape}")
            
            # Step 1: Feature Engineering
            logging.info(f"\n🔄 Applying Feature Engineering...")
            train_df = self.feature_engineering(train_df)
            test_df = self.feature_engineering(test_df)
            
            # Step 2: Identify Feature Types
            logging.info(f"\n🔄 Identifying Feature Types...")
            numerical_cols, categorical_cols = self.identify_feature_types(train_df)
            
            # Step 3: Build Preprocessor
            logging.info(f"\n🔄 Building Preprocessor Pipeline...")
            preprocessor, num_cols, cat_cols = self.get_data_transformer_object()
            
            # Step 4: Separate Features and Target
            logging.info(f"\n🔄 Separating Features and Target...")
            X_train, y_train = self.prepare_features_target(train_df, target_col='TotalAmount')
            X_test, y_test = self.prepare_features_target(test_df, target_col='TotalAmount')
            
            # Step 5: Apply Transformations
            logging.info(f"\n🔄 Applying Transformations...")
            train_arr, test_arr = self.apply_transformations(X_train, y_train, X_test, y_test, preprocessor)
            
            # Save Preprocessor Object
            logging.info(f"\n💾 Saving Preprocessor Object...")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info(f"  ✓ Preprocessor saved to: {self.data_transformation_config.preprocessor_obj_file_path}")
            
            logging.info("\n")
            logging.info("╔" + "=" * 78 + "╗")
            logging.info("║" + " " * 78 + "║")
            logging.info("║" + "✓ DATA TRANSFORMATION COMPLETED SUCCESSFULLY".center(78) + "║")
            logging.info("║" + " " * 78 + "║")
            logging.info("╚" + "=" * 78 + "╝")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.error(f"\n✗ DATA TRANSFORMATION FAILED: {str(e)}")
            raise CustomException(e, sys)
