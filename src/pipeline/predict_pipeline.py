"""
Prediction Pipeline Module
This module handles making predictions on new retail transaction data
Uses trained model and preprocessor to predict transaction amounts
"""

import sys
import os
from datetime import datetime
from typing import Union, List

import pandas as pd
import numpy as np

from src.exceptions import CustomException
from src.logger import logging
from src.utils import load_object


# ============================================================================
# PREDICTION PIPELINE CLASS
# ============================================================================

class PredictPipeline:
    """
    Pipeline for making predictions on new retail transaction data
    
    Processes:
    1. Load saved model and preprocessor
    2. Validate input data
    3. Apply preprocessing transformations
    4. Generate predictions
    5. Format and return results
    """
    
    def __init__(self):
        """Initialize prediction pipeline by loading model and preprocessor."""
        logging.info("Initializing Prediction Pipeline...")
        self.model = None
        self.preprocessor = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """
        STEP 1: Load Model and Preprocessor Objects
        
        Loads the trained model and preprocessor from artifacts folder
        
        Raises:
            CustomException: If artifacts cannot be loaded
        """
        logging.info("\n" + "=" * 80)
        logging.info("STEP 1: LOADING MODEL AND PREPROCESSOR ARTIFACTS")
        logging.info("=" * 80)
        
        # Define artifact paths
        model_path = os.path.join("artifacts", "model.pkl")
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        
        # Check if files exist
        logging.info(f"\n🔍 Checking artifact files...")
        if not os.path.exists(model_path):
            logging.error(f"✗ Model file not found: {model_path}")
            logging.error(f"  Current working directory: {os.getcwd()}")
            logging.error(f"  Available files in artifacts/:")
            if os.path.exists("artifacts"):
                logging.error(f"    {os.listdir('artifacts')}")
            raise CustomException(
                f"Model file not found at: {model_path}\n"
                f"Current working directory: {os.getcwd()}\n"
                f"Please ensure model.pkl exists in the artifacts folder.",
                sys
            )
        
        if not os.path.exists(preprocessor_path):
            logging.error(f"✗ Preprocessor file not found: {preprocessor_path}")
            raise CustomException(
                f"Preprocessor file not found at: {preprocessor_path}\n"
                f"Current working directory: {os.getcwd()}\n"
                f"Please ensure preprocessor.pkl exists in the artifacts folder.",
                sys
            )
        
        logging.info(f"  ✓ Both artifact files found")
        
        try:
            # Load Model
            logging.info(f"\n📦 Loading Model from {model_path}...")
            self.model = load_object(file_path=model_path)
            
            if self.model is None:
                raise ValueError("Model object is None after loading")
            
            logging.info(f"  ✓ Model loaded successfully")
            logging.info(f"  Model type: {type(self.model).__name__}")
            
            # Load Preprocessor
            logging.info(f"\n📦 Loading Preprocessor from {preprocessor_path}...")
            self.preprocessor = load_object(file_path=preprocessor_path)
            
            if self.preprocessor is None:
                raise ValueError("Preprocessor object is None after loading")
            
            logging.info(f"  ✓ Preprocessor loaded successfully")
            logging.info(f"  Preprocessor type: {type(self.preprocessor).__name__}")
            
        except FileNotFoundError as e:
            logging.error(f"✗ File not found during loading: {str(e)}")
            raise CustomException(
                f"Could not find required artifact file:\n{str(e)}\n"
                f"Please run the training pipeline first to generate model.pkl and preprocessor.pkl",
                sys
            )
        except ValueError as e:
            logging.error(f"✗ Artifact is None after loading: {str(e)}")
            raise CustomException(
                f"Artifact file is corrupted or empty:\n{str(e)}\n"
                f"Please retrain the model to generate valid artifacts.",
                sys
            )
        except Exception as e:
            logging.error(f"✗ Error loading artifacts: {str(e)}")
            raise CustomException(
                f"Error loading artifacts:\n{str(e)}\n"
                f"Troubleshooting steps:\n"
                f"1. Ensure both model.pkl and preprocessor.pkl exist in artifacts/\n"
                f"2. Run the training pipeline if artifacts don't exist\n"
                f"3. Check file permissions\n"
                f"4. Ensure Python version compatibility for pickle files",
                sys
            )
    
    def _verify_artifacts(self):
        """
        STEP 0.5: Verify that artifacts were loaded successfully
        
        Ensures both model and preprocessor are valid objects before proceeding
        with predictions.
        
        Raises:
            CustomException: If either artifact is None or invalid
        """
        logging.info("\n" + "=" * 80)
        logging.info("STEP 0.5: VERIFYING ARTIFACTS")
        logging.info("=" * 80)
        
        try:
            if self.model is None:
                raise ValueError("Model is None - failed to load model artifact")
            
            if self.preprocessor is None:
                raise ValueError("Preprocessor is None - failed to load preprocessor artifact")
            
            # Verify model has predict method
            if not hasattr(self.model, 'predict'):
                raise ValueError(
                    f"Model does not have 'predict' method. "
                    f"Model type: {type(self.model).__name__}"
                )
            
            # Verify preprocessor has transform method
            if not hasattr(self.preprocessor, 'transform'):
                raise ValueError(
                    f"Preprocessor does not have 'transform' method. "
                    f"Preprocessor type: {type(self.preprocessor).__name__}"
                )
            
            logging.info(f"\n✓ Artifact verification successful:")
            logging.info(f"  ✓ Model is valid ({type(self.model).__name__})")
            logging.info(f"  ✓ Model has predict method")
            logging.info(f"  ✓ Preprocessor is valid ({type(self.preprocessor).__name__})")
            logging.info(f"  ✓ Preprocessor has transform method")
            logging.info(f"\n✓ Pipeline ready for predictions")
            
        except Exception as e:
            logging.error(f"✗ Artifact verification failed: {str(e)}")
            raise CustomException(
                f"Artifact verification failed:\n{str(e)}\n\n"
                f"This usually means:\n"
                f"1. The artifact files are corrupted or incomplete\n"
                f"2. The trained models are not compatible\n"
                f"3. You need to run the training pipeline again\n\n"
                f"Solution: Run train_pipeline.py to generate valid artifacts",
                sys
            )
    
    def validate_input(self, features):
        """
        STEP 2: Validate Input Data
        
        Args:
            features (pd.DataFrame): Input features to validate
            
        Raises:
            CustomException: If validation fails
        """
        logging.info("\n" + "=" * 80)
        logging.info("STEP 2: VALIDATING INPUT DATA")
        logging.info("=" * 80)
        
        try:
            logging.info(f"\n✓ Input Data Validation:")
            logging.info(f"  Shape: {features.shape}")
            logging.info(f"  Columns: {features.columns.tolist()}")
            
            # Check for missing values
            if features.isnull().any().any():
                missing_cols = features.columns[features.isnull().any()].tolist()
                logging.warning(f"  ⚠️  Missing values found in columns: {missing_cols}")
            else:
                logging.info(f"  ✓ No missing values")
            
            logging.info(f"  Data types: {features.dtypes.to_dict()}")
            
        except Exception as e:
            logging.error(f"✗ Validation error: {str(e)}")
            raise CustomException(e, sys)
    
    def transform_features(self, features):
        """
        STEP 3: Apply Preprocessing Transformations
        
        Args:
            features (pd.DataFrame): Raw features
            
        Returns:
            np.ndarray: Transformed feature array
            
        Raises:
            CustomException: If transformation fails
        """
        logging.info("\n" + "=" * 80)
        logging.info("STEP 3: APPLYING PREPROCESSING TRANSFORMATIONS")
        logging.info("=" * 80)
        
        try:
            # Verify preprocessor is available
            if self.preprocessor is None:
                logging.error(f"✗ Preprocessor is None - artifacts not loaded properly")
                raise ValueError("Preprocessor is None. Please ensure model artifacts are valid.")
            
            if not hasattr(self.preprocessor, 'transform'):
                logging.error(f"✗ Preprocessor does not have transform method")
                raise ValueError(f"Preprocessor type {type(self.preprocessor).__name__} does not support transformation")
            
            logging.info(f"\n🔄 Transforming features using preprocessor...")
            logging.info(f"  Input shape: {features.shape}")
            logging.info(f"  Input columns: {features.columns.tolist()}")
            
            # Ensure columns are in correct order (numerical first, then categorical)
            numerical_columns = [
                'Quantity', 'UnitPrice', 'CustomerID', 'Month', 'Day', 'Hour',
                'DayOfWeek', 'Quarter', 'TotalAmount', 'IsHighPrice',
                'ItemsPerInvoice', 'CustomerFrequency'
            ]
            categorical_columns = ['Country', 'QuantityCategory']
            
            # Reorder features to match preprocessor's expected column order
            expected_columns = numerical_columns + categorical_columns
            
            # Check if all expected columns exist
            missing_cols = [col for col in expected_columns if col not in features.columns]
            if missing_cols:
                logging.error(f"✗ Missing columns: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Reorder dataframe columns
            features = features[expected_columns]
            logging.info(f"  ✓ Columns reordered to match preprocessor")
            
            # Apply preprocessing transformations
            data_scaled = self.preprocessor.transform(features)
            
            # Convert sparse matrix to dense array if necessary
            if hasattr(data_scaled, 'toarray'):
                logging.info(f"  ℹ️  Converting sparse matrix to dense array")
                data_scaled = data_scaled.toarray()
            
            logging.info(f"  ✓ Features transformed successfully")
            logging.info(f"  Output shape: {data_scaled.shape}")
            logging.info(f"  Data type: {type(data_scaled).__name__}")
            
            if isinstance(data_scaled, np.ndarray):
                logging.info(f"  Value range: min={np.min(data_scaled):.4f}, max={np.max(data_scaled):.4f}")
            
            return data_scaled
            
        except Exception as e:
            logging.error(f"✗ Error during transformation: {str(e)}")
            raise CustomException(e, sys)
    
    def generate_predictions(self, data_scaled):
        """
        STEP 4: Generate Predictions Using Model
        
        Args:
            data_scaled (np.ndarray): Transformed feature array
            
        Returns:
            np.ndarray: Predicted values
            
        Raises:
            CustomException: If prediction fails
        """
        logging.info("\n" + "=" * 80)
        logging.info("STEP 4: GENERATING PREDICTIONS")
        logging.info("=" * 80)
        
        try:
            # Verify model is available
            if self.model is None:
                logging.error(f"✗ Model is None - artifacts not loaded properly")
                raise ValueError("Model is None. Please ensure model artifacts are valid.")
            
            if not hasattr(self.model, 'predict'):
                logging.error(f"✗ Model does not have predict method")
                raise ValueError(f"Model type {type(self.model).__name__} does not support prediction")
            
            logging.info(f"\n🤖 Running model prediction...")
            logging.info(f"  Input shape: {data_scaled.shape}")
            logging.info(f"  Input type: {type(data_scaled).__name__}")
            
            # Ensure data is numpy array (not sparse matrix)
            if hasattr(data_scaled, 'toarray'):
                logging.info(f"  ℹ️  Converting sparse matrix to dense array")
                data_scaled = data_scaled.toarray()
            
            # Ensure data is 2D array
            if data_scaled.ndim == 1:
                logging.info(f"  ℹ️  Reshaping 1D array to 2D")
                data_scaled = data_scaled.reshape(1, -1)
            
            # Generate predictions
            predictions = self.model.predict(data_scaled)
            
            # Reshape to 1D if needed
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            
            logging.info(f"  ✓ Prediction successful")
            logging.info(f"  Output shape: {predictions.shape}")
            logging.info(f"  Output type: {type(predictions).__name__}")
            logging.info(f"  Prediction statistics:")
            
            if isinstance(predictions, np.ndarray):
                logging.info(f"    - Mean: £{np.mean(predictions):.2f}")
                logging.info(f"    - Std Dev: £{np.std(predictions):.2f}")
                logging.info(f"    - Min: £{np.min(predictions):.2f}")
                logging.info(f"    - Max: £{np.max(predictions):.2f}")
            
            return predictions
            
        except Exception as e:
            logging.error(f"✗ Error during prediction: {str(e)}")
            logging.error(f"  Data shape: {data_scaled.shape if hasattr(data_scaled, 'shape') else 'unknown'}")
            logging.error(f"  Model type: {type(self.model).__name__ if self.model else 'None'}")
            raise CustomException(e, sys)
    
    def format_results(self, predictions, original_data=None):
        """
        STEP 5: Format and Return Results
        
        Args:
            predictions (np.ndarray): Model predictions
            original_data (pd.DataFrame, optional): Original input data
            
        Returns:
            Union[np.ndarray, pd.DataFrame]: Formatted predictions
        """
        logging.info("\n" + "=" * 80)
        logging.info("STEP 5: FORMATTING RESULTS")
        logging.info("=" * 80)
        
        try:
            logging.info(f"\n📊 Formatting prediction results...")
            
            # Convert to positive values (if any negative from model)
            predictions = np.abs(predictions)
            
            logging.info(f"  ✓ Results formatted")
            logging.info(f"    Predictions shape: {predictions.shape}")
            
            # If original data provided, create results dataframe
            if original_data is not None:
                results_df = original_data.copy()
                results_df['PredictedAmount'] = predictions
                logging.info(f"  ✓ Results combined with input data")
                return results_df
            
            return predictions
            
        except Exception as e:
            logging.error(f"✗ Error formatting results: {str(e)}")
            raise CustomException(e, sys)
    
    def predict_single(self, features):
        """
        MAIN METHOD: Make Prediction on Single Record
        
        Orchestrates the complete prediction pipeline for a single transaction
        
        Args:
            features (pd.DataFrame): Single transaction features
            
        Returns:
            float: Predicted transaction amount
            
        Raises:
            CustomException: If any step fails
        """
        logging.info("\n")
        logging.info("╔" + "=" * 78 + "╗")
        logging.info("║" + " " * 78 + "║")
        logging.info("║" + "SINGLE RECORD PREDICTION".center(78) + "║")
        logging.info("║" + " " * 78 + "║")
        logging.info("╚" + "=" * 78 + "╝")
        
        try:
            # Step 2: Validate input
            self.validate_input(features)
            
            # Step 3: Transform features
            data_scaled = self.transform_features(features)
            
            # Step 4: Generate prediction
            predictions = self.generate_predictions(data_scaled)
            
            # Step 5: Format results
            result = self.format_results(predictions, features)
            
            logging.info("\n✓ Prediction completed successfully")
            
            return result
            
        except Exception as e:
            logging.error(f"\n✗ Prediction failed: {str(e)}")
            raise CustomException(e, sys)
    
    def predict_batch(self, features):
        """
        MAIN METHOD: Make Predictions on Batch of Records
        
        Orchestrates the complete prediction pipeline for multiple transactions
        
        Args:
            features (pd.DataFrame): Multiple transaction features
            
        Returns:
            pd.DataFrame: Results with original data and predictions
            
        Raises:
            CustomException: If any step fails
        """
        logging.info("\n")
        logging.info("╔" + "=" * 78 + "╗")
        logging.info("║" + " " * 78 + "║")
        logging.info("║" + "BATCH PREDICTIONS".center(78) + "║")
        logging.info("║" + " " * 78 + "║")
        logging.info("╚" + "=" * 78 + "╝")
        
        try:
            # Step 2: Validate input
            self.validate_input(features)
            
            # Step 3: Transform features
            data_scaled = self.transform_features(features)
            
            # Step 4: Generate predictions
            predictions = self.generate_predictions(data_scaled)
            
            # Step 5: Format results
            result = self.format_results(predictions, features)
            
            logging.info("\n✓ Batch predictions completed successfully")
            
            return result
            
        except Exception as e:
            logging.error(f"\n✗ Batch prediction failed: {str(e)}")
            raise CustomException(e, sys)
    
    def predict(self, features):
        """
        WRAPPER METHOD: Automatically route to batch or single prediction
        
        Args:
            features (pd.DataFrame): Transaction features to predict
            
        Returns:
            Union[float, np.ndarray, pd.DataFrame]: Predictions
        """
        if len(features) == 1:
            return self.predict_single(features)
        else:
            return self.predict_batch(features)


# ============================================================================
# CUSTOM DATA CLASS
# ============================================================================

class CustomData:
    """
    Custom Data Class for Retail Transaction Input
    
    Handles creation of input dataframes for making predictions on new transactions.
    Stores retail transaction parameters and converts them to proper format.
    
    Attributes:
        Quantity: Number of items in transaction
        UnitPrice: Price per item (£)
        CustomerID: Unique customer identifier
        Country: Customer's country
        InvoiceDate: Transaction date and time
        ItemsPerInvoice: Number of different items in invoice
        CustomerFrequency: Number of previous purchases by customer
    """
    
    def __init__(self,
                 Quantity: int,
                 UnitPrice: float,
                 CustomerID: int,
                 Country: str,
                 InvoiceDate: str,
                 ItemsPerInvoice: int = 1,
                 CustomerFrequency: int = 1):
        """
        Initialize CustomData with retail transaction parameters.
        
        Args:
            Quantity (int): Number of items in transaction (e.g., 6)
            UnitPrice (float): Price per item in GBP (e.g., 2.55)
            CustomerID (int): Unique customer ID (e.g., 17850)
            Country (str): Customer's country (e.g., 'United Kingdom')
            InvoiceDate (str): Transaction date/time (e.g., '12/1/2010 8:26')
            ItemsPerInvoice (int, optional): Number of items in invoice. Defaults to 1.
            CustomerFrequency (int, optional): Customer purchase frequency. Defaults to 1.
        """
        logging.info("\n" + "=" * 80)
        logging.info("INITIALIZING CUSTOM DATA FOR PREDICTION")
        logging.info("=" * 80)
        
        self.Quantity = Quantity
        self.UnitPrice = UnitPrice
        self.CustomerID = CustomerID
        self.Country = Country
        self.InvoiceDate = InvoiceDate
        self.ItemsPerInvoice = ItemsPerInvoice
        self.CustomerFrequency = CustomerFrequency
        
        logging.info(f"\n✓ Transaction Parameters Received:")
        logging.info(f"  Quantity: {self.Quantity} items")
        logging.info(f"  Unit Price: £{self.UnitPrice}")
        logging.info(f"  Customer ID: {self.CustomerID}")
        logging.info(f"  Country: {self.Country}")
        logging.info(f"  Invoice Date: {self.InvoiceDate}")
        logging.info(f"  Items Per Invoice: {self.ItemsPerInvoice}")
        logging.info(f"  Customer Frequency: {self.CustomerFrequency}")
    
    def extract_temporal_features(self):
        """
        Extract temporal features from InvoiceDate.
        
        Returns:
            dict: Dictionary with extracted temporal features
        """
        logging.info(f"\n🕐 Extracting Temporal Features...")
        
        try:
            date_obj = pd.to_datetime(self.InvoiceDate)
            
            temporal_features = {
                'Month': date_obj.month,
                'Day': date_obj.day,
                'Hour': date_obj.hour,
                'DayOfWeek': date_obj.dayofweek,
                'Quarter': date_obj.quarter
            }
            
            logging.info(f"  ✓ Temporal features extracted:")
            for key, value in temporal_features.items():
                logging.info(f"    - {key}: {value}")
            
            return temporal_features
            
        except Exception as e:
            logging.error(f"✗ Error extracting temporal features: {str(e)}")
            raise CustomException(e, sys)
    
    def create_quantity_category(self):
        """
        Create quantity category based on quantity value.
        
        Returns:
            str: Category ('Low', 'Medium', or 'High')
        """
        logging.info(f"\n📦 Creating Quantity Category...")
        
        # These thresholds should match training data
        if self.Quantity <= 10:
            category = 'Low'
        elif self.Quantity <= 50:
            category = 'Medium'
        else:
            category = 'High'
        
        logging.info(f"  ✓ Quantity: {self.Quantity} → Category: {category}")
        
        return category
    
    def create_is_high_price(self, median_price=2.5):
        """
        Determine if unit price is high.
        
        Args:
            median_price (float): Median price threshold (default from training data)
            
        Returns:
            int: 1 if high price, 0 otherwise
        """
        logging.info(f"\n💷 Determining Price Level...")
        
        is_high = 1 if self.UnitPrice > median_price else 0
        price_level = "High" if is_high == 1 else "Normal"
        
        logging.info(f"  Unit Price: £{self.UnitPrice}")
        logging.info(f"  Median Price: £{median_price}")
        logging.info(f"  ✓ Price Level: {price_level} (IsHighPrice: {is_high})")
        
        return is_high
    
    def get_data_as_data_frame(self):
        """
        MAIN METHOD: Convert all parameters to DataFrame for prediction
        
        Creates a properly formatted DataFrame with all required features
        for the model prediction pipeline.
        
        Returns:
            pd.DataFrame: Single-row dataframe with all features
            
        Raises:
            CustomException: If conversion fails
        """
        logging.info("\n" + "=" * 80)
        logging.info("CONVERTING TRANSACTION DATA TO DATAFRAME")
        logging.info("=" * 80)
        
        try:
            # Extract temporal features
            temporal_features = self.extract_temporal_features()
            
            # Create additional features
            QuantityCategory = self.create_quantity_category()
            IsHighPrice = self.create_is_high_price()
            
            # Create transaction amount (for reference)
            TotalAmount = self.Quantity * self.UnitPrice
            logging.info(f"\n💰 Calculating Transaction Amount:")
            logging.info(f"  {self.Quantity} × £{self.UnitPrice} = £{TotalAmount:.2f}")
            
            # Combine all features
            custom_data_input_dict = {
                'Quantity': [self.Quantity],
                'UnitPrice': [self.UnitPrice],
                'CustomerID': [self.CustomerID],
                'Month': [temporal_features['Month']],
                'Day': [temporal_features['Day']],
                'Hour': [temporal_features['Hour']],
                'DayOfWeek': [temporal_features['DayOfWeek']],
                'Quarter': [temporal_features['Quarter']],
                'TotalAmount': [TotalAmount],
                'IsHighPrice': [IsHighPrice],
                'ItemsPerInvoice': [self.ItemsPerInvoice],
                'CustomerFrequency': [self.CustomerFrequency],
                'Country': [self.Country],
                'QuantityCategory': [QuantityCategory]
            }
            
            # Create DataFrame
            df = pd.DataFrame(custom_data_input_dict)
            
            logging.info(f"\n✓ DataFrame Created Successfully:")
            logging.info(f"  Shape: {df.shape}")
            logging.info(f"  Columns: {df.columns.tolist()}")
            logging.info(f"\n  Data Preview:")
            for col in df.columns:
                logging.info(f"    {col}: {df[col].values[0]}")
            
            return df
            
        except Exception as e:
            logging.error(f"✗ Error converting to dataframe: {str(e)}")
            raise CustomException(e, sys)


# ============================================================================
# USAGE EXAMPLE (Documentation)
# ============================================================================

"""
USAGE EXAMPLE:

1. Single Transaction Prediction:
   
   from src.pipeline.predict_pipeline import CustomData, PredictPipeline
   
   # Create custom data for a transaction
   data = CustomData(
       Quantity=6,
       UnitPrice=2.55,
       CustomerID=17850,
       Country='United Kingdom',
       InvoiceDate='12/1/2010 8:26',
       ItemsPerInvoice=1,
       CustomerFrequency=5
   )
   
   # Convert to DataFrame
   input_df = data.get_data_as_data_frame()
   
   # Make prediction
   pipeline = PredictPipeline()
   result = pipeline.predict_single(input_df)
   print(result)

2. Batch Predictions:
   
   # Load multiple transactions from CSV
   batch_df = pd.read_csv('new_transactions.csv')
   
   # Make batch predictions
   pipeline = PredictPipeline()
   results = pipeline.predict_batch(batch_df)
   print(results)
"""

