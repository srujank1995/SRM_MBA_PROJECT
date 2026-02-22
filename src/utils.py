"""
Utility Functions Module
================================================================================
This module provides utility functions for the retail transaction prediction
system, including:
- Object serialization/deserialization
- Model evaluation and hyperparameter tuning
- Metrics calculation and reporting
- Data validation utilities

Author: Srujan Vijay Kinjawadekar
Date: February 2026
Version: 1.0.0
================================================================================
"""

import os
import sys
from typing import Dict, Any, Tuple, List, Union, Optional
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pickle
import dill
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, root_mean_squared_error
)
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.exceptions import CustomException
from src.logger import logging

# ============================================================================
# OBJECT SERIALIZATION UTILITIES
# ============================================================================

def save_object(file_path: str, obj: Any, use_dill: bool = False) -> None:
    """
    Save a Python object to disk using pickle or dill serialization.
    
    This function serializes and saves any Python object (models, preprocessors,
    etc.) to a binary file. It automatically creates parent directories if they
    don't exist.
    
    Args:
        file_path (str): Absolute or relative path where object should be saved.
                         Example: 'artifacts/model.pkl'
        obj (Any): Python object to serialize and save
        use_dill (bool, optional): If True, use dill serialization instead of pickle.
                                   Dill can handle more complex Python objects.
                                   Defaults to False.
    
    Returns:
        None
    
    Raises:
        CustomException: If file cannot be created or object cannot be serialized
        
    Example:
        >>> from sklearn.preprocessing import StandardScaler
        >>> scaler = StandardScaler()
        >>> save_object('artifacts/scaler.pkl', scaler)
        >>> # Object is now saved to disk
    
    Note:
        - Creates parent directories automatically
        - Overwrites existing files at the path
        - For large objects, dill serialization may be more reliable
    """
    logging.info(f"\n💾 Saving object to: {file_path}")
    
    try:
        # Extract directory path
        dir_path = os.path.dirname(file_path)
        
        # Create directory if it doesn't exist
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"  ✓ Directory created/verified: {dir_path}")
        
        # Choose serializer
        serializer = dill if use_dill else pickle
        serializer_name = "dill" if use_dill else "pickle"
        
        # Save object
        with open(file_path, 'wb') as file_obj:
            serializer.dump(obj, file_obj)
        
        # Get file size
        file_size = os.path.getsize(file_path) / 1024  # Size in KB
        
        logging.info(f"  ✓ Object saved successfully")
        logging.info(f"  - Serializer: {serializer_name}")
        logging.info(f"  - Object type: {type(obj).__name__}")
        logging.info(f"  - File size: {file_size:.2f} KB")
        
    except FileNotFoundError as e:
        logging.error(f"✗ File path not found: {file_path}")
        raise CustomException(f"Cannot save object - invalid file path: {file_path}", sys)
    except PermissionError as e:
        logging.error(f"✗ Permission denied: {file_path}")
        raise CustomException(f"Permission denied - cannot write to {file_path}", sys)
    except Exception as e:
        logging.error(f"✗ Error saving object: {str(e)}")
        raise CustomException(f"Failed to save object: {str(e)}", sys)


def load_object(file_path: str, use_dill: bool = False) -> Any:
    """
    Load a Python object from disk that was previously saved with save_object().
    
    This function deserializes and loads a Python object from a binary file.
    It automatically selects the appropriate deserializer based on file format.
    
    Args:
        file_path (str): Path to the saved object file
                         Example: 'artifacts/model.pkl'
        use_dill (bool, optional): If True, use dill deserialization.
                                   Defaults to False (uses pickle).
    
    Returns:
        Any: The deserialized Python object
    
    Raises:
        CustomException: If file doesn't exist, cannot be read, or deserialization fails
        
    Example:
        >>> model = load_object('artifacts/model.pkl')
        >>> print(type(model))  # <class 'sklearn.ensemble.RandomForestRegressor'>
    
    Note:
        - File must have been saved with save_object()
        - Object must be deserializable
        - Large objects may take time to load
    """
    logging.info(f"\n📂 Loading object from: {file_path}")
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logging.error(f"✗ File not found: {file_path}")
            raise FileNotFoundError(f"Object file not found: {file_path}")
        
        # Get file info
        file_size = os.path.getsize(file_path) / 1024
        logging.info(f"  File size: {file_size:.2f} KB")
        
        # Choose deserializer
        deserializer = dill if use_dill else pickle
        deserializer_name = "dill" if use_dill else "pickle"
        
        # Load object
        with open(file_path, 'rb') as file_obj:
            obj = deserializer.load(file_obj)
        
        logging.info(f"  ✓ Object loaded successfully")
        logging.info(f"  - Deserializer: {deserializer_name}")
        logging.info(f"  - Object type: {type(obj).__name__}")
        
        return obj
        
    except FileNotFoundError as e:
        raise CustomException(str(e), sys)
    except PermissionError:
        logging.error(f"✗ Permission denied: {file_path}")
        raise CustomException(f"Permission denied - cannot read {file_path}", sys)
    except Exception as e:
        logging.error(f"✗ Error loading object: {str(e)}")
        raise CustomException(f"Failed to load object: {str(e)}", sys)


# ============================================================================
# MODEL EVALUATION UTILITIES
# ============================================================================

def calculate_regression_metrics(y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics for model evaluation.
    
    Computes multiple regression performance metrics to provide a holistic
    view of model performance. Metrics include R², RMSE, MAE, MAPE, and MSE.
    
    Args:
        y_true (np.ndarray): Actual/true values
        y_pred (np.ndarray): Predicted values from model
    
    Returns:
        Dict[str, float]: Dictionary containing:
            - 'r2': R² score (coefficient of determination)
            - 'rmse': Root Mean Squared Error
            - 'mae': Mean Absolute Error
            - 'mape': Mean Absolute Percentage Error
            - 'mse': Mean Squared Error
    
    Raises:
        CustomException: If metrics calculation fails
        
    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
        >>> y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        >>> metrics = calculate_regression_metrics(y_true, y_pred)
        >>> print(f"R² Score: {metrics['r2']:.4f}")
    
    Note:
        - All arrays must have the same length
        - MAPE may be undefined if y_true contains zeros
    """
    logging.info(f"\n📊 Calculating Regression Metrics...")
    
    try:
        # Ensure arrays are numpy arrays
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Validate shapes
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        if len(y_true) == 0:
            raise ValueError("Empty arrays provided")
        
        # Calculate metrics
        metrics = {
            'r2': float(r2_score(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
        }
        
        # Calculate MAPE (with safety check for division by zero)
        try:
            metrics['mape'] = float(mean_absolute_percentage_error(y_true, y_pred))
        except ValueError:
            logging.warning("  ⚠️  MAPE calculation skipped (y_true contains zero values)")
            metrics['mape'] = np.inf
        
        logging.info(f"  ✓ Metrics calculated:")
        for metric_name, metric_value in metrics.items():
            if metric_value != np.inf:
                logging.info(f"    - {metric_name.upper()}: {metric_value:.4f}")
            else:
                logging.info(f"    - {metric_name.upper()}: undefined")
        
        return metrics
        
    except Exception as e:
        logging.error(f"✗ Error calculating metrics: {str(e)}")
        raise CustomException(f"Metrics calculation failed: {str(e)}", sys)


def evaluate_models(X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    models: Dict[str, Any],
                    param: Dict[str, Dict[str, List]],
                    cv: int = 5,
                    verbose: int = 1) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple machine learning models with hyperparameter tuning.
    
    This function performs GridSearchCV for each model to find optimal
    hyperparameters, then evaluates performance on training and testing sets.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target values
        X_test (np.ndarray): Testing features
        y_test (np.ndarray): Testing target values
        models (Dict[str, Any]): Dictionary of models to evaluate
                                 Example: {'RandomForest': RandomForestRegressor()}
        param (Dict[str, Dict[str, List]]): Hyperparameter grids for each model
                                            Example: {'RandomForest': {'n_estimators': [10, 50, 100]}}
        cv (int, optional): Number of cross-validation folds. Defaults to 5.
        verbose (int, optional): Verbosity level (0=silent, 1=normal). Defaults to 1.
    
    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary with model names as keys and
                                     evaluation metrics as values. Example:
                                     {
                                         'RandomForest': {
                                             'best_params': {...},
                                             'cv_score': 0.85,
                                             'train_r2': 0.88,
                                             'test_r2': 0.82,
                                             'train_rmse': 150.23,
                                             'test_rmse': 165.45,
                                             ...
                                         }
                                     }
    
    Raises:
        CustomException: If model evaluation fails
        
    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> models = {'RF': RandomForestRegressor()}
        >>> params = {'RF': {'n_estimators': [10, 50, 100]}}
        >>> results = evaluate_models(X_train, y_train, X_test, y_test, models, params)
        >>> print(results['RF']['test_r2'])
    
    Note:
        - GridSearchCV may be time-consuming for large datasets
        - Use cv=3 for faster iteration during development
    """
    logging.info("\n" + "=" * 80)
    logging.info("EVALUATING MACHINE LEARNING MODELS")
    logging.info("=" * 80)
    
    try:
        report = {}
        model_names = list(models.keys())
        total_models = len(model_names)
        
        logging.info(f"\n📊 Found {total_models} models to evaluate")
        
        for idx, model_name in enumerate(model_names, 1):
            logging.info(f"\n[{idx}/{total_models}] Evaluating: {model_name}")
            logging.info("-" * 80)
            
            try:
                # Get model and parameters
                model = models[model_name]
                params = param.get(model_name, {})
                
                logging.info(f"  Model type: {type(model).__name__}")
                logging.info(f"  Hyperparameter grid size: {len(params)}")
                
                # ============================================================
                # HYPERPARAMETER TUNING WITH GRIDSEARCHCV
                # ============================================================
                logging.info(f"\n  🔍 Performing GridSearchCV (cv={cv})...")
                
                if params:
                    gs = GridSearchCV(
                        estimator=model,
                        param_grid=params,
                        cv=cv,
                        n_jobs=-1,
                        verbose=verbose,
                        scoring='r2'
                    )
                    
                    gs.fit(X_train, y_train)
                    
                    best_params = gs.best_params_
                    best_cv_score = gs.best_score_
                    
                    logging.info(f"    ✓ Best CV score: {best_cv_score:.4f}")
                    logging.info(f"    ✓ Best parameters: {best_params}")
                    
                    # Set best parameters
                    model.set_params(**best_params)
                else:
                    logging.warning(f"    ⚠️  No hyperparameters provided")
                    best_params = {}
                    best_cv_score = None
                
                # ============================================================
                # MODEL TRAINING WITH BEST PARAMETERS
                # ============================================================
                logging.info(f"\n  🏋️  Training model with best parameters...")
                model.fit(X_train, y_train)
                logging.info(f"    ✓ Model training completed")
                
                # ============================================================
                # MODEL PREDICTION
                # ============================================================
                logging.info(f"\n  🎯 Making predictions...")
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                logging.info(f"    ✓ Predictions completed")
                
                # ============================================================
                # PERFORMANCE EVALUATION
                # ============================================================
                logging.info(f"\n  📈 Calculating performance metrics...")
                
                # Training metrics
                train_metrics = calculate_regression_metrics(y_train, y_train_pred)
                
                # Testing metrics
                test_metrics = calculate_regression_metrics(y_test, y_test_pred)
                
                # Compile results
                report[model_name] = {
                    'best_params': best_params,
                    'cv_score': best_cv_score,
                    'train_r2': train_metrics['r2'],
                    'train_rmse': train_metrics['rmse'],
                    'train_mae': train_metrics['mae'],
                    'train_mape': train_metrics['mape'],
                    'test_r2': test_metrics['r2'],
                    'test_rmse': test_metrics['rmse'],
                    'test_mae': test_metrics['mae'],
                    'test_mape': test_metrics['mape'],
                }
                
                logging.info(f"\n  📊 Performance Summary:")
                logging.info(f"    Training R²: {train_metrics['r2']:.4f}")
                logging.info(f"    Testing R²:  {test_metrics['r2']:.4f}")
                logging.info(f"    Training RMSE: {train_metrics['rmse']:.4f}")
                logging.info(f"    Testing RMSE:  {test_metrics['rmse']:.4f}")
                
            except Exception as e:
                logging.error(f"  ✗ Error evaluating {model_name}: {str(e)}")
                report[model_name] = {'error': str(e)}
        
        # ====================================================================
        # SUMMARY REPORT
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("EVALUATION SUMMARY")
        logging.info("=" * 80)
        
        # Find best model
        best_model = max(
            (m for m in report.keys() if 'error' not in report[m]),
            key=lambda x: report[x]['test_r2'],
            default=None
        )
        
        if best_model:
            logging.info(f"\n🏆 Best Model: {best_model}")
            logging.info(f"   Test R² Score: {report[best_model]['test_r2']:.4f}")
        
        return report
        
    except Exception as e:
        logging.error(f"✗ Model evaluation failed: {str(e)}")
        raise CustomException(f"Model evaluation error: {str(e)}", sys)


# ============================================================================
# DATA VALIDATION UTILITIES
# ============================================================================

def validate_data(df: pd.DataFrame, 
                  required_columns: List[str] ) -> Tuple[bool, str]:
    """
    Validate a DataFrame for common data quality issues.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (List[str], optional): List of required column names
    
    Returns:
        Tuple[bool, str]: (is_valid, message)
        
    Example:
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> is_valid, msg = validate_data(df, ['A', 'B'])
        >>> print(f"Valid: {is_valid}")
    """
    logging.info(f"\n✅ Validating DataFrame...")
    
    try:
        # Check if empty
        if df.empty:
            return False, "DataFrame is empty"
        
        logging.info(f"  Shape: {df.shape}")
        
        # Check required columns
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                return False, f"Missing columns: {missing_cols}"
        
        logging.info(f"  ✓ Validation passed")
        return True, "DataFrame is valid"
        
    except Exception as e:
        logging.error(f"✗ Validation error: {str(e)}")
        return False, f"Validation error: {str(e)}"
