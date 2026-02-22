"""
Custom Exception Classes Module
================================================================================
This module defines custom exception classes for the retail transaction
prediction system. It provides specialized exceptions for different error
scenarios with detailed error context and stack traces.

Features:
- Hierarchical exception structure
- Detailed error messages with file names and line numbers
- Stack trace extraction and formatting
- Additional error context (severity levels, error codes)
- Integration with logging system
- Error aggregation and reporting utilities

Author: Srujan Vijay Kinjawadekar
Date: February 2026
Version: 1.0.0
================================================================================
"""

import sys
import traceback
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

from src.logger import logging

# ============================================================================
# ERROR FORMATTING UTILITIES
# ============================================================================

def extract_error_context(error_detail: Any) -> Dict[str, Any]:
    """
    Extract detailed error context from exception info.
    
    This function extracts comprehensive information about where an error
    occurred, including file name, function name, line number, and the full
    stack trace.
    
    Args:
        error_detail: The sys.exc_info() tuple containing exception information
        
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'file_name': Name of the file where error occurred
            - 'function_name': Name of the function where error occurred
            - 'line_number': Line number where error occurred
            - 'stack_trace': Full stack trace as string
            - 'error_type': Type of exception
            - 'timestamp': When error occurred
            
    Example:
        >>> try:
        >>>     1 / 0
        >>> except:
        >>>     context = extract_error_context(sys.exc_info())
        >>>     print(context['file_name'])
    """
    try:
        exc_type, exc_value, exc_tb = error_detail
        
        # Extract traceback information
        file_name = exc_tb.tb_frame.f_code.co_filename
        function_name = exc_tb.tb_frame.f_code.co_name
        line_number = exc_tb.tb_lineno
        
        # Format stack trace
        stack_trace = ''.join(traceback.format_tb(exc_tb))
        
        # Get exception type name
        error_type = exc_type.__name__ if exc_type else 'Unknown'
        
        return {
            'file_name': file_name,
            'function_name': function_name,
            'line_number': line_number,
            'stack_trace': stack_trace,
            'error_type': error_type,
            'timestamp': datetime.now().isoformat(),
        }
    except Exception as e:
        logging.warning(f"Could not extract error context: {str(e)}")
        return {
            'file_name': 'unknown',
            'function_name': 'unknown',
            'line_number': 0,
            'stack_trace': '',
            'error_type': 'Unknown',
            'timestamp': datetime.now().isoformat(),
        }


def format_error_message(error_message: str, error_detail: Any) -> str:
    """
    Format error message with detailed context information.
    
    Creates a comprehensive error message that includes:
    - File name and line number
    - Function name
    - Original error message
    - Stack trace
    
    Args:
        error_message (str): The error message to format
        error_detail: sys.exc_info() tuple with exception details
        
    Returns:
        str: Formatted error message with full context
        
    Example:
        >>> try:
        >>>     risky_operation()
        >>> except Exception as e:
        >>>     formatted = format_error_message(str(e), sys.exc_info())
        >>>     print(formatted)
    """
    context = extract_error_context(error_detail)
    
    formatted_message = (
        f"\n{'═' * 80}\n"
        f"ERROR DETAILS\n"
        f"{'═' * 80}\n"
        f"Error Type: {context['error_type']}\n"
        f"File: {context['file_name']}\n"
        f"Function: {context['function_name']}\n"
        f"Line: {context['line_number']}\n"
        f"Timestamp: {context['timestamp']}\n"
        f"{'─' * 80}\n"
        f"Error Message:\n{error_message}\n"
        f"{'─' * 80}\n"
        f"Stack Trace:\n{context['stack_trace']}"
        f"{'═' * 80}\n"
    )
    
    return formatted_message


# ============================================================================
# CUSTOM EXCEPTION CLASSES (Hierarchical Structure)
# ============================================================================

class CustomException(Exception):
    """
    Base Custom Exception Class
    
    Serves as the base exception class for all custom exceptions in the system.
    Captures detailed error context including file name, function name, and
    line number where the error occurred.
    
    Attributes:
        error_message (str): Formatted error message with context
        error_context (dict): Detailed error context information
        
    Example:
        >>> try:
        >>>     problematic_code()
        >>> except Exception as e:
        >>>     raise CustomException(str(e), sys.exc_info())
    """
    
    def __init__(self, error_message: str, error_detail: Any):
        """
        Initialize CustomException with error context.
        
        Args:
            error_message (str): The error message
            error_detail: sys.exc_info() tuple with exception details
        """
        super().__init__(error_message)
        
        self.error_context = extract_error_context(error_detail)
        self.error_message = format_error_message(
            error_message,
            error_detail
        )
        self.timestamp = datetime.now()
        
        # Log the error
        logging.error(self.error_message)
    
    def __str__(self) -> str:
        """Return formatted error message."""
        return self.error_message
    
    def __repr__(self) -> str:
        """Return representation of the exception."""
        return (
            f"CustomException("
            f"file={self.error_context['file_name']}, "
            f"line={self.error_context['line_number']}, "
            f"type={self.error_context['error_type']}"
            f")"
        )
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get detailed error context.
        
        Returns:
            Dict[str, Any]: Error context information
        """
        return self.error_context


class DataIngestionException(CustomException):
    """
    Exception for Data Ingestion errors.
    
    Raised when errors occur during data loading, validation, or splitting.
    
    Common causes:
    - File not found
    - Invalid data format
    - Insufficient permissions
    - Data validation failures
    
    Example:
        >>> try:
        >>>     load_data(path)
        >>> except FileNotFoundError as e:
        >>>     raise DataIngestionException(str(e), sys.exc_info())
    """
    pass


class DataTransformationException(CustomException):
    """
    Exception for Data Transformation errors.
    
    Raised when errors occur during feature engineering, encoding, or scaling.
    
    Common causes:
    - Invalid feature types
    - Missing required columns
    - Preprocessing pipeline errors
    - Encoding/scaling failures
    
    Example:
        >>> try:
        >>>     transform_features(df)
        >>> except ValueError as e:
        >>>     raise DataTransformationException(str(e), sys.exc_info())
    """
    pass


class ModelTrainingException(CustomException):
    """
    Exception for Model Training errors.
    
    Raised when errors occur during model training or hyperparameter tuning.
    
    Common causes:
    - Invalid model parameters
    - GridSearchCV failures
    - Insufficient training data
    - Out of memory errors
    - Incompatible algorithm with data
    
    Example:
        >>> try:
        >>>     train_model(X, y, params)
        >>> except Exception as e:
        >>>     raise ModelTrainingException(str(e), sys.exc_info())
    """
    pass


class ModelEvaluationException(CustomException):
    """
    Exception for Model Evaluation errors.
    
    Raised when errors occur during prediction or metric calculation.
    
    Common causes:
    - Model not trained
    - Incompatible input shape
    - Metric calculation failures
    - Data type mismatches
    
    Example:
        >>> try:
        >>>     calculate_metrics(y_true, y_pred)
        >>> except Exception as e:
        >>>     raise ModelEvaluationException(str(e), sys.exc_info())
    """
    pass


class PredictionException(CustomException):
    """
    Exception for Prediction Pipeline errors.
    
    Raised when errors occur during the prediction process.
    
    Common causes:
    - Model artifacts not found
    - Invalid input features
    - Model serialization issues
    - Preprocessor compatibility problems
    
    Example:
        >>> try:
        >>>     make_prediction(data)
        >>> except Exception as e:
        >>>     raise PredictionException(str(e), sys.exc_info())
    """
    pass


class PreprocessorException(CustomException):
    """
    Exception for Preprocessor errors.
    
    Raised when errors occur in the preprocessing pipeline.
    
    Common causes:
    - Serialization failures
    - Invalid fitted state
    - Incompatible data shapes
    - Unknown categories in encoding
    
    Example:
        >>> try:
        >>>     preprocessor.fit_transform(X)
        >>> except Exception as e:
        >>>     raise PreprocessorException(str(e), sys.exc_info())
    """
    pass


class ArtifactException(CustomException):
    """
    Exception for Artifact handling errors.
    
    Raised when errors occur loading/saving model or preprocessor artifacts.
    
    Common causes:
    - File not found
    - Permission denied
    - Corrupted pickle files
    - Python version compatibility issues
    - Insufficient disk space
    
    Example:
        >>> try:
        >>>     save_object(path, obj)
        >>> except Exception as e:
        >>>     raise ArtifactException(str(e), sys.exc_info())
    """
    pass


class ValidationException(CustomException):
    """
    Exception for Data Validation errors.
    
    Raised when data validation fails.
    
    Common causes:
    - Missing required columns
    - Data type mismatches
    - Invalid value ranges
    - Schema violations
    
    Example:
        >>> try:
        >>>     validate_input(df)
        >>> except ValueError as e:
        >>>     raise ValidationException(str(e), sys.exc_info())
    """
    pass


class ConfigurationException(CustomException):
    """
    Exception for Configuration errors.
    
    Raised when configuration is invalid or incomplete.
    
    Common causes:
    - Missing configuration files
    - Invalid parameter values
    - Incompatible settings
    - Missing environment variables
    
    Example:
        >>> try:
        >>>     load_config()
        >>> except Exception as e:
        >>>     raise ConfigurationException(str(e), sys.exc_info())
    """
    pass


# ============================================================================
# ERROR REPORTING UTILITIES
# ============================================================================

class ErrorReporter:
    """
    Utility class for comprehensive error reporting and analysis.
    
    Provides methods to track, aggregate, and report errors from the system.
    """
    
    def __init__(self):
        """Initialize error reporter."""
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
    
    def report_error(self, exception: CustomException, 
                     severity: str = 'ERROR',
                     additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Report an error to the system.
        
        Args:
            exception (CustomException): The exception to report
            severity (str): Severity level ('INFO', 'WARNING', 'ERROR', 'CRITICAL')
            additional_info (Dict, optional): Additional contextual information
        """
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'exception_type': type(exception).__name__,
            'message': str(exception),
            'context': exception.get_context(),
            'additional_info': additional_info or {}
        }
        
        self.errors.append(error_record)
        
        # Update error counts
        error_type = type(exception).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log based on severity
        if severity == 'CRITICAL':
            logging.critical(f"CRITICAL ERROR: {error_record['message']}")
        elif severity == 'ERROR':
            logging.error(f"ERROR: {error_record['message']}")
        elif severity == 'WARNING':
            logging.warning(f"WARNING: {error_record['message']}")
        else:
            logging.info(f"INFO: {error_record['message']}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of all reported errors.
        
        Returns:
            Dict[str, Any]: Summary statistics
        """
        return {
            'total_errors': len(self.errors),
            'error_counts': self.error_counts,
            'first_error': self.errors[0]['timestamp'] if self.errors else None,
            'last_error': self.errors[-1]['timestamp'] if self.errors else None,
        }
    
    def clear_errors(self) -> None:
        """Clear error history."""
        self.errors = []
        self.error_counts = {}


# ============================================================================
# GLOBAL ERROR REPORTER INSTANCE
# ============================================================================

error_reporter = ErrorReporter()


# ============================================================================
# USAGE EXAMPLES AND DOCUMENTATION
# ============================================================================

"""
USAGE EXAMPLES:

1. Basic Error Handling:
   
   from src.exceptions import CustomException
   
   try:
       result = risky_operation()
   except Exception as e:
       raise CustomException(str(e), sys.exc_info())

2. Specific Exception Types:
   
   from src.exceptions import DataIngestionException, DataTransformationException
   
   try:
       data = load_data(path)
   except FileNotFoundError as e:
       raise DataIngestionException(f"Cannot load data from {path}: {str(e)}", sys.exc_info())
   
   try:
       transformed = transform_data(data)
   except ValueError as e:
       raise DataTransformationException(f"Invalid features: {str(e)}", sys.exc_info())

3. Error Reporting:
   
   from src.exceptions import error_reporter, PredictionException
   
   try:
       predictions = model.predict(X)
   except Exception as e:
       exc = PredictionException(str(e), sys.exc_info())
       error_reporter.report_error(exc, severity='ERROR', 
                                   additional_info={'shape': X.shape})
   
   # Get error summary
   summary = error_reporter.get_error_summary()
   print(f"Total errors: {summary['total_errors']}")

4. Error Context Access:
   
   try:
       process_data()
   except Exception as e:
       exc = CustomException(str(e), sys.exc_info())
       context = exc.get_context()
       print(f"Error in file: {context['file_name']}")
       print(f"Line number: {context['line_number']}")
       print(f"Function: {context['function_name']}")


EXCEPTION HIERARCHY:

CustomException (Base)
├── DataIngestionException
├── DataTransformationException
├── ModelTrainingException
├── ModelEvaluationException
├── PredictionException
├── PreprocessorException
├── ArtifactException
├── ValidationException
└── ConfigurationException


BEST PRACTICES:

1. Always pass sys.exc_info() to exception constructor
2. Use specific exception types for different errors
3. Include meaningful error messages
4. Use error_reporter for critical errors
5. Provide additional context when available
6. Log error details for debugging
"""