"""
Enterprise-Level Flask Application for Retail Transaction Prediction
================================================================================
This module provides a production-ready Flask application for predicting
retail transaction outcomes using a trained machine learning model.

Features:
- RESTful API endpoints for predictions
- Comprehensive error handling and validation
- Structured logging and monitoring
- Configuration management
- Data schema validation
- Request/Response formatting
- Health checks and status endpoints
- CORS support for cross-origin requests

Data Attributes (from data.csv):
- InvoiceNo: Unique transaction identifier
- StockCode: Product identifier
- Description: Product description
- Quantity: Number of items purchased
- InvoiceDate: Transaction date and time
- UnitPrice: Price per unit
- CustomerID: Customer identifier
- Country: Customer country

Author: Srujan Kinjawadekar
Date: February 2026
Version: 1.0.0
================================================================================
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import json

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from functools import wraps

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging
from src.exceptions import CustomException
from src.utils import load_object


# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================

class Config:
    """Application configuration settings."""
    DEBUG = os.getenv('FLASK_DEBUG', False)
    TESTING = os.getenv('FLASK_TESTING', False)
    JSON_SORT_KEYS = False
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
    PREDICTION_TIMEOUT = 30  # seconds
    
    # Required fields for transaction data
    REQUIRED_FIELDS = {
        'invoice_no': str,
        'stock_code': str,
        'description': str,
        'quantity': (int, float),
        'invoice_date': str,
        'unit_price': (int, float),
        'customer_id': str,
        'country': str
    }


# ============================================================================
# FLASK APPLICATION FACTORY
# ============================================================================

def create_app(config_class=Config):
    """
    Application factory for creating Flask app instances.
    
    Args:
        config_class: Configuration class to use
        
    Returns:
        Flask application instance
    """
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config.from_object(config_class)
    
    # Enable CORS for cross-origin requests
    CORS(app)
    
    logging.info(f"Flask application created successfully")
    return app


# Initialize Flask application
application = create_app()
app = application


# ============================================================================
# DECORATOR FUNCTIONS
# ============================================================================

def log_request(f):
    """Decorator to log incoming requests."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        logging.info(f"Request: {request.method} {request.path}")
        logging.debug(f"Remote Address: {request.remote_addr}")
        if request.method in ['POST', 'PUT', 'PATCH']:
            logging.debug(f"Request Body: {request.get_json(silent=True)}")
        
        response = f(*args, **kwargs)
        return response
    return decorated_function


def handle_errors(f):
    """Decorator to handle exceptions gracefully."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logging.error(f"Validation Error: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Validation Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 400
        except CustomException as e:
            logging.error(f"Custom Exception: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Processing Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
        except Exception as e:
            logging.exception(f"Unexpected Error: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "timestamp": datetime.now().isoformat()
            }), 500
    return decorated_function


def validate_transaction_data(f):
    """Decorator to validate transaction data."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            if request.method == 'POST':
                data = request.get_json()
                if not data:
                    return jsonify({
                        "success": False,
                        "error": "Invalid Request",
                        "message": "Request body cannot be empty",
                        "timestamp": datetime.now().isoformat()
                    }), 400

                # Normalize various incoming key styles (InvoiceNo, InvoiceNo, invoice_no, etc.)
                def _normalize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
                    mapping = {
                        'invoiceno': 'invoice_no', 'invoice_no': 'invoice_no', 'invoiceNo': 'invoice_no',
                        'stockcode': 'stock_code', 'stock_code': 'stock_code', 'stockCode': 'stock_code',
                        'description': 'description', 'Description': 'description',
                        'quantity': 'quantity', 'Quantity': 'quantity',
                        'invoicedate': 'invoice_date', 'invoice_date': 'invoice_date', 'invoiceDate': 'invoice_date',
                        'unitprice': 'unit_price', 'unit_price': 'unit_price', 'unitPrice': 'unit_price',
                        'customerid': 'customer_id', 'customer_id': 'customer_id', 'customerId': 'customer_id', 'CustomerID': 'customer_id',
                        'country': 'country', 'Country': 'country'
                    }
                    out = {}
                    for k, v in d.items():
                        k_clean = k.replace(' ', '').replace('-', '').lower()
                        canonical = mapping.get(k, mapping.get(k_clean))
                        if canonical:
                            out[canonical] = v
                        else:
                            out[k] = v
                    return out

                data_norm = _normalize_keys(data)

                # Validate required fields (using canonical names)
                missing_fields = [field for field in Config.REQUIRED_FIELDS
                                  if field not in data_norm]
                if missing_fields:
                    return jsonify({
                        "success": False,
                        "error": "Missing Required Fields",
                        "message": f"Missing fields: {', '.join(missing_fields)}",
                        "timestamp": datetime.now().isoformat()
                    }), 400
        except Exception as e:
            logging.error(f"Validation decorator error: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Validation Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 400
        
        return f(*args, **kwargs)
    return decorated_function


# ============================================================================
# HEALTH CHECK & STATUS ENDPOINTS
# ============================================================================

@app.route('/api/v1/health', methods=['GET'])
@log_request
def health_check():
    """
    Health check endpoint to verify application status.
    
    Returns:
        JSON response with application health status
    """
    logging.info("Health check requested")
    try:
        return jsonify({
            "success": True,
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }), 200
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        return jsonify({
            "success": False,
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat()
        }), 503


@app.route('/api/v1/status', methods=['GET'])
@log_request
def status():
    """
    Get application status and configuration information.
    
    Returns:
        JSON response with detailed status information
    """
    logging.info("Status check requested")
    return jsonify({
        "success": True,
        "application": "Retail Transaction Prediction System",
        "version": "1.0.0",
        "environment": "production" if not app.config['DEBUG'] else "development",
        "timestamp": datetime.now().isoformat(),
        "supported_fields": list(Config.REQUIRED_FIELDS.keys())
    }), 200


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
@log_request
def index():
    """
    Home page endpoint.
    
    Returns:
        Rendered home page template
    """
    logging.info("Home page accessed")
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
@log_request
@handle_errors
def predict_web():
    """
    Web-based prediction endpoint with form interface.
    
    Supports GET (displays form) and POST (processes prediction).
    
    Expected Form Fields:
    - invoice_no: Unique transaction identifier
    - stock_code: Product code
    - description: Product description
    - quantity: Number of items
    - invoice_date: Transaction date (YYYY-MM-DD format)
    - unit_price: Price per unit
    - customer_id: Customer identifier
    - country: Customer country
    
    Returns:
        On GET: Rendered prediction form
        On POST: Rendered results page with prediction
    """
    if request.method == 'GET':
        logging.info("Prediction form requested")
        return render_template('Home.html')
    
    try:
        # Extract form data
        logging.info("Processing prediction request from web form")
        
        form_data = {
            'invoice_no': request.form.get('invoice_no', ''),
            'stock_code': request.form.get('stock_code', ''),
            'description': request.form.get('description', ''),
            'quantity': float(request.form.get('quantity', 0)),
            'invoice_date': request.form.get('invoice_date', ''),
            'unit_price': float(request.form.get('unit_price', 0)),
            'customer_id': request.form.get('customer_id', ''),
            'country': request.form.get('country', '')
        }
        
        logging.debug(f"Form data received: {form_data}")
        
        # Validate quantity and unit price
        if form_data['quantity'] <= 0:
            raise ValueError("Quantity must be greater than 0")
        if form_data['unit_price'] <= 0:
            raise ValueError("Unit Price must be greater than 0")
        
        # Create custom data object and generate prediction
        custom_data = CustomData(**form_data)
        pred_df = custom_data.get_data_as_data_frame()
        
        logging.debug(f"Data frame created: {pred_df.shape}")
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        logging.info(f"Prediction generated: {results[0]}")
        
        return render_template('Home.html', 
                             results=results[0],
                             prediction_successful=True,
                             transaction_data=form_data)
    
    except ValueError as e:
        logging.error(f"Validation error in web prediction: {str(e)}")
        return render_template('Home.html', 
                             error=str(e),
                             prediction_successful=False), 400
    except Exception as e:
        logging.exception(f"Unexpected error in web prediction: {str(e)}")
        return render_template('Home.html', 
                             error="An error occurred during prediction",
                             prediction_successful=False), 500


@app.route('/api/v1/predict', methods=['POST'])
@log_request
@handle_errors
@validate_transaction_data
def predict_api():
    """
    RESTful API endpoint for single transaction prediction.
    
    Expected JSON Body:
    {
        "invoice_no": "536365",
        "stock_code": "85123A",
        "description": "WHITE HANGING HEART T-LIGHT HOLDER",
        "quantity": 6,
        "invoice_date": "2010-12-01",
        "unit_price": 2.55,
        "customer_id": "17850",
        "country": "United Kingdom"
    }
    
    Returns:
        JSON response with prediction result
    """
    try:
        logging.info("API prediction request received")
        data = request.get_json()
        logging.debug(f"Request data: {data}")

        # Normalize keys to canonical internal names
        def _normalize(d: Dict[str, Any]) -> Dict[str, Any]:
            mapping = {
                'invoiceno': 'invoice_no', 'invoice_no': 'invoice_no', 'invoiceNo': 'invoice_no',
                'stockcode': 'stock_code', 'stock_code': 'stock_code', 'stockCode': 'stock_code',
                'description': 'description', 'Description': 'description',
                'quantity': 'quantity', 'Quantity': 'quantity',
                'invoicedate': 'invoice_date', 'invoice_date': 'invoice_date', 'invoiceDate': 'invoice_date',
                'unitprice': 'unit_price', 'unit_price': 'unit_price', 'unitPrice': 'unit_price',
                'customerid': 'customer_id', 'customer_id': 'customer_id', 'customerId': 'customer_id', 'CustomerID': 'customer_id',
                'country': 'country', 'Country': 'country'
            }
            out = {}
            for k, v in d.items():
                k_clean = k.replace(' ', '').replace('-', '').lower()
                canonical = mapping.get(k, mapping.get(k_clean))
                if canonical:
                    out[canonical] = v
                else:
                    out[k] = v
            return out

        data = _normalize(data)
        
        # Validate numeric fields
        if not isinstance(data.get('quantity'), (int, float)):
            raise ValueError("Quantity must be a number")
        if not isinstance(data.get('unit_price'), (int, float)):
            raise ValueError("Unit Price must be a number")
        
        if data['quantity'] <= 0:
            raise ValueError("Quantity must be greater than 0")
        if data['unit_price'] <= 0:
            raise ValueError("Unit Price must be greater than 0")
        
        # Create custom data object (map canonical names to CustomData constructor)
        custom_data = CustomData(
            Quantity=int(data['quantity']),
            UnitPrice=float(data['unit_price']),
            CustomerID=int(str(data['customer_id'])) if data.get('customer_id') not in [None, ''] else 0,
            Country=str(data['country']),
            InvoiceDate=str(data['invoice_date']),
            ItemsPerInvoice=int(data.get('items_per_invoice', 1)),
            CustomerFrequency=int(data.get('customer_frequency', 1))
        )
        
        # Generate prediction
        pred_df = custom_data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        prediction_result = predict_pipeline.predict(pred_df)
        
        logging.info(f"Prediction successful: {prediction_result[0]}")
        
        response = {
            "success": True,
            "prediction": float(prediction_result[0]),
            "transaction": data,
            "timestamp": datetime.now().isoformat(),
            "model_version": "1.0.0"
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logging.exception(f"Error in API prediction: {str(e)}")
        raise


@app.route('/api/v1/predict/batch', methods=['POST'])
@log_request
@handle_errors
def predict_batch():
    """
    RESTful API endpoint for batch transaction predictions.
    
    Expected JSON Body:
    {
        "transactions": [
            {
                "invoice_no": "536365",
                "stock_code": "85123A",
                "description": "WHITE HANGING HEART T-LIGHT HOLDER",
                "quantity": 6,
                "invoice_date": "2010-12-01",
                "unit_price": 2.55,
                "customer_id": "17850",
                "country": "United Kingdom"
            },
            ...more transactions...
        ]
    }
    
    Returns:
        JSON response with batch predictions
    """
    try:
        logging.info("Batch prediction request received")
        data = request.get_json()
        
        if not data or 'transactions' not in data:
            return jsonify({
                "success": False,
                "error": "Invalid Request",
                "message": "'transactions' field is required",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        transactions = data['transactions']
        if not isinstance(transactions, list):
            return jsonify({
                "success": False,
                "error": "Invalid Request",
                "message": "'transactions' must be a list",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        logging.info(f"Processing batch of {len(transactions)} transactions")
        
        predictions = []
        predict_pipeline = PredictPipeline()
        
        for idx, transaction in enumerate(transactions):
            try:
                # Normalize keys to canonical internal names
                def _normalize(d: Dict[str, Any]) -> Dict[str, Any]:
                    mapping = {
                        'invoiceno': 'invoice_no', 'invoice_no': 'invoice_no', 'invoiceNo': 'invoice_no',
                        'stockcode': 'stock_code', 'stock_code': 'stock_code', 'stockCode': 'stock_code',
                        'description': 'description', 'Description': 'description',
                        'quantity': 'quantity', 'Quantity': 'quantity',
                        'invoicedate': 'invoice_date', 'invoice_date': 'invoice_date', 'invoiceDate': 'invoice_date',
                        'unitprice': 'unit_price', 'unit_price': 'unit_price', 'unitPrice': 'unit_price',
                        'customerid': 'customer_id', 'customer_id': 'customer_id', 'customerId': 'customer_id', 'CustomerID': 'customer_id',
                        'country': 'country', 'Country': 'country'
                    }
                    out = {}
                    for k, v in d.items():
                        k_clean = k.replace(' ', '').replace('-', '').lower()
                        canonical = mapping.get(k, mapping.get(k_clean))
                        if canonical:
                            out[canonical] = v
                        else:
                            out[k] = v
                    return out

                trans_norm = _normalize(transaction)
                
                # Validate numeric fields
                if not isinstance(trans_norm.get('quantity'), (int, float)):
                    raise ValueError("Quantity must be a number")
                if not isinstance(trans_norm.get('unit_price'), (int, float)):
                    raise ValueError("Unit Price must be a number")
                
                if trans_norm['quantity'] <= 0:
                    raise ValueError("Quantity must be greater than 0")
                if trans_norm['unit_price'] <= 0:
                    raise ValueError("Unit Price must be greater than 0")
                
                # Create custom data object with PascalCase parameters
                custom_data = CustomData(
                    Quantity=int(trans_norm['quantity']),
                    UnitPrice=float(trans_norm['unit_price']),
                    CustomerID=int(str(trans_norm['customer_id'])) if trans_norm.get('customer_id') not in [None, ''] else 0,
                    Country=str(trans_norm['country']),
                    InvoiceDate=str(trans_norm['invoice_date']),
                    ItemsPerInvoice=int(trans_norm.get('items_per_invoice', 1)),
                    CustomerFrequency=int(trans_norm.get('customer_frequency', 1))
                )
                
                pred_df = custom_data.get_data_as_data_frame()
                prediction = predict_pipeline.predict(pred_df)
                
                predictions.append({
                    "transaction_index": idx,
                    "invoice_no": trans_norm.get('invoice_no'),
                    "prediction": float(prediction[0]),
                    "status": "success"
                })
                
            except Exception as e:
                logging.warning(f"Error predicting transaction {idx}: {str(e)}")
                predictions.append({
                    "transaction_index": idx,
                    "invoice_no": transaction.get('invoice_no'),
                    "status": "failed",
                    "error": str(e)
                })
        
        logging.info(f"Batch prediction completed: {len(predictions)} results")
        
        return jsonify({
            "success": True,
            "total_transactions": len(transactions),
            "successful_predictions": sum(1 for p in predictions if p['status'] == 'success'),
            "failed_predictions": sum(1 for p in predictions if p['status'] == 'failed'),
            "predictions": predictions,
            "timestamp": datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logging.exception(f"Error in batch prediction: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Batch Processing Error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 Not Found errors."""
    logging.warning(f"404 Not Found: {request.path}")
    return jsonify({
        "success": False,
        "error": "Not Found",
        "message": f"Endpoint {request.path} does not exist",
        "timestamp": datetime.now().isoformat()
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 Method Not Allowed errors."""
    logging.warning(f"405 Method Not Allowed: {request.method} {request.path}")
    return jsonify({
        "success": False,
        "error": "Method Not Allowed",
        "message": f"Method {request.method} is not allowed for {request.path}",
        "timestamp": datetime.now().isoformat()
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 Internal Server errors."""
    logging.error(f"500 Internal Server Error: {str(error)}")
    return jsonify({
        "success": False,
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.now().isoformat()
    }), 500


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logging.info("=" * 80)
    logging.info("Starting Retail Transaction Prediction System")
    logging.info(f"Debug Mode: {app.config['DEBUG']}")
    logging.info(f"Testing Mode: {app.config['TESTING']}")
    logging.info("=" * 80)
    
    # Run Flask application
    app.run(
        host="0.0.0.0",
        port=int(os.getenv('PORT', 5000)),
        debug=app.config['DEBUG'],
        use_reloader=app.config['DEBUG'],
        threaded=True
    )        


