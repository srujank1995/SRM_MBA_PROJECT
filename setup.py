"""
Setup Configuration Module
================================================================================
This module contains the setup configuration for the AI-Based Retail 
Transaction Prediction System package.

The setup.py file is used to package and distribute the project. It defines
project metadata, dependencies, entry points, and other configuration details
required for installation via pip.

Author: Srujan Vijay Kinjawadekar
Date: February 2026
Version: 1.0.0
================================================================================
"""

from setuptools import find_packages, setup
from typing import List
import os

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Special flag for editable installations
EDITABLE_INSTALL_FLAG = '-e .'

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_requirements(file_path: str) -> List[str]:
    """
    Parse and retrieve project dependencies from requirements.txt file.
    
    This function reads the requirements.txt file and returns a list of
    package names with their version specifications. It automatically filters
    out the editable install flag (-e .) which is used for development
    installations.
    
    Args:
        file_path (str): Absolute or relative path to requirements.txt file
        
    Returns:
        List[str]: List of requirement strings (e.g., ['pandas>=1.3.0', 'numpy==1.21.0'])
        
    Raises:
        FileNotFoundError: If requirements.txt file is not found
        IOError: If file cannot be read
        
    Example:
        >>> requirements = get_requirements('requirements.txt')
        >>> print(requirements)
        ['pandas>=1.3.0', 'numpy==1.21.0', 'scikit-learn>=0.24.0']
    """
    requirements = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file_obj:
            requirements = file_obj.readlines()
            
            # Clean up requirement strings
            requirements = [
                req.strip()
                for req in requirements
                if req.strip() and not req.startswith('#')  # Remove empty lines and comments
            ]
            
            # Remove editable install flag if present
            if EDITABLE_INSTALL_FLAG in requirements:
                requirements.remove(EDITABLE_INSTALL_FLAG)
                
    except FileNotFoundError:
        raise FileNotFoundError(f"Requirements file not found: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading requirements file {file_path}: {str(e)}")
    
    return requirements


def get_long_description() -> str:
    """
    Read and return the project's long description from README.md.
    
    Returns:
        str: Content of README.md file, used as PyPI long description
        
    Note:
        If README.md is not found, returns a default description.
    """
    readme_path = os.path.join(PROJECT_ROOT, 'README.md')
    
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return (
            "AI-Based Retail Transaction Prediction System\n"
            "============================================\n\n"
            "A machine learning system for predicting retail transaction amounts "
            "using features like quantity, price, customer data, and temporal information."
        )


# ============================================================================
# SETUP CONFIGURATION
# ============================================================================

setup(
    # ========================================================================
    # BASIC PROJECT INFORMATION
    # ========================================================================
    name='retail-transaction-predictor',
    version='1.0.0',
    description=(
        'AI-Based Retail Transaction Prediction System for forecasting transaction amounts'
    ),
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    
    # ========================================================================
    # AUTHOR INFORMATION
    # ========================================================================
    author='Srujan Vijay Kinjawadekar',
    author_email='srujan.kinjawadekar@srmedu.in',
    maintainer='Srujan Vijay Kinjawadekar',
    maintainer_email='srujan.kinjawadekar@srmedu.in',
    
    # ========================================================================
    # PROJECT URLs
    # ========================================================================
    url='https://github.com/yourusername/retail-transaction-predictor',
    project_urls={
        'Documentation': 'https://github.com/yourusername/retail-transaction-predictor/wiki',
        'Source Code': 'https://github.com/yourusername/retail-transaction-predictor',
        'Bug Tracker': 'https://github.com/yourusername/retail-transaction-predictor/issues',
    },
    
    # ========================================================================
    # LICENSE & LEGAL
    # ========================================================================
    license='MIT',
    
    # ========================================================================
    # CLASSIFICATION & METADATA
    # ========================================================================
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Natural Language :: English',
    ],
    keywords=[
        'machine-learning',
        'retail',
        'forecasting',
        'prediction',
        'transaction-analysis',
        'regression',
        'data-science',
    ],
    
    # ========================================================================
    # PYTHON VERSION REQUIREMENT
    # ========================================================================
    python_requires='>=3.8',
    
    # ========================================================================
    # PACKAGE DISCOVERY & CONFIGURATION
    # ========================================================================
    packages=find_packages(
        where=PROJECT_ROOT,
        include=['src*'],
    ),
    
    # ========================================================================
    # DEPENDENCIES
    # ========================================================================
    install_requires=get_requirements(os.path.join(PROJECT_ROOT, 'requirements.txt')),
    
    # ========================================================================
    # OPTIONAL DEPENDENCIES (EXTRAS)
    # ========================================================================
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'isort>=5.10.0',
            'mypy>=0.950',
        ],
        'docs': [
            'sphinx>=4.5.0',
            'sphinx-rtd-theme>=1.0.0',
            'sphinx-autodoc-typehints>=1.18.0',
        ],
        'jupyter': [
            'jupyter>=1.0.0',
            'jupyterlab>=3.0.0',
            'notebook>=6.0.0',
        ],
    },
    
    # ========================================================================
    # ENTRY POINTS (Command-line scripts and plugins)
    # ========================================================================
    entry_points={
        'console_scripts': [
            # Uncomment and modify when command-line interface is ready
            # 'retail-predictor=src.cli:main',
        ],
    },
    
    # ========================================================================
    # ADDITIONAL PACKAGE DATA
    # ========================================================================
    include_package_data=True,
    package_data={
        'src': [
            'templates/*.html',
            'static/**/*',
            'artifacts/**/*',
        ],
    },
    
    # ========================================================================
    # ZIP SAFETY
    # ========================================================================
    zip_safe=False,
    
    # ========================================================================
    # SETUP CONFIGURATION
    # ========================================================================
    setup_requires=['setuptools>=40.8.0'],
)


# ============================================================================
# DOCUMENTATION & NOTES
# ============================================================================

"""
INSTALLATION INSTRUCTIONS:
==========================

1. Development Installation (Editable Mode):
   pip install -e .

2. Full Installation with Development Tools:
   pip install -e .[dev,docs,jupyter]

3. Production Installation:
   pip install .


UNINSTALLATION:
===============
   pip uninstall retail-transaction-predictor


BUILD & DISTRIBUTION:
=====================

1. Create distribution packages:
   python setup.py sdist bdist_wheel

2. Upload to PyPI (requires twine):
   twine upload dist/*


PROJECT STRUCTURE:
==================

retail-transaction-predictor/
├── src/
│   ├── __init__.py
│   ├── logger.py
│   ├── exceptions.py
│   ├── utils.py
│   ├── component/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   ├── templates/
│   └── static/
├── notebook/
│   ├── Data/
│   │   └── data.csv
│   ├── EDA_Dataset_Performance.ipynb
│   └── Model_training.ipynb
├── artifacts/
│   ├── train.csv
│   ├── test.csv
│   ├── raw.csv
│   ├── preprocessor.pkl
│   └── model.pkl
├── setup.py
├── requirements.txt
├── app.py
└── README.md


KEY DEPENDENCIES:
=================

Core Libraries:
  - pandas (data manipulation)
  - numpy (numerical computing)
  - scikit-learn (machine learning)

Web Framework:
  - flask (web application)

Serialization:
  - pickle (model persistence)

Development:
  - pytest (testing)
  - black (code formatting)


TROUBLESHOOTING:
================

Issue: "requirements.txt not found"
Solution: Ensure requirements.txt exists in the project root directory

Issue: "No module named 'src'"
Solution: Install with: pip install -e .

Issue: "Python version not compatible"
Solution: Ensure Python >= 3.8 is installed


VERSION HISTORY:
================

v1.0.0 (February 22, 2026)
  - Initial release
  - Complete data pipeline (ingestion, transformation, training)
  - Prediction pipeline for batch and single transaction prediction
  - Feature engineering for retail transaction data
  - Model serialization and loading


CONTACT & SUPPORT:
==================

Author: Srujan Vijay Kinjawadekar
Email: srujan.kinjawadekar@srmedu.in
Institution: SRM Institute of Science and Technology
Program: MBA (Semester 4)
"""