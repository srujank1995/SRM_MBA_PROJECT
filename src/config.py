"""Configuration settings for the demand forecasting project."""
import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Load config
_config = load_config()


@dataclass
class DataConfig:
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    sample_data_path: str = "data/sample_data.csv"
    target_column: str = "Quantity"
    date_column: str = "InvoiceDate"
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfig:
    artifacts_dir: str = "artifacts"
    models_dir: str = "models"
    logs_dir: str = "logs"


@dataclass
class FeatureConfig:
    categorical_columns: List[str] = field(
        default_factory=lambda: ["Country", "StockCode"]
    )
    numerical_columns: List[str] = field(
        default_factory=lambda: ["UnitPrice"]
    )
    lag_periods: List[int] = field(default_factory=lambda: [1, 7, 14, 30])
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 30])


DATA_CONFIG = DataConfig()
MODEL_CONFIG = ModelConfig()
FEATURE_CONFIG = FeatureConfig()
