"""
config.py
=========
Application configuration loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # MLflow
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_model_serve_uri: str = os.getenv("MLFLOW_MODEL_SERVE_URI", "http://localhost:5001")

    # Data paths
    data_dir: str = os.getenv("DATA_DIR", "data/simulated")
    eda_dir: str = os.getenv("EDA_DIR", "eda")
    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "mlartifacts")
    config_dir: str = os.getenv("CONFIG_DIR", "training/config")

    # Streaming
    stream_interval_sec: float = float(os.getenv("STREAM_INTERVAL_SEC", "1.0"))

    # Model
    best_model_name: str = "hx_frost_dP_error_lasso"
    best_model_run_id: str = "90a44c352c3d4fe3a45df3e63a3e01b9"

    # Physics constants
    alarm_threshold_pa: float = 943.3
    dp_ideal_pa: float = 618.4


settings = Settings()
