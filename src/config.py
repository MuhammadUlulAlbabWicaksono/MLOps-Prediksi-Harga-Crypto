# src/config.py
import os

class MLflowConfig:
    """Single Source of Truth untuk konfigurasi MLflow dan Model."""
    
    # EKSPERIMEN BARU: Memaksa pembuatan jalur proxy HTTP (S3) murni
    EXPERIMENT_NAME = "BTC_Pipeline_Final"
    MODEL_NAME = "BTC_Price_Predictor"
    DEFAULT_ARTIFACT_PATH = "xgboost-model"
    
    THRESHOLDS = {
        "max_rmse": 250.0,
        "min_r2": 0.30
    }