# src/config.py
import os

class MLflowConfig:
    """Single Source of Truth untuk konfigurasi MLflow dan Model."""
    
    # Kredensial & Repositori
    REPO_OWNER = "MuhammadUlulAlbabWicaksono"
    REPO_NAME = "MLOps-Prediksi-Harga-Crypto"
    
    # MLflow Metadata
    EXPERIMENT_NAME = "BTC_Model_Registry_Prod"
    MODEL_NAME = "BTC_Price_Predictor"
    DEFAULT_ARTIFACT_PATH = "xgboost-model"
    
    # Threshold Evaluasi Performa
    THRESHOLDS = {
        "max_rmse": 250.0,
        "min_r2": 0.30
    }