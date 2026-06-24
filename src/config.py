# src/config.py
import os

class MLflowConfig:
    REPO_OWNER = "MuhammadUlulAlbabWicaksono"
    REPO_NAME = "MLOps-Prediksi-Harga-Crypto"
    
    # GANTI NAMA: Memutus rantai sejarah ke mlruns lokal
    EXPERIMENT_NAME = "BTC_Production_V3" 
    MODEL_NAME = "BTC_Price_Predictor"
    DEFAULT_ARTIFACT_PATH = "xgboost-model"
    
    THRESHOLDS = {
        "max_rmse": 250.0,
        "min_r2": 0.30
    }