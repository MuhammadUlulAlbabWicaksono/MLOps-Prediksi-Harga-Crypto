class MLflowConfig:
    """Single Source of Truth untuk konfigurasi MLflow dan Model."""
    
    # NAMA BARU: Memaksa pembuatan skema mlflow-artifacts:/ yang murni
    EXPERIMENT_NAME = "BTC_MLOps_Registry"
    MODEL_NAME = "BTC_Price_Predictor"
    DEFAULT_ARTIFACT_PATH = "xgboost-model"
    
    THRESHOLDS = {
        "max_rmse": 250.0,
        "min_r2": 0.30
    }