class MLflowConfig:
    REPO_OWNER = "MuhammadUlulAlbabWicaksono"
    REPO_NAME = "mlops-prediksi-harga-crypto" 
    
    EXPERIMENT_NAME = "BTC_Model_Registry_Final"
    MODEL_NAME = "BTC_Price_Predictor"
    DEFAULT_ARTIFACT_PATH = "xgboost-model"
    
    THRESHOLDS = {
        "max_rmse": 250.0,
        "min_r2": 0.30
    }