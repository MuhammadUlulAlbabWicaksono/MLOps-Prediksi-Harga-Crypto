class MLflowConfig:
    REPO_OWNER = "MuhammadUlulAlbabWicaksono"
    # UBAH: Gunakan huruf kecil semua sesuai URL DagsHub untuk mencegah S3 Path Error
    REPO_NAME = "mlops-prediksi-harga-crypto" 
    
    EXPERIMENT_NAME = "BTC_Pipeline_Final"
    MODEL_NAME = "BTC_Price_Predictor"
    DEFAULT_ARTIFACT_PATH = "xgboost-model"
    
    THRESHOLDS = {
        "max_rmse": 250.0,
        "min_r2": 0.30
    }