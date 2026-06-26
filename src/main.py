import logging
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="BTC Price Predictor API", version="1.0")

model = None
MODEL_NAME = "BTC_Price_Predictor"

@app.on_event("startup")
def load_model():
    global model
    try:
        logger.info("Mencoba mengunduh model dari MLflow Registry (S3)...")
        model_uri = f"models:/{MODEL_NAME}@staging"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("SUCCESS: Model loaded successfully from MLflow!")
    except Exception as e:
        logger.warning(f"Peringatan: Gagal memuat model dari MLflow. Detail: {e}")
        logger.warning("Pastikan MLflow Server menyala dan model berstatus 'staging' tersedia.")

class FeatureInput(BaseModel):
    price: float
    market_cap: float
    volume: float
    MA_7: float
    MA_30: float
    RSI_14: float
    MACD: float
    MACD_Signal: float

@app.get("/")
def health_check():
    return {"status": "success", "message": "API Inferensi MLOps Aktif!"}

@app.post("/predict")
def predict(data: FeatureInput):
    if model is None:
        return {"status": "error", "message": "Model belum dimuat ke dalam memori!"}
    
    # Format input akan disesuaikan untuk model XGBoost (dummy response untuk sekarang)
    return {"status": "success", "prediction_usd": 65000.00}
