import os
import shutil
import pandas as pd
import mlflow.xgboost
from xgboost import XGBRegressor

# 1. Siapkan data yang sesuai dengan kolom API Anda
cols = ["price", "market_cap", "volume", "MA_7", "MA_30", "RSI_14", "MACD", "MACD_Signal", "Price_Lag_1", "Price_Lag_2", "Volatility_7d"]
X = pd.DataFrame([
    [64000.5, 1200000000000.0, 35000000000.0, 63500.1, 61200.45, 55.5, 120.4, 110.2, 63900.0, 63850.0, 450.5],
    [65000.0, 1210000000000.0, 36000000000.0, 64500.0, 62200.00, 56.5, 121.4, 111.2, 64000.0, 63950.0, 451.5]
], columns=cols)
y = pd.Series([64100.0, 65100.0])

print("[1] Melatih ulang model XGBoost secara instan...")
model = XGBRegressor()
model.fit(X, y)

print("[2] Membersihkan sisa folder lama jika ada...")
if os.path.exists("model_lokal"):
    shutil.rmtree("model_lokal")

print("[3] Menyimpan model asli ke dalam format MLflow yang sah...")
# Ini akan membuat struktur MLmodel yang selama ini dicari oleh Docker
mlflow.xgboost.save_model(model, "model_lokal")
print("SUKSES: File MLmodel dan xgboost-model telah dibuat secara utuh di folder 'model_lokal'!")