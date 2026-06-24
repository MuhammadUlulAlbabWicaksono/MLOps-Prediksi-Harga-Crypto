import os
import glob
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import MLflowConfig

# MURNI MLFLOW: Membaca URI langsung dari sistem GitHub Actions
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

def get_latest_data():
    processed_files = glob.glob("data/processed/btc_processed_*.csv")
    if not processed_files:
        raise FileNotFoundError("Tidak ada file data terproses.")
    return max(processed_files)

def main(n_estimators, max_depth, learning_rate):
    latest_file = get_latest_data()
    print(f"[INFO] Dataset: {latest_file}")
    
    df = pd.read_csv(latest_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    df['target_price'] = df['price'].shift(-1)
    df.dropna(inplace=True) 
    
    features = ['price', 'market_cap', 'volume', 'MA_7', 'MA_30', 'RSI_14', 'MACD', 'MACD_Signal']
    X = df[features]
    y = df['target_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    mlflow.set_experiment(MLflowConfig.EXPERIMENT_NAME)
    
    with mlflow.start_run():
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"[INFO] RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")
        
        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": learning_rate})
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        
        print(f"[INFO] Menyimpan model ke path: {MLflowConfig.DEFAULT_ARTIFACT_PATH}")
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path=MLflowConfig.DEFAULT_ARTIFACT_PATH
        )
        print("[INFO] Model fisik berhasil diunggah via MLflow HTTP Proxy.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()
    main(args.n_estimators, args.max_depth, args.learning_rate)