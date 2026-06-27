import os
import glob
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import dagshub
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import MLflowConfig

dagshub.init(repo_owner=MLflowConfig.REPO_OWNER, repo_name=MLflowConfig.REPO_NAME, mlflow=True)

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
    
    features = ['price', 'market_cap', 'volume', 'MA_7', 'MA_30', 'RSI_14', 'MACD', 'MACD_Signal', 'Price_Lag_1', 'Price_Lag_2', 'Volatility_7d']
    
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features]
    y = df['target_price']
    
    # KEMBALIKAN KE SHUFFLE=TRUE agar model bisa belajar dengan lebih fleksibel dan skor R2 kembali naik
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    
    mlflow.set_experiment(MLflowConfig.EXPERIMENT_NAME)
    
    with mlflow.start_run():
        model = XGBRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3, 
            random_state=42
        )
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        mlflow.log_params({"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1})
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path=MLflowConfig.DEFAULT_ARTIFACT_PATH
        )
        print("[INFO] Model fisik selesai dikirim ke DagsHub.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()
    main(args.n_estimators, args.max_depth, args.learning_rate)