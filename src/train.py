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
import dagshub

dagshub.init(repo_owner='MuhammadUlulAlbabWicaksono', repo_name='MLOps-Prediksi-Harga-Crypto', mlflow=True)

def get_latest_data():
    processed_files = glob.glob("data/processed/btc_processed_*.csv")
    if not processed_files:
        raise FileNotFoundError("Tidak ada file data terproses. Jalankan preprocess.py dahulu.")
    return max(processed_files, key=os.path.getctime)

def main(n_estimators, max_depth, learning_rate):
    latest_file = get_latest_data()
    print(f"Menggunakan dataset: {latest_file}")
    
    # Load data
    df = pd.read_csv(latest_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    df['target_price'] = df['price'].shift(-1)
    df.dropna(inplace=True) 
    
    # Fitur (X) dan Target (y)
    features = ['price', 'market_cap', 'volume', 'MA_7', 'MA_30', 'RSI_14', 'MACD', 'MACD_Signal']
    X = df[features]
    y = df['target_price']
    
    # Train-test split (Time-series tidak boleh di-shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Set MLflow tracking
    mlflow.set_experiment("BTC_Price_Prediction_XGBoost")
    
    with mlflow.start_run():
        print(f"Melatih model dengan: n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}")
        
        model = XGBRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Metrik Model -> RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        mlflow.xgboost.log_model(model, "xgboost-model")
        print("Eksperimen berhasil dicatat di MLflow!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()
    
    main(args.n_estimators, args.max_depth, args.learning_rate)