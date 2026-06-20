import os
import glob
import json
import pandas as pd
import numpy as np
import talib

def preprocess_latest_data():
    raw_files = glob.glob("data/raw/btc_raw_*.json")
    if not raw_files:
        print("Tidak ada file mentah di data/raw/. Jalankan ingest_data.py terlebih dahulu.")
        return
    
    latest_file = max(raw_files, key=os.path.getctime)
    print(f"Memproses file JSON terbaru: {latest_file}")
    
    with open(latest_file, "r") as f:
        data = json.load(f)
        
    df_prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df_market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
    df_volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
    
    df = df_prices.merge(df_market_caps, on='timestamp').merge(df_volumes, on='timestamp')
    
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.drop(columns=['timestamp'], inplace=True)
    
    df.ffill(inplace=True)
    
    print("Mengekstraksi fitur teknikal (MA, RSI, MACD)...")
    prices = df['price'].values
    
    df['MA_7'] = talib.SMA(prices, timeperiod=7)
    df['MA_30'] = talib.SMA(prices, timeperiod=30)
    
    df['RSI_14'] = talib.RSI(prices, timeperiod=14)
    
    macd, macdsignal, macdhist = talib.MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    
    df.dropna(inplace=True)
    
    os.makedirs("data/processed", exist_ok=True)
    
    output_filename = os.path.basename(latest_file).replace("raw", "processed").replace(".json", ".csv")
    output_path = os.path.join("data/processed", output_filename)
    
    df.to_csv(output_path)
    print(f"Sukses! Data siap latih telah disimpan di: {output_path}")

if __name__ == "__main__":
    preprocess_latest_data()