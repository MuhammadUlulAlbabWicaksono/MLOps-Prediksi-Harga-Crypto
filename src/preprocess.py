import os
import glob
import json
import pandas as pd
import numpy as np

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
    
    print("Mengekstraksi fitur teknikal (MA, RSI, MACD) menggunakan Pandas native...")
    prices = df['price']
    
    # 1. Moving Averages (MA)
    df['MA_7'] = prices.rolling(window=7).mean()
    df['MA_30'] = prices.rolling(window=30).mean()
    
    # 2. RSI (Relative Strength Index)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 3. MACD
    ema_12 = prices.ewm(span=12, adjust=False).mean()
    ema_26 = prices.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df.dropna(inplace=True)
    
    os.makedirs("data/processed", exist_ok=True)
    
    output_filename = os.path.basename(latest_file).replace("raw", "processed").replace(".json", ".csv")
    output_path = os.path.join("data/processed", output_filename)
    
    df.to_csv(output_path)
    print(f"Sukses! Data siap latih telah disimpan di: {output_path}")

if __name__ == "__main__":
    preprocess_latest_data()