import os
import glob
import json
import pandas as pd
import numpy as np
import talib

def preprocess_latest_data():
    # 1. Mencari file data mentah terbaru
    raw_files = glob.glob("data/raw/btc_raw_*.json")
    if not raw_files:
        print("Tidak ada file mentah di data/raw/. Jalankan ingest_data.py terlebih dahulu.")
        return
    
    latest_file = max(raw_files, key=os.path.getctime)
    print(f"Memproses file JSON terbaru: {latest_file}")
    
    # 2. Membaca data JSON
    with open(latest_file, "r") as f:
        data = json.load(f)
        
    # 3. Ekstraksi struktur nested array menjadi DataFrame
    df_prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df_market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
    df_volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
    
    # Menggabungkan data berdasarkan timestamp
    df = df_prices.merge(df_market_caps, on='timestamp').merge(df_volumes, on='timestamp')
    
    # 4. Cleaning & Transformasi Waktu
    # Konversi Unix millisecond ke format datetime yang valid
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.drop(columns=['timestamp'], inplace=True)
    
    # Mengisi missing values jika ada (Forward Fill)
    df.ffill(inplace=True)
    
    # 5. Feature Engineering (Indikator Teknikal)
    print("Mengekstraksi fitur teknikal (MA, RSI, MACD)...")
    prices = df['price'].values
    
    # Moving Averages (7 & 30 period)
    df['MA_7'] = talib.SMA(prices, timeperiod=7)
    df['MA_30'] = talib.SMA(prices, timeperiod=30)
    
    # Relative Strength Index (14 period)
    df['RSI_14'] = talib.RSI(prices, timeperiod=14)
    
    # Moving Average Convergence Divergence
    macd, macdsignal, macdhist = talib.MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    
    # Membuang baris awal yang bernilai NaN akibat perhitungan periode indikator
    df.dropna(inplace=True)
    
    # 6. Menyimpan Data Bersih ke Feature Store (Lokal)
    os.makedirs("data/processed", exist_ok=True)
    
    # Menggunakan nama file yang sama dengan raw, namun ekstensi CSV
    output_filename = os.path.basename(latest_file).replace("raw", "processed").replace(".json", ".csv")
    output_path = os.path.join("data/processed", output_filename)
    
    df.to_csv(output_path)
    print(f"Sukses! Data siap latih telah disimpan di: {output_path}")

if __name__ == "__main__":
    preprocess_latest_data()