import requests
import json
import os
from datetime import datetime

def fetch_coingecko_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "7"
    }
    
    print("Memulai pengambilan data dari CoinGecko API...")
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"data/raw/btc_raw_{timestamp}.json"

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
            
        print(f"Sukses! Data berhasil disimpan di: {filepath}")
    else:
        print(f"Gagal mengambil data. HTTP Status Code: {response.status_code}")

if __name__ == "__main__":
    fetch_coingecko_data()