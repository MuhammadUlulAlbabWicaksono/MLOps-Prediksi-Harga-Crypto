import requests
import json
import os
import time # Tambahan untuk jeda waktu (sleep)
from datetime import datetime

def fetch_coingecko_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "7"
    }
    
    # --- MODIFIKASI: Parameter Error Handling ---
    max_retries = 3
    base_delay = 30 # Waktu tunggu dasar 30 detik
    
    # Pastikan folder ada sebelum menyimpan file
    os.makedirs("data/raw", exist_ok=True)
    # ------------------------------------------

    print("Memulai pengambilan data dari CoinGecko API...")
    
    # --- MODIFIKASI: Logika pengulangan (Retry) & Exponential Backoff ---
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"data/raw/btc_raw_{timestamp}.json"

                with open(filepath, "w") as f:
                    json.dump(data, f, indent=4)
                    
                print(f"Sukses! Data berhasil disimpan di: {filepath}")
                return # Keluar dari fungsi jika sukses
            else:
                print(f"Percobaan {attempt + 1} Gagal. HTTP Status Code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"Percobaan {attempt + 1} Gagal karena masalah koneksi: {e}")
        
        # Jika belum percobaan terakhir, lakukan backoff
        if attempt < max_retries - 1:
            sleep_time = base_delay * (2 ** attempt) # Menghasilkan jeda 30s, lalu 60s
            print(f"Menerapkan exponential backoff. Menunggu {sleep_time} detik sebelum mencoba lagi...\n")
            time.sleep(sleep_time)
            
    print("Pengambilan data dihentikan. Gagal setelah maksimal 3 kali percobaan.")
    # ----------------------------------------------------------------------

if __name__ == "__main__":
    fetch_coingecko_data()