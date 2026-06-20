# MLOps-Prediksi-Harga-Crypto
Repositori untuk proyek Sistem Prediksi Harga Cryptocurrency Real-time berbasis Continual Learning.

## Struktur Direktori Proyek (LK-04)
```text
├── data/
│   ├── processed/      # Menyimpan file .csv hasil pembersihan dan ekstraksi fitur
│   └── raw/            # Menyimpan file respons mentah JSON dari API
├── src/
│   ├── ingest_data.py  # Skrip pengumpul data dinamis dengan mekanisme Exponential Backoff
│   └── preprocess.py   # Skrip otomasi pembersihan data & feature engineering
└── README.md