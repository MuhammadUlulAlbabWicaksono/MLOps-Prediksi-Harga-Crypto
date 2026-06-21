# MLOps-Prediksi-Harga-Crypto
Repositori untuk proyek Sistem Prediksi Harga Cryptocurrency Real-time berbasis Continual Learning.

## Struktur Direktori Proyek
```text
├── .dvc/               # Konfigurasi internal Data Version Control
├── data/
│   ├── processed/      # Menyimpan file .csv siap latih (diabaikan oleh Git, dilacak DVC)
│   ├── raw/            # Menyimpan respons mentah JSON API (diabaikan oleh Git, dilacak DVC)
│   ├── processed.dvc   # File metadata penunjuk DVC untuk direktori processed
│   └── raw.dvc         # File metadata penunjuk DVC untuk direktori raw
├── src/
│   ├── ingest_data.py  # Skrip pengumpul data dinamis dengan mekanisme Exponential Backoff
│   └── preprocess.py   # Skrip otomasi pembersihan data & feature engineering
└── README.md