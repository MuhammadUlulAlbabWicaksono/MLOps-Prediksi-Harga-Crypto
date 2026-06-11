# MLOps-Prediksi-Harga-Crypto
Repositori untuk proyek Sistem Prediksi Harga Cryptocurrency Real-time berbasis Continual Learning.

## Cara Menjalankan Pipeline ETL (Data Ingestion & Preprocessing)
Sistem ini menggunakan skrip otomatis untuk menarik data dinamis dan melakukan *feature engineering* secara *real-time*.

1. **Jalankan Ingestion Data:**
   Buka terminal Codespaces dan jalankan perintah berikut untuk mengekstrak data dari CoinGecko:
   ```bash
   python src/ingest_data.py