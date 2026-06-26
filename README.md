Markdown
# MLOps: Sistem Prediksi Harga Cryptocurrency Real-time

Proyek ini merupakan implementasi *end-to-end Machine Learning Operations* (MLOps) untuk memprediksi harga Bitcoin (BTC/USD) secara *real-time*. Sistem ini memanfaatkan data dinamis dari CoinGecko API dan menerapkan pendekatan *Continual Learning* untuk mendeteksi *data drift* serta melakukan *retraining* model secara otomatis.

Proyek ini dikembangkan sebagai pemenuhan tugas mata kuliah MLOps di Fakultas Ilmu Komputer, Universitas Brawijaya.

---

## Tech Stack & Arsitektur
Sistem ini dibangun menggunakan arsitektur modular (*microservices*) dengan *tech stack* berikut:
* **Orchestrator:** Apache Airflow
* **Data Storage & Versioning:** MinIO (Raw Data Lake), PostgreSQL (Metadata/Feature Store), & DVC (Data Version Control)
* **ML Lifecycle:** Scikit-learn/XGBoost (Modeling) & MLflow (Experiment Tracking & Registry)
* **Serving:** FastAPI & Docker Compose
* **Monitoring:** Prometheus, Grafana, & Evidently AI (Drift Detection)

---

## Struktur Direktori Proyek
Proyek ini mengadopsi standar *Cookiecutter Data Science* yang telah diintegrasikan dengan DVC dan Docker untuk menjaga kerapian, skalabilitas, dan *versioning* yang optimal:

```text
├── .devcontainer/      # Konfigurasi environment GitHub Codespaces
├── .dvc/               # Konfigurasi internal Data Version Control
├── config/             # File konfigurasi sistem, variabel environment, dan parameter ML
├── data/
│   ├── processed/      # Menyimpan file .csv siap latih (dilacak DVC)
│   ├── raw/            # Menyimpan respons mentah JSON API (dilacak DVC)
├── docs/               # Dokumentasi tambahan dan diagram arsitektur sistem
├── models/             # Direktori untuk menyimpan artefak model dan scaler lokal
├── src/
│   ├── main.py         # Script utama FastAPI (Inference Engine)
│   ├── ingest_data.py  # Skrip pengumpul data dinamis (CoinGecko API)
│   └── preprocess.py   # Skrip otomasi pembersihan data & feature engineering
├── tests/              # Script unit testing untuk menjaga kualitas kode (CI/CD)
├── .env.example        # Templat environment variables untuk Docker
├── docker-compose.yaml # Orkestrasi microservices (Postgres, MinIO, MLflow, FastAPI)
├── Dockerfile          # Instruksi kompilasi image untuk layanan FastAPI
├── Dockerfile.mlflow   # Instruksi kompilasi image khusus untuk peladen MLflow
└── README.md
Status Pengembangan Saat Ini
Data Ingestion & Versioning (LK-04 & LK-05): Skrip pengambilan data dari CoinGecko API dengan Exponential Backoff telah selesai. Seluruh dataset dilacak menggunakan DVC (Data Version Control) untuk menjaga Data Lineage.

Model Registry & Experiment Tracking (LK-07 & LK-08): Pelatihan model menggunakan algoritma time-series/XGBoost telah terintegrasi dengan MLflow. Model terbaik dicatat dalam Model Registry dengan alias Staging/Production. Metadata model turut dilacak oleh DVC (model_metadata.yaml.dvc).

Microservices Orchestration (LK-09): Infrastruktur sistem telah dimigrasikan ke dalam arsitektur kontainer berbasis Docker Compose. Sistem kini mengorkestrasi 4 layanan terisolasi: db-service (PostgreSQL), s3-service (MinIO), mlflow-service (Tracking Server), dan api-service (FastAPI) yang saling terhubung melalui custom bridge network dan menggunakan Persistence Volumes untuk mengamankan data.

Instruksi Penggunaan (Environment Setup)
Proyek ini dirancang agar 100% reproducible menggunakan GitHub Codespaces dan Docker.

Tahap 1: Inisialisasi Lingkungan (Codespaces)
Buka halaman utama repositori ini pada GitHub.

Klik tombol hijau Code, pilih tab Codespaces, lalu klik Create codespace on main.

Tunggu hingga GitHub mengalokasikan OS berbasis container dan menginstal dependensi dasar.

Tahap 2: Menjalankan Arsitektur MLOps (Docker Compose)
Setelah masuk ke dalam terminal VS Code (Codespaces), jalankan perintah berikut untuk menghidupkan seluruh layanan (Database, Storage, MLflow, dan API):

Bash
# 1. Salin template variabel lingkungan (environment variables)
cp .env.example .env

# 2. Bangun dan nyalakan seluruh arsitektur microservices di latar belakang
docker compose up -d --build

# 3. Verifikasi status kesehatan kontainer (Pastikan semuanya "Up (healthy)")
docker compose ps
Akses Layanan:

FastAPI (Swagger UI): http://localhost:8000/docs

MLflow UI: http://localhost:5000

MinIO Console (Data Lake): http://localhost:9001 (Gunakan kredensial dari file .env)

Alur Kerja / Pipeline Mendatang (Roadmap)
[x] Data Ingestion: Fetching data periodik dari CoinGecko API & Data Versioning (DVC).

[x] Feature Engineering: Ekstraksi indikator teknikal (RSI, MA) dan penyimpanan ke Feature Store.

[x] Model Training & Registry: Pelatihan model dan pencatatan eksperimen via MLflow secara tersentralisasi.

[x] Model Deployment: Mengisolasi sistem dengan Docker Compose dan mengekspos REST API menggunakan FastAPI yang dapat mengunduh model secara dinamis.

[ ] Monitoring & Retraining: Pemantauan data drift via Evidently AI untuk memicu retraining model secara otomatis via Apache Airflow.