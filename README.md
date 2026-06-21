# MLOps: Sistem Prediksi Harga Cryptocurrency Real-time

Proyek ini merupakan implementasi *end-to-end Machine Learning Operations* (MLOps) untuk memprediksi harga Bitcoin (BTC/USD) secara *real-time*. Sistem ini memanfaatkan data dinamis dari CoinGecko API dan menerapkan pendekatan *Continual Learning* untuk mendeteksi *data drift* serta melakukan *retraining* model secara otomatis.

Proyek ini dikembangkan sebagai pemenuhan tugas mata kuliah MLOps di Fakultas Ilmu Komputer, Universitas Brawijaya.

---

## Tech Stack & Arsitektur
Sistem ini dibangun menggunakan arsitektur modular dengan *tech stack* berikut:
* **Orchestrator:** Apache Airflow
* **Data Storage & Versioning:** MinIO (Raw Data Lake), PostgreSQL (Feature Store), & DVC (Data Version Control)
* **ML Lifecycle:** Scikit-learn/XGBoost (Modeling) & MLflow (Experiment Tracking & Registry)
* **Serving:** FastAPI & Docker
* **Monitoring:** Prometheus, Grafana, & Evidently AI (Drift Detection)

---

## Struktur Direktori Proyek
Proyek ini mengadopsi standar *Cookiecutter Data Science* yang telah diintegrasikan dengan DVC untuk menjaga kerapian, skalabilitas, dan *versioning* data yang optimal:

```text
├── .devcontainer/      # Konfigurasi environment GitHub Codespaces
├── .dvc/               # Konfigurasi internal Data Version Control
├── config/             # File konfigurasi sistem, variabel environment, dan parameter ML
├── data/
│   ├── processed/      # Menyimpan file .csv siap latih (diabaikan oleh Git, dilacak DVC)
│   ├── raw/            # Menyimpan respons mentah JSON API (diabaikan oleh Git, dilacak DVC)
│   ├── processed.dvc   # File metadata penunjuk DVC untuk direktori processed
│   └── raw.dvc         # File metadata penunjuk DVC untuk direktori raw
├── docs/               # Dokumentasi tambahan dan diagram arsitektur sistem
├── models/             # Direktori untuk menyimpan artefak model dan scaler
├── notebooks/          # File Jupyter Notebook untuk eksperimen, EDA, dan prototyping awal
├── src/
│   ├── ingest_data.py  # Skrip pengumpul data dinamis dengan mekanisme Exponential Backoff
│   └── preprocess.py   # Skrip otomasi pembersihan data & feature engineering
├── tests/              # Script unit testing untuk menjaga kualitas kode (CI/CD)
└── README.md

Status Pengembangan Saat Ini
Data Ingestion (LK-04): Skrip ingest_data.py telah selesai diimplementasikan untuk mengambil data dari CoinGecko API. Skrip ini dilengkapi dengan mekanisme Exponential Backoff untuk menangani rate limit atau kegagalan koneksi secara otomatis.

Data Version Control (LK-05): DVC telah diinisialisasi dan dikonfigurasi. Data mentah (data/raw/) dan data hasil pemrosesan (data/processed/) kini dilacak sepenuhnya oleh DVC melalui file .dvc masing-masing, sementara file data aktualnya diabaikan oleh Git agar repositori tetap ringan.

Instruksi Penggunaan (Environment Setup)
Proyek ini dirancang agar 100% reproducible menggunakan GitHub Codespaces, sehingga Anda tidak perlu melakukan instalasi manual di komputer lokal.

Langkah-langkah menjalankan environment:

Buka halaman utama repositori ini pada GitHub.

Klik tombol hijau Code di pojok kanan atas tabel file.

Pilih tab Codespaces, lalu klik Create codespace on main.

Tunggu beberapa menit. GitHub akan secara otomatis:

Menyiapkan OS berbasis container.

Menginstal Python 3.10.

Menginstal semua dependensi dari requirements.txt (DVC, Pandas, Scikit-learn, MLflow, FastAPI, dll).

Menyiapkan ekstensi VS Code (Jupyter, GitLens).

Setelah VS Code terbuka di browser, Anda bisa langsung menjalankan script melalui terminal yang tersedia di bagian bawah layar.

Alur Kerja / Pipeline Mendatang (Roadmap)
[SELESAI] Data Ingestion: Fetching data periodik dari CoinGecko API & Data Versioning (DVC).

[BERJALAN] Feature Engineering: Ekstraksi indikator teknikal (RSI, MA) melalui preprocess.py dan penyimpanan ke Feature Store.

[MENDATANG] Model Training & Registry: Pelatihan model time-series dan pencatatan eksperimen via MLflow.

[MENDATANG] Model Deployment: Ekspos REST API menggunakan FastAPI untuk melayani prediksi harga.

[MENDATANG] Monitoring & Retraining: Pemantauan data drift via Evidently AI untuk memicu retraining model secara otomatis via Apache Airflow.