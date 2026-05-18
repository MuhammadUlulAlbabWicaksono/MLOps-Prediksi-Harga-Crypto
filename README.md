# 📈 MLOps: Sistem Prediksi Harga Cryptocurrency Real-time

Proyek ini merupakan implementasi *end-to-end Machine Learning Operations* (MLOps) untuk memprediksi harga Bitcoin (BTC/USD) secara *real-time*. Sistem ini memanfaatkan data dinamis dari CoinGecko API dan menerapkan pendekatan *Continual Learning* untuk mendeteksi *data drift* serta melakukan *retraining* model secara otomatis.

Proyek ini dikembangkan sebagai pemenuhan tugas mata kuliah MLOps di Fakultas Ilmu Komputer, Universitas Brawijaya.

## Tech Stack & Arsitektur
Sistem ini dibangun menggunakan arsitektur modular dengan *tech stack* berikut:
* **Orchestrator:** Apache Airflow
* **Data Storage:** MinIO (Raw Data Lake) & PostgreSQL (Feature Store)
* **ML Lifecycle:** Scikit-learn/XGBoost (Modeling) & MLflow (Experiment Tracking & Registry)
* **Serving:** FastAPI & Docker
* **Monitoring:** Prometheus, Grafana, & Evidently AI (Drift Detection)

## Struktur Direktori
Proyek ini mengadopsi standar *Cookiecutter Data Science* untuk menjaga kerapian dan skalabilitas:
* `data/` : Berisi subfolder `raw/` untuk data mentah dari API dan `processed/` untuk data fitur.
* `models/` : Direktori untuk menyimpan artefak model dan *scaler*.
* `notebooks/` : File Jupyter Notebook untuk eksperimen, EDA, dan *prototyping* awal.
* `src/` : Kode sumber utama sistem (script *ingestion*, *preprocessing*, *training*, *serving*).
* `config/` : Berisi *file* konfigurasi sistem, variabel environment, dan parameter ML.
* `tests/` : Script *unit testing* untuk menjaga kualitas kode (CI/CD).
* `docs/` : Dokumentasi tambahan dan diagram arsitektur sistem.
* `.devcontainer/` : Konfigurasi *environment* GitHub Codespaces.

## Instruksi Penggunaan (Environment Setup)
Proyek ini dirancang agar 100% *reproducible* menggunakan **GitHub Codespaces**. sehingga tidak perlu melakukan instalasi manual di komputer lokal.

**Langkah-langkah menjalankan environment:**
1. Buka halaman utama repositori ini pada GitHub.
2. Klik tombol hijau **Code** di pojok kanan atas tabel *file*.
3. Pilih tab **Codespaces**, lalu klik **Create codespace on main**.
4. Tunggu beberapa menit. GitHub akan secara otomatis:
   - Menyiapkan OS berbasis container.
   - Menginstal Python 3.10.
   - Menginstal semua dependensi dari `requirements.txt` (Pandas, Scikit-learn, MLflow, FastAPI, dll).
   - Menyiapkan ekstensi VS Code (Jupyter, GitLens).
5. Setelah VS Code terbuka di browser, Anda bisa langsung menjalankan *script* melalui terminal yang tersedia di bagian bawah layar.

## 📌 Alur Kerja / Pipeline Mendatang (Roadmap)
1. **Data Ingestion:** *Fetching* data periodik dari CoinGecko setiap 10 menit via Airflow.
2. **Feature Engineering:** Ekstraksi indikator teknikal (RSI, MA) dan penyimpanan ke Feature Store.
3. **Model Training & Registry:** Pelatihan model *time-series* dan pencatatan eksperimen via MLflow.
4. **Model Deployment:** Ekspos REST API menggunakan FastAPI untuk melayani prediksi harga.
5. **Monitoring & Retraining:** Pemantauan *data drift* via Evidently AI untuk memicu *retraining* model secara otomatis.
