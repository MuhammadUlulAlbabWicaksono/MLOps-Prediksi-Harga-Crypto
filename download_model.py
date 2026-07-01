import os
import dagshub
import mlflow

print("[1] Menginisialisasi koneksi otentikasi DagsHub...")
dagshub.init(repo_owner='MuhammadUlulAlbabWicaksono', repo_name='mlops-prediksi-harga-crypto', mlflow=True)

print("[2] Mengunduh artefak model dari Registry...")
# Membuat folder penampung lokal
os.makedirs("model_lokal", exist_ok=True)

# Mengunduh seluruh isi folder model staging secara utuh
try:
    mlflow.artifacts.download_artifacts(
        artifact_uri="models:/BTC_Price_Predictor@staging",
        dst_path="model_lokal"
    )
    print("[3] Sukses! Artefak model telah berhasil diunduh di folder 'model_lokal'.")
except Exception as e:
    print(f"[!] Gagal mengunduh artefak: {e}")