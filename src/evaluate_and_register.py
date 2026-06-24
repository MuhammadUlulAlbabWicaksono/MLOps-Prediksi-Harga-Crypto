import os
import sys
import json
import mlflow
from mlflow import MlflowClient

# Menghubungkan ke DagsHub via Secrets (Tanpa library tambahan)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

def evaluate_and_register():
    print("\n--- MEMULAI EVALUASI & AUTO-REGISTRY ---")
    client = MlflowClient()
    
    experiment_name = "BTC_Model_Registry_Prod"
    model_name = "BTC_Price_Predictor"
    
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Gagal: Eksperimen '{experiment_name}' tidak ditemukan.")
        sys.exit(1)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        print("Gagal: Riwayat Run kosong.")
        sys.exit(1)

    latest_run = runs[0]
    run_id = latest_run.info.run_id
    metrics = latest_run.data.metrics
    tags = latest_run.data.tags # Mengambil kumpulan metadata

    rmse = metrics.get("rmse", float('inf'))
    r2 = metrics.get("r2", -float('inf'))

    print(f"Run ID: {run_id} | RMSE: {rmse:.2f} | R2: {r2:.2f}")

    if rmse < 250.0 and r2 > 0.30:
        print("\n[LULUS] Model memenuhi threshold.")
        
        # PENDEKATAN NATIVE: Membaca path artifact asli dari metadata
        model_history_json = tags.get("mlflow.log-model.history")
        if not model_history_json:
            print("GAGAL: Metadata log-model.history tidak ditemukan. Model tidak tercatat di server.")
            sys.exit(1)
        
        model_history = json.loads(model_history_json)
        # Ekstrak artifact_path aktual yang digunakan oleh MLflow di server
        actual_artifact_path = model_history[0]["artifact_path"]
        
        # Susun URI secara dinamis
        model_uri = f"runs:/{run_id}/{actual_artifact_path}"
        print(f"Model fisik ditemukan di URI dinamis: {model_uri}")
        
        # Mendaftarkan model
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
        
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=model_version.version
        )
        print(f"SUKSES: Model '{model_name}' Versi {model_version.version} diset ke 'staging'.")
    else:
        print("\n[GAGAL] Metrik di bawah standar. Registrasi dibatalkan.")
        sys.exit(1) 

if __name__ == "__main__":
    evaluate_and_register()