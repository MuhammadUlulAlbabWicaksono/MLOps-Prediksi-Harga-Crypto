import sys
import dagshub
import mlflow
from mlflow import MlflowClient

# DIUBAH: Disamakan menjadi MLOps-Prediksi-Harga-Crypto
dagshub.init(repo_owner='MuhammadUlulAlbabWicaksono', repo_name='MLOps-Prediksi-Harga-Crypto', mlflow=True)

def evaluate_and_register():
    print("\n--- MEMULAI EVALUASI & AUTO-REGISTRY ---")
    client = MlflowClient()
    experiment_name = "BTC_Pipeline_Production"
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

    rmse = metrics.get("rmse", float('inf'))
    r2 = metrics.get("r2", -float('inf'))

    print(f"Run ID: {run_id} | RMSE: {rmse:.2f} | R2: {r2:.2f}")

    if rmse < 250.0 and r2 > 0.30:
        print("\n[LULUS] Model memenuhi threshold.")
        
        model_uri = f"runs:/{run_id}/xgboost-model"
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