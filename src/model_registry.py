import os
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import yaml

def main():
    print("Memulai proses Model Registry...\n")
    client = MlflowClient()
    experiment_name = "BTC_Price_Prediction_XGBoost"
    model_name = "BTC_Price_Predictor"
    
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Eksperimen '{experiment_name}' tidak ditemukan. Pastikan LK-06 sudah dijalankan.")
        
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"]
    )
    
    run_terbaik = runs[0]
    run_kedua_terbaik = runs[1]
    
    print("=== 1. REGISTRASI & VERSIONING MODEL ===")
    print(f"Mendaftarkan Run ID {run_kedua_terbaik.info.run_id} ke registry...")
    mv1 = mlflow.register_model(model_uri=f"runs:/{run_kedua_terbaik.info.run_id}/xgboost-model", name=model_name)
    
    print(f"Mendaftarkan Run ID {run_terbaik.info.run_id} ke registry...")
    mv2 = mlflow.register_model(model_uri=f"runs:/{run_terbaik.info.run_id}/xgboost-model", name=model_name)
    
    print("\n=== 2. TRANSISI STAGE MODEL ===")
    print(f"Mengubah status Versi {mv1.version} menjadi 'Staging'...")
    client.transition_model_version_stage(name=model_name, version=mv1.version, stage="Staging")
    
    print(f"Mengubah status Versi {mv2.version} menjadi 'Production'...")
    client.transition_model_version_stage(name=model_name, version=mv2.version, stage="Production")
    
    print("\n=== 3. SINKRONISASI METADATA UNTUK DVC ===")
    metadata = {
        "model_name": model_name,
        "production_version": int(mv2.version),
        "run_id": run_terbaik.info.run_id,
        "metrics": {
            "rmse": run_terbaik.data.metrics["rmse"],
            "r2_score": run_terbaik.data.metrics["r2"]
        }
    }
    
    # Memastikan file disimpan di root direktori (sejajar dengan README.md)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(root_dir, "model_metadata.yaml")
    
    with open(yaml_path, "w") as f:
        yaml.dump(metadata, f, sort_keys=False)
    print(f"File metadata berhasil dibuat di: {yaml_path}")
    
    print("\n=== 4. VERIFIKASI INFERENSI (SIMULASI API) ===")
    print("Memuat model dengan stage 'Production' dari Registry...")
    model_uri = f"models:/{model_name}/Production"
    production_model = mlflow.pyfunc.load_model(model_uri)
    
    dummy_input = pd.DataFrame({
        'price': [64000], 'market_cap': [1.2e12], 'volume': [3.5e10],
        'MA_7': [63500], 'MA_30': [62000], 'RSI_14': [55],
        'MACD': [150], 'MACD_Signal': [140]
    })
    
    prediksi = production_model.predict(dummy_input)
    print(f">> Hasil Prediksi Harga Berikutnya: ${prediksi[0]:.2f}")
    print("Verifikasi SUKSES! Model siap untuk deployment.")

if __name__ == "__main__":
    main()