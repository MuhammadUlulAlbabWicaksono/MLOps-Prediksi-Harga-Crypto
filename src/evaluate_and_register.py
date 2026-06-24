import sys
import json
import logging
import dagshub
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from config import MLflowConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

dagshub.init(repo_owner=MLflowConfig.REPO_OWNER, repo_name=MLflowConfig.REPO_NAME, mlflow=True)

def get_dynamic_artifact_path(run_tags):
    """
    Handling Programatik: 
    Membaca artifact_path secara dinamis dari metadata bawaan MLflow.
    Jika metadata tidak ditemukan, fallback ke konstanta config.
    """
    history_json = run_tags.get("mlflow.log-model.history")
    if history_json:
        try:
            history = json.loads(history_json)
            # Mengambil path dari model pertama yang di-log
            return history[0].get("artifact_path", MLflowConfig.DEFAULT_ARTIFACT_PATH)
        except (json.JSONDecodeError, IndexError, KeyError):
            logger.warning("Gagal mem-parsing log-model.history. Menggunakan default config.")
    
    return MLflowConfig.DEFAULT_ARTIFACT_PATH

def evaluate_and_register():
    logger.info("=== MEMULAI EVALUASI & AUTO-REGISTRY ===")
    client = MlflowClient()
    
    try:
        # 1. Mendapatkan Eksperimen
        experiment = client.get_experiment_by_name(MLflowConfig.EXPERIMENT_NAME)
        if not experiment:
            raise ValueError(f"Eksperimen '{MLflowConfig.EXPERIMENT_NAME}' tidak ditemukan di server.")

        # 2. Mendapatkan Run Terbaru
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        if not runs:
            raise ValueError("Riwayat Run kosong. Pastikan proses training berjalan sebelum skrip ini.")

        latest_run = runs[0]
        run_id = latest_run.info.run_id
        metrics = latest_run.data.metrics
        
        # 3. Ekstraksi Metrik
        rmse = metrics.get("rmse", float('inf'))
        r2 = metrics.get("r2", -float('inf'))
        logger.info(f"Run ID : {run_id}")
        logger.info(f"Metrik : RMSE = {rmse:.2f} | R2 = {r2:.2f}")

        # 4. Evaluasi Metrik
        if rmse >= MLflowConfig.THRESHOLDS["max_rmse"] or r2 <= MLflowConfig.THRESHOLDS["min_r2"]:
            logger.error("Evaluasi GAGAL: Metrik tidak memenuhi standar minimum untuk produksi.")
            sys.exit(1) # Graceful exit untuk menghentikan CI/CD pipeline
            
        logger.info("Evaluasi LULUS: Model memenuhi standar threshold produksi.")

        # 5. Resolusi Path Programatik & Pembuatan URI
        actual_artifact_path = get_dynamic_artifact_path(latest_run.data.tags)
        model_uri = f"runs:/{run_id}/{actual_artifact_path}"
        logger.info(f"Mencoba meregistrasi model dari URI target: {model_uri}")
        
        # 6. Registrasi Model
        model_version = mlflow.register_model(
            model_uri=model_uri, 
            name=MLflowConfig.MODEL_NAME
        )
        
        # 7. Automasi Stage ke Staging menggunakan sistem Alias
        client.set_registered_model_alias(
            name=MLflowConfig.MODEL_NAME,
            alias="staging",
            version=model_version.version
        )
        logger.info(f"REGISTRASI SUKSES: '{MLflowConfig.MODEL_NAME}' versi {model_version.version} resmi berstatus 'staging'.")

    except MlflowException as e:
        logger.error(f"[MLFLOW ERROR] Terdapat kesalahan pada operasi API MLflow.")
        logger.error(f"Pesan Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[SYSTEM ERROR] Terjadi kegagalan sistem internal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    evaluate_and_register()