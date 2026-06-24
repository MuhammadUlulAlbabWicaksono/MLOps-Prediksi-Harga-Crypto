import sys
import logging
import dagshub
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from config import MLflowConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

dagshub.init(repo_owner=MLflowConfig.REPO_OWNER, repo_name=MLflowConfig.REPO_NAME, mlflow=True)

def evaluate_and_register():
    logger.info("=== MEMULAI EVALUASI & AUTO-REGISTRY ===")
    client = MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name(MLflowConfig.EXPERIMENT_NAME)
        if not experiment:
            raise ValueError(f"Eksperimen '{MLflowConfig.EXPERIMENT_NAME}' tidak ditemukan.")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        if not runs:
            raise ValueError("Riwayat Run kosong.")

        latest_run = runs[0]
        run_id = latest_run.info.run_id
        metrics = latest_run.data.metrics
        
        rmse = metrics.get("rmse", float('inf'))
        r2 = metrics.get("r2", -float('inf'))
        logger.info(f"Run ID : {run_id}")
        logger.info(f"Metrik : RMSE = {rmse:.2f} | R2 = {r2:.2f}")

        if rmse >= MLflowConfig.THRESHOLDS["max_rmse"] or r2 <= MLflowConfig.THRESHOLDS["min_r2"]:
            logger.error("Evaluasi GAGAL: Metrik tidak memenuhi standar.")
            sys.exit(1)
            
        logger.info("Evaluasi LULUS: Model memenuhi standar threshold produksi.")

        # --- SOLUSI FUNDAMENTAL: BYPASS DOWNLOAD VALIDATION ---
        # Menggunakan MlflowClient API low-level untuk menghindari bug download direktori
        
        # 1. Pastikan model container (Registry) sudah dibuat
        try:
            client.get_registered_model(MLflowConfig.MODEL_NAME)
        except MlflowException:
            logger.info(f"Membuat registry baru untuk '{MLflowConfig.MODEL_NAME}'")
            client.create_registered_model(MLflowConfig.MODEL_NAME)
        
        # 2. Susun absolute source URI murni dari server
        source_uri = f"{latest_run.info.artifact_uri}/{MLflowConfig.DEFAULT_ARTIFACT_PATH}"
        logger.info(f"Mendaftarkan model langsung dari backend source: {source_uri}")
        
        # 3. Create model version (Langsung perintah ke backend, tanpa download lokal!)
        model_version = client.create_model_version(
            name=MLflowConfig.MODEL_NAME,
            source=source_uri,
            run_id=run_id
        )
        
        # 4. Set Alias Staging
        client.set_registered_model_alias(
            name=MLflowConfig.MODEL_NAME,
            alias="staging",
            version=model_version.version
        )
        logger.info(f"REGISTRASI SUKSES: '{MLflowConfig.MODEL_NAME}' versi {model_version.version} berstatus 'staging'.")

    except MlflowException as e:
        logger.error(f"[MLFLOW ERROR] Terdapat kesalahan pada operasi API MLflow.")
        logger.error(f"Detail: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[SYSTEM ERROR] Terjadi kegagalan sistem internal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    evaluate_and_register()