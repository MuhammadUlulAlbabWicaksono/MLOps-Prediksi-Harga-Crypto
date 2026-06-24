import os
import sys
import logging
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from config import MLflowConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# MURNI MLFLOW: Membaca URI langsung dari sistem GitHub Actions
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

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
            logger.error("Evaluasi GAGAL: Metrik tidak memenuhi standar minimum.")
            sys.exit(1)
            
        logger.info("Evaluasi LULUS: Model memenuhi standar threshold produksi.")

        # Resolusi URI menggunakan Single Source of Truth
        model_uri = f"runs:/{run_id}/{MLflowConfig.DEFAULT_ARTIFACT_PATH}"
        logger.info(f"Mencoba meregistrasi model dari URI target: {model_uri}")
        
        model_version = mlflow.register_model(
            model_uri=model_uri, 
            name=MLflowConfig.MODEL_NAME
        )
        
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