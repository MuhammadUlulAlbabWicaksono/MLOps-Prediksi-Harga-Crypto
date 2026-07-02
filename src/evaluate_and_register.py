import sys
import logging
import dagshub
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.entities import Run
from config import MLflowConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def initialize_tracking_server() -> None:
    """Menginisialisasi koneksi ke DagsHub dan MLflow Tracking Server."""
    dagshub.init(
        repo_owner=MLflowConfig.REPO_OWNER, 
        repo_name=MLflowConfig.REPO_NAME, 
        mlflow=True
    )

def get_latest_run(client: MlflowClient, experiment_name: str) -> Run:
    """Mengambil run pelatihan terakhir dari eksperimen yang ditentukan."""
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Eksperimen '{experiment_name}' tidak ditemukan di tracking server.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError(f"Riwayat Run kosong untuk eksperimen '{experiment_name}'.")
        
    return runs[0]

def evaluate_model_quality(run: Run) -> bool:
    """Mengevaluasi apakah metrik model memenuhi standar (Quality Gate)."""
    metrics = run.data.metrics
    rmse = metrics.get("rmse", float('inf'))
    r2 = metrics.get("r2", -float('inf'))
    
    logger.info(f"Run ID : {run.info.run_id}")
    logger.info(f"Metrik : RMSE = {rmse:.2f} | R2 = {r2:.2f}")

    threshold_rmse = MLflowConfig.THRESHOLDS.get("max_rmse", 400)
    threshold_r2 = MLflowConfig.THRESHOLDS.get("min_r2", 0.6)

    if rmse >= threshold_rmse or r2 <= threshold_r2:
        logger.error("Evaluasi GAGAL: Metrik tidak memenuhi standar threshold.")
        return False
        
    logger.info("Evaluasi LULUS: Model memenuhi standar threshold produksi.")
    return True

def register_new_model_version(client: MlflowClient, run: Run, model_name: str, artifact_path: str) -> str:
    """Mendaftarkan versi model baru menggunakan backend source URI secara langsung."""
    try:
        client.get_registered_model(model_name)
    except MlflowException:
        logger.info(f"Membuat registry baru untuk entitas model '{model_name}'")
        client.create_registered_model(model_name)
    
    source_uri = f"{run.info.artifact_uri}/{artifact_path}"
    logger.info(f"Mendaftarkan model dari backend source: {source_uri}")
    
    model_version = client.create_model_version(
        name=model_name,
        source=source_uri,
        run_id=run.info.run_id
    )
    
    return model_version.version

def assign_model_alias(client: MlflowClient, model_name: str, version: str, alias: str) -> None:
    """Memberikan alias (tag environment) pada versi model tertentu."""
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version
    )
    logger.info(f"REGISTRASI SUKSES: '{model_name}' versi {version} kini berstatus '{alias}'.")

def main() -> None:
    """Orkestrator utama untuk alur evaluasi dan registrasi pipeline CI/CD."""
    logger.info("=== MEMULAI EVALUASI & AUTO-REGISTRY ===")
    
    try:
        initialize_tracking_server()
        client = MlflowClient()
        
        # 1. Akuisisi Data Run
        latest_run = get_latest_run(client, MLflowConfig.EXPERIMENT_NAME)
        
        # 2. Gerbang Kualitas (Quality Gate)
        is_passed = evaluate_model_quality(latest_run)
        if not is_passed:
            sys.exit(1)
            
        # 3. Registrasi ke Model Registry
        version = register_new_model_version(
            client=client, 
            run=latest_run, 
            model_name=MLflowConfig.MODEL_NAME, 
            artifact_path=MLflowConfig.DEFAULT_ARTIFACT_PATH
        )
        
        # 4. Pelabelan Lingkungan
        assign_model_alias(
            client=client, 
            model_name=MLflowConfig.MODEL_NAME, 
            version=version, 
            alias="staging"
        )
        
    except MlflowException as e:
        logger.error(f"[MLFLOW ERROR] Terdapat kesalahan pada operasi API MLflow: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[SYSTEM ERROR] Terjadi kegagalan sistem internal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()