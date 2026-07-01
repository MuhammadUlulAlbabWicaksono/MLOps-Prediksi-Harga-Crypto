import mlflow
from mlflow.tracking import MlflowClient
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "BTC_Price_Predictor"

def evaluate_and_promote():
    client = MlflowClient()
    
    try:
        # 1. Dapatkan model Champion (Production) saat ini
        champion_version = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
        champion_run = client.get_run(champion_version.run_id)
        champion_rmse = champion_run.data.metrics.get("rmse", float('inf'))
        champion_r2 = champion_run.data.metrics.get("r2", float('-inf'))
        
        logger.info(f"Champion (Version {champion_version.version}): RMSE = {champion_rmse}, R2 = {champion_r2}")

        # 2. Dapatkan model Challenger (Terbaru, Staging/None)
        challenger_version = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging"])[0]
        challenger_run = client.get_run(challenger_version.run_id)
        challenger_rmse = challenger_run.data.metrics.get("rmse", float('inf'))
        challenger_r2 = challenger_run.data.metrics.get("r2", float('-inf'))
        
        logger.info(f"Challenger (Version {challenger_version.version}): RMSE = {challenger_rmse}, R2 = {challenger_r2}")

        # 3. Logika Komparasi Matematika
        # Challenger harus memiliki Error lebih rendah DAN R2 lebih tinggi
        is_rmse_better = challenger_rmse < champion_rmse
        is_r2_better = challenger_r2 > champion_r2

        if is_rmse_better and is_r2_better:
            logger.info("Challenger outperforms Champion. Initiating promotion to Production...")
            # Demote Champion lama ke Archived
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=champion_version.version,
                stage="Archived"
            )
            # Promote Challenger baru ke Production
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=challenger_version.version,
                stage="Production"
            )
            logger.info("Promotion successful. New Champion installed.")
        else:
            logger.info("Challenger failed to outperform Champion. Discarding Challenger.")
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=challenger_version.version,
                stage="Archived"
            )

    except IndexError:
        logger.error("Could not find required model versions (Champion/Challenger) in MLflow registry.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during evaluation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    evaluate_and_promote()