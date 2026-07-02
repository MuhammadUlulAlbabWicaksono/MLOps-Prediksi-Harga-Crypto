import sys
import logging
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
from config import MLflowConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = MLflowConfig.MODEL_NAME

def evaluate_and_promote():
    logger.info("Menginisialisasi koneksi ke DagsHub...")
    dagshub.init(
        repo_owner=MLflowConfig.REPO_OWNER, 
        repo_name=MLflowConfig.REPO_NAME, 
        mlflow=True
    )
    
    client = MlflowClient()
    
    try:
        # 1. Ambil SEMUA versi model sekaligus (Bypass bug API yang deprecated)
        logger.info(f"Mengambil seluruh riwayat versi model '{MODEL_NAME}'...")
        all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        
        if not all_versions:
            logger.error("GAGAL: Model belum terdaftar sama sekali di DagsHub.")
            sys.exit(1)

        # Urutkan dari versi terbaru (angka terbesar) ke terlama
        all_versions_sorted = sorted(all_versions, key=lambda v: int(v.version), reverse=True)

        # 2. Cari Champion (Model dengan stage/alias Production, ATAU ambil versi paling awal jika belum ada)
        champion_version = None
        for v in all_versions_sorted:
            # Mencocokkan baik sistem lama (stages) maupun sistem baru (aliases)
            if v.current_stage.lower() == "production" or "Production" in v.aliases or "production" in v.aliases:
                champion_version = v
                break
                
        if not champion_version:
            logger.warning("Label 'Production' tidak ditemukan. Menggunakan versi TERLAMA sebagai Champion sementara.")
            champion_version = all_versions_sorted[-1] # Menjadikan V1 sebagai Champion default

        # 3. Cari Challenger (Model versi terbaru yang BUKAN Champion)
        challenger_version = None
        for v in all_versions_sorted:
            if v.version != champion_version.version:
                challenger_version = v
                break
                
        if not challenger_version:
            logger.error("GAGAL: Hanya ada 1 versi model di registry. CT Pipeline membutuhkan minimal 2 versi (Model Lama vs Model Baru).")
            sys.exit(1)

        # 4. Tarik Metrik Kedua Model
        logger.info(f"Champion terpilih: Version {champion_version.version}")
        champion_run = client.get_run(champion_version.run_id)
        champion_rmse = champion_run.data.metrics.get("rmse", float('inf'))
        champion_r2 = champion_run.data.metrics.get("r2", float('-inf'))
        logger.info(f"Metrik Champion: RMSE = {champion_rmse:.2f}, R2 = {champion_r2:.2f}")

        logger.info(f"Challenger terpilih: Version {challenger_version.version}")
        challenger_run = client.get_run(challenger_version.run_id)
        challenger_rmse = challenger_run.data.metrics.get("rmse", float('inf'))
        challenger_r2 = challenger_run.data.metrics.get("r2", float('-inf'))
        logger.info(f"Metrik Challenger: RMSE = {challenger_rmse:.2f}, R2 = {challenger_r2:.2f}")

        # 5. Logika Komparasi
        is_rmse_better = challenger_rmse < champion_rmse
        is_r2_better = challenger_r2 > champion_r2

        if is_rmse_better and is_r2_better:
            logger.info("Challenger outperforms Champion! Initiating promotion...")
            # Gunakan Aliases (Metode Modern MLflow)
            client.set_registered_model_alias(MODEL_NAME, "Production", challenger_version.version)
            client.set_registered_model_alias(MODEL_NAME, "Archived", champion_version.version)
            logger.info("Promotion successful. New Champion installed via Aliases.")
        else:
            logger.info("Challenger failed to outperform Champion. Discarding Challenger.")
            client.set_registered_model_alias(MODEL_NAME, "Archived", challenger_version.version)

    except Exception as e:
        logger.error(f"Terjadi kesalahan sistem: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    evaluate_and_promote()