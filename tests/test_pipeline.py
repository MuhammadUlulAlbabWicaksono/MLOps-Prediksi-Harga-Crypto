import os
import glob
import pandas as pd
import pytest

def get_latest_data():
    """Mengambil file dataset terproses paling baru."""
    processed_files = glob.glob("data/processed/btc_processed_*.csv")
    if not processed_files:
        return None
    return max(processed_files, key=os.path.getctime)

def test_data_exists():
    """Verifikasi eksistensi file dataset terproses."""
    data_path = get_latest_data()
    assert data_path is not None, "File dataset tidak ditemukan di folder data/processed/"
    assert os.path.exists(data_path), f"File tidak valid: {data_path}"

def test_data_preprocessing():
    """Verifikasi integritas dataset sebelum pelatihan."""
    data_path = get_latest_data()
    assert data_path is not None, "Membatalkan tes: Dataset kosong."
    
    df = pd.read_csv(data_path)
    
    # Verifikasi dimensi (minimal baris terisi dan kolom fitur teknikal ada)
    assert df.shape[0] > 0, "Dataset kosong."
    assert df.shape[1] >= 5, "Kolom fitur esensial kurang."
    
    # Verifikasi nilai kosong (mencegah error XGBoost)
    null_counts = df.isnull().sum().sum()
    assert null_counts == 0, f"Ditemukan {null_counts} nilai null dalam dataset."