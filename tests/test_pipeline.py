import os
import pandas as pd
import pytest

DATA_PATH = "data/raw/bitcoin_data.csv"

def test_raw_data_exists():
    """Verifikasi eksistensi file dataset."""
    assert os.path.exists(DATA_PATH), f"File tidak ditemukan: {DATA_PATH}"

def test_data_preprocessing():
    """Verifikasi integritas dataset."""
    df = pd.read_csv(DATA_PATH)
    
    # Verifikasi dimensi
    assert df.shape[0] > 0, "Dataset kosong."
    assert df.shape[1] >= 5, "Kolom fitur kurang."
    
    # Verifikasi nilai kosong
    null_counts = df.isnull().sum().sum()
    assert null_counts == 0, f"Ditemukan {null_counts} nilai null."