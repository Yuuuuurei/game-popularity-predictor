import os
import json
import pandas as pd

def align_features(X: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    """
    Menyesuaikan dataframe X agar memiliki kolom yang sesuai dengan expected_columns:
    - Menambahkan kolom yang hilang dengan nilai 0.
    - Menghapus kolom yang tidak dikenal.
    - Mengurutkan kolom sesuai expected_columns.
    """
    X = X.copy()

    # Tambah kolom yang tidak ada
    for col in expected_columns:
        if col not in X.columns:
            X[col] = 0

    # Hanya ambil kolom yang ada di expected_columns
    X = X[[col for col in expected_columns]]

    return X

def save_expected_columns(columns: list, path: str = "model/expected_columns.json"):
    """
    Simpan daftar kolom fitur ke file JSON.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(columns, f)

def load_expected_columns(path: str = "model/expected_columns.json") -> list:
    """
    Muat daftar kolom fitur dari file JSON.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected columns file tidak ditemukan di {path}")
    
    with open(path, "r") as f:
        return json.load(f)