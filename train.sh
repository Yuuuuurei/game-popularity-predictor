#!/bin/bash

echo "----------------------------------------"
echo "Mengecek dan membuat virtual environment..."
echo "----------------------------------------"

# Buat virtual environment jika belum ada
if [ ! -d ".env" ]; then
    echo "Membuat virtual environment..."
    python3 -m venv .env
fi

# Aktifkan virtual environment
echo "Mengaktifkan virtual environment..."
source .env/bin/activate

# Cek apakah pip tersedia
if ! command -v pip &> /dev/null; then
    echo "pip tidak ditemukan. Pastikan Python sudah terinstal."
    exit 1
fi

# Install dependencies
echo "----------------------------------------"
echo "Menginstall dependencies dari requirements.txt..."
echo "----------------------------------------"
pip install -r requirements.txt

# Jalankan Train dengan PYTHONPATH=.
echo "----------------------------------------"
echo "Menjalankan Train Model..."
echo "----------------------------------------"

PYTHONPATH=. python src/train_model.py

echo
echo "----------------------------------------"
echo "Train selesai dijalankan."
