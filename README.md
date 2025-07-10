# 🎮 Steam Game Popularity Predictor

Aplikasi berbasis machine learning untuk memprediksi tingkat popularitas game Steam berdasarkan fitur-fitur seperti harga, genre, tag, playtime, dan lainnya. Didukung dengan antarmuka Streamlit dan model CatBoost terlatih.

## 🔍 Fitur Utama

- Prediksi tingkat popularitas game: Kurang Populer, Cukup Populer, atau Sangat Populer
- Dukungan input file .csv untuk batch prediction
- Visualisasi prediksi:
  - Distribusi popularitas
  - Scatter plot: Harga vs Peak CCU
  - Top Tags
- Training model baru (CatBoost, LightGBM, XGBoost, dll.) dengan mudah
- Antarmuka dashboard interaktif menggunakan Streamlit

## 📦 Dataset

Dataset utama berasal dari HuggingFace:
🔗 [FronkonGames / Steam Games Dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset)
Dataset ini telah dimodifikasi secara manual untuk kebutuhan proyek, dengan penyesuaian sebagai berikut:

- Menghapus kolom-kolom besar yang tidak relevan (seperti `Header image`, `Screenshots`, `Movies`, dan lain lain)
- Mempertahankan hanya kolom yang berkaitan langsung dengan prediksi popularitas, seperti:
  - Informasi umum: `Name`, `Release date`, `Price`, `Required age`
  - Statistik pengguna: `Positive`, `Negative`, `Average playtime`, `Peak CCU`
  - Metadata konten: `Tags`, `Genres`, `Categories`, `Developers`, `Publishers`
  - Target kolom: `Estimated owners` (digunakan untuk membuat label kelas popularitas)
    📝 Dataset akhir ini digunakan untuk training dan evaluasi model klasifikasi popularitas game.

---

## 🧠 Model

Model utama yang digunakan:

- **CatBoostClassifier** (pre-trained, disimpan dalam folder `model/`)
- Label `Estimated owners` dibagi menjadi 3 kelas berdasarkan bins:

| Range Owner      | Kelas          |
| ---------------- | -------------- |
| 0 – 10.000       | Kurang Populer |
| 10.001 – 100.000 | Cukup Populer  |
| > 100.000        | Sangat Populer |

---

## 🚀 Cara Menjalankan

### Opsi 1: **Jalankan Dashboard (Prediksi)**

1. **Windows (gunakan file .bat)**

   ```
   run.bat
   ```

2. **macOS dan Linux (gunakan file .sh)**
   ```
   chmod +x run.sh
   ./run.sh
   ```

- Menjalankan executables tersebut akan:
  - Membuat virtual environment
  - Menginstall `requirements.txt`
  - Menjalankan dashboard Streamlit

---

### Opsi 2: **Jalankan Training**

1. **Windows (gunakan file .bat)**

   ```
   train.bat
   ```

2. **macOS dan Linux (gunakan file .sh)**
   ```
   chmod +x run.sh
   ./train.sh
   ```

---

## 📊 Evaluasi Model

| Model              | Akurasi    |
| ------------------ | ---------- |
| LogisticRegression | 89.88%     |
| RandomForest       | 91.25%     |
| ExtraTrees         | 89.25%     |
| NaiveBayes         | 84.40%     |
| XGBoost            | 91.83%     |
| LGBM               | 91.82%     |
| **CatBoost**       | **91.92%** |

---

## 📁 Upload CSV Anda

Format CSV yang valid:

- Name, Release date, Price, Peak CCU, Required age
- Positive, Negative, Average playtime forever, Median playtime forever
- Genres, Tags, Categories, Developers, Publishers

---

## 🖼️ Visualisasi Dashboard

- Distribusi Kelas Prediksi
- Harga vs Peak CCU
- Top 10 Tag Terpopuler

---

## 🧾 Catatan

- Pastikan file .csv memiliki format kolom yang benar (lihat bagian contoh).
- Pastikan model dan label encoder tersedia di folder model/ jika tidak melakukan training sendiri.

---

## 📦 Dependency

Daftar utama ada di requirements.txt:

```
catboost
lightgbm
xgboost
pandas
numpy
scikit-learn
seaborn
streamlit
```

---

🤝 Kontribusi
Pull request, saran fitur baru, dan perbaikan bug sangat terbuka.

---

## 📜 Lisensi

Proyek ini dilisensikan di bawah MIT License.
