import pandas as pd
import joblib
import os

from src.preprocessing import preprocess, feature_engineering

# Paths ke model dan label encoder
MODEL_PATH = "model/popularity_classifier_catboost_manualbin3.joblib"
ENCODER_PATH = "model/popularity_label_encoder_manualbin3.joblib"
FEATURE_COLS_PATH = "model/feature_columns.joblib"

def load_model():
    if not all(os.path.exists(p) for p in [MODEL_PATH, ENCODER_PATH, FEATURE_COLS_PATH]):
        raise FileNotFoundError("Model, label encoder, atau feature_columns tidak ditemukan.")

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    feature_columns = joblib.load(FEATURE_COLS_PATH)
    return model, label_encoder, feature_columns

def align_features(X, expected_columns):
    # Reindex untuk menghindari fragmentasi dan warning
    return X.reindex(columns=expected_columns, fill_value=0)

def predict_popularity(input_data: pd.DataFrame):
    """
    input_data: DataFrame dengan format kolom seperti data mentah (`games_selected.csv`)
    """
    # Preprocessing + feature engineering
    clean_df = preprocess(input_data)
    X, _ = feature_engineering(clean_df)

    # Load model dan encoder
    model, label_encoder, expected_columns = load_model()

    # Align features
    X = align_features(X, expected_columns)

    # Prediksi
    preds = model.predict(X)
    preds_label = label_encoder.inverse_transform(preds)

    # Tambahkan hasil ke dataframe
    result_df = input_data.copy()
    result_df["Predicted Popularity Class"] = preds_label

    return result_df[["Name", "Predicted Popularity Class"]]

# Untuk testing mandiri
if __name__ == "__main__":
    sample_df = pd.read_csv("data/test.csv")  # berisi 1–5 baris game baru
    result = predict_popularity(sample_df)
    print(result)