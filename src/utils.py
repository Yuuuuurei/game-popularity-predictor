import joblib
import numpy as np
import pandas as pd

from src.preprocessing import preprocess, feature_engineering


def load_model(model_path="model/best_model.pkl"):
    return joblib.load(model_path)

def load_binner(binner_path="model/popularity_binner.pkl"):
    try:
        return joblib.load(binner_path)
    except FileNotFoundError:
        return None

def prepare_features(input_df, use_embedding=True):
    df_clean = preprocess(input_df, use_text_embedding=use_embedding)
    X, _ = feature_engineering(df_clean, use_embedding=use_embedding)
    return X

def is_classifier_model(model_path: str):
    return "classifier" in model_path.lower()

def supports_predict_proba(model):
    return hasattr(model, "predict_proba")
    

def predict_popularity(input_df, model, binner=None, use_embedding=True, return_proba=False):
    X = prepare_features(input_df, use_embedding=use_embedding)

    if is_classifier_model(model.__class__.__name__) or hasattr(model, "predict_proba"):
        preds = model.predict(X)
        proba = model.predict_proba(X) if return_proba and supports_predict_proba(model) else None
        if binner and hasattr(binner, "inverse_transform"):
            labels = binner.inverse_transform(preds.reshape(-1, 1)).flatten()
        else:
            labels = preds
        return preds, labels, proba
    else:
        preds = model.predict(X)
        return preds, preds, None  # regresi: label = prediksi numerik
