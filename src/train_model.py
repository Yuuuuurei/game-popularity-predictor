import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.preprocessing import preprocess, feature_engineering

SEED = 42


def bin_manual_popularity(y):
    # Definisi batas-batas manual
    bins = [0, 10_000, 100_000, float("inf")]
    labels = ["Kurang Populer", "Cukup Populer", "Sangat Populer"]

    y_binned = pd.cut(y, bins=bins, labels=labels, include_lowest=True)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_binned)

    print("\n📊 Distribusi kelas target (manual binning):")
    for label in label_encoder.classes_:
        count = np.sum(y_binned == label)
        print(f"{label:<15}: {count} data")

    return y_encoded, label_encoder


def check_nan(X):
    print("\n📋 Cek NaN setelah feature_engineering:")
    nan_summary = X.isna().sum()
    if nan_summary.sum() > 0:
        print("⚠️ Terdapat NaN pada fitur berikut:")
        print(nan_summary[nan_summary > 0].sort_values(ascending=False))
        print(f"Total baris dengan minimal 1 NaN: {(X.isna().any(axis=1)).sum()}")
    else:
        print("✅ Tidak ada NaN setelah feature_engineering.")


def train_classification_models(X, y):
    y_cls, label_encoder = bin_manual_popularity(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.2, random_state=SEED, stratify=y_cls
    )

    classifiers = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=SEED))
        ]),
        "RandomForest": RandomForestClassifier(random_state=SEED),
        "ExtraTrees": ExtraTreesClassifier(random_state=SEED),
        "NaiveBayes": GaussianNB(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=SEED),
        "LGBM": LGBMClassifier(random_state=SEED),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=SEED)
    }

    best_model, best_score, best_name = None, 0, None

    print("\n🔍 Model Comparison (Accuracy):")
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name:<20}: Accuracy = {acc:.4f}")

        if acc > best_score:
            best_model, best_score, best_name = model, acc, name

    print(f"\n🏆 Best Model: {best_name} with Accuracy = {best_score:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(
        y_test,
        best_model.predict(X_test),
        target_names=label_encoder.classes_
    ))

    joblib.dump(best_model, f"model/popularity_classifier_{best_name.lower()}_manualbin3.joblib")
    joblib.dump(label_encoder, "model/popularity_label_encoder_manualbin3.joblib")
    joblib.dump(X.columns.tolist(), "model/feature_columns.joblib")

    print("💾 Model dan label encoder disimpan.")

    return best_model, label_encoder


if __name__ == "__main__":
    print("\n📦 Loading data")
    df_raw = pd.read_csv("data/games_selected.csv", low_memory=False)
    df_clean = preprocess(df_raw)
    X, y = feature_engineering(df_clean)

    check_nan(X)
    train_classification_models(X, y)
