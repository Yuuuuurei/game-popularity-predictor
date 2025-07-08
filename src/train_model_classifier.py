import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.preprocessing import preprocess, feature_engineering

SEED = 42


def bin_popularity(y, n_bins=3):
    """Buat label klasifikasi dari target numerik (EstimatedOwnersAvg)."""
    bins = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    y_binned = bins.fit_transform(y.values.reshape(-1, 1)).astype(int).flatten()
    return y_binned, bins


def train_classification_models(X, y):
    y_cls, bin_encoder = bin_popularity(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_cls, test_size=0.2, random_state=SEED, stratify=y_cls)

    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForest": RandomForestClassifier(random_state=SEED),
        "ExtraTrees": ExtraTreesClassifier(random_state=SEED),
        "SVM": SVC(random_state=SEED),
        "KNeighbors": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=SEED),
        "LGBM": LGBMClassifier(random_state=SEED),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=SEED)
    }

    best_model = None
    best_score = 0

    print("🔍 Model Comparison (Accuracy):")
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name:<20}: Accuracy = {acc:.4f}")

        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name

    print(f"\n🏆 Best Model: {best_name} with Accuracy = {best_score:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, best_model.predict(X_test)))

    joblib.dump(best_model, "model/best_classifier.pkl")
    joblib.dump(bin_encoder, "model/popularity_binner.pkl")

    return best_model, bin_encoder


if __name__ == "__main__":
    df_raw = pd.read_csv("data/games_selected.csv")
    df_clean = preprocess(df_raw, use_text_embedding=True)
    X, y = feature_engineering(df_clean, use_embedding=True)
    train_classification_models(X, y)
