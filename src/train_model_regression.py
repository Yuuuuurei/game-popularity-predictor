import pandas as pd
import numpy as np
import joblib
import os

from sklearn import linear_model, ensemble
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

import lightgbm as lgbm
import xgboost as xgb
import catboost as cb

from src.preprocessing import preprocess, feature_engineering

SEED = 42

regressors = {
    'LinearRegression': linear_model.LinearRegression(),
    'Ridge': linear_model.Ridge(random_state=SEED),
    'Lasso': linear_model.Lasso(random_state=SEED),
    'ElasticNet': linear_model.ElasticNet(random_state=SEED),
    'CatBoostRegressor': cb.CatBoostRegressor(random_state=SEED, verbose=0),
    'LGBMRegressor': lgbm.LGBMRegressor(random_state=SEED, verbose=-1),
    'XGBRegressor': xgb.XGBRegressor(random_state=SEED, verbosity=0),
    'ExtraTreesRegressor': ensemble.ExtraTreesRegressor(random_state=SEED, n_jobs=-1),
    'RandomForestRegressor': ensemble.RandomForestRegressor(random_state=SEED, n_jobs=-1),
}

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    return rmse, r2

def train_and_evaluate_models(df_path, use_text_embedding=False):
    print("📦 Loading and preprocessing data...")
    df_raw = pd.read_csv(df_path)
    df_clean = preprocess(df_raw, use_text_embedding=use_text_embedding)
    X, y = feature_engineering(df_clean, use_embedding=use_text_embedding)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    print("\n🔍 Training and evaluating models...")
    results = []
    for name, model in regressors.items():
        print(f"Training {name}...")
        rmse, r2 = evaluate_model(model, X_train, X_test, y_train, y_test)
        results.append((name, rmse, r2))

    results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2"])
    results_df = results_df.sort_values(by="RMSE")
    print("\n✅ Model Evaluation Results:")
    print(results_df)

    # Save best model
    best_model_name = results_df.iloc[0]["Model"]
    best_model = regressors[best_model_name].fit(X, y)
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, f"model/{best_model_name}_model.joblib")
    print(f"\n💾 Best model saved to model/{best_model_name}_model.joblib")

if __name__ == "__main__":
    train_and_evaluate_models("data/games_selected.csv", use_text_embedding=True)