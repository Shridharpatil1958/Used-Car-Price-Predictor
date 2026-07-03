"""
Model training for the CarDekho Used Car Price Prediction project.

Trains and compares:
  1. Linear Regression (baseline)
  2. Random Forest Regressor
  3. XGBoost Regressor
Then tunes the best-performing model with RandomizedSearchCV and saves:
  - models/best_model.pkl        (trained sklearn/xgboost estimator)
  - models/encoders.json         (target-encoding maps for brand/model)
  - models/feature_columns.json  (exact column order the model expects)
  - models/metrics.json          (comparison table + final metrics)

Run:
    python src/train.py
"""
import os
import json
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from preprocess import (
    full_training_pipeline,
    fit_target_encoders,
    build_model_matrix,
    save_encoders,
    TARGET_COL,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(model, X_test, y_test_log):
    """Evaluate a model trained on log-target. Returns metrics in ORIGINAL price scale
    (more interpretable than log-scale RMSE/MAE for a business audience)."""
    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)
    true = np.expm1(y_test_log)
    return {
        "RMSE": rmse(true, pred),
        "MAE": float(mean_absolute_error(true, pred)),
        "R2": float(r2_score(true, pred)),
    }


def main():
    print("Loading and cleaning data...")
    df = full_training_pipeline()
    print(f"Training data shape after cleaning: {df.shape}")

    # Train/test split BEFORE fitting encoders, to avoid target leakage
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Train: {train_df.shape[0]} rows | Test: {test_df.shape[0]} rows")

    # Fit target encoders on TRAIN ONLY
    encoders = fit_target_encoders(train_df, cols=["brand", "model"], target_col=TARGET_COL)
    save_encoders(encoders)

    X_train = build_model_matrix(train_df, encoders)
    X_test = build_model_matrix(test_df, encoders)

    # Align test columns to train columns (in case a category produced a onehot
    # column in one split but not the other)
    feature_columns = list(X_train.columns)
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)

    y_train_log = np.log1p(train_df[TARGET_COL].values)
    y_test_log = np.log1p(test_df[TARGET_COL].values)

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}

    # ---- 1. Linear Regression (baseline) ----
    print("\nTraining Linear Regression (baseline)...")
    lr = LinearRegression()
    cv_scores = cross_val_score(lr, X_train, y_train_log, cv=kfold, scoring="neg_root_mean_squared_error")
    lr.fit(X_train, y_train_log)
    results["LinearRegression"] = {
        "cv_rmse_log": float(-cv_scores.mean()),
        **evaluate(lr, X_test, y_test_log),
    }

    # ---- 2. Random Forest ----
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=300, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(rf, X_train, y_train_log, cv=kfold, scoring="neg_root_mean_squared_error")
    rf.fit(X_train, y_train_log)
    results["RandomForest"] = {
        "cv_rmse_log": float(-cv_scores.mean()),
        **evaluate(rf, X_test, y_test_log),
    }

    # ---- 3. XGBoost ----
    print("Training XGBoost...")
    xgb = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    cv_scores = cross_val_score(xgb, X_train, y_train_log, cv=kfold, scoring="neg_root_mean_squared_error")
    xgb.fit(X_train, y_train_log)
    results["XGBoost"] = {
        "cv_rmse_log": float(-cv_scores.mean()),
        **evaluate(xgb, X_test, y_test_log),
    }

    print("\n--- Model comparison (test set, original price scale) ---")
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df[["cv_rmse_log", "RMSE", "MAE", "R2"]]
    print(comparison_df.round(4))

    # Pick best model by test RMSE (original price scale)
    best_name = comparison_df["RMSE"].idxmin()
    print(f"\nBest model before tuning: {best_name}")

    # ---- Hyperparameter tuning on the best model (XGBoost is expected winner) ----
    if best_name == "XGBoost":
        print("\nTuning XGBoost with RandomizedSearchCV...")
        param_dist = {
            "n_estimators": [200, 300, 400, 600],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
        }
        search = RandomizedSearchCV(
            XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            param_distributions=param_dist,
            n_iter=25,
            cv=kfold,
            scoring="neg_root_mean_squared_error",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )
        t0 = time.time()
        search.fit(X_train, y_train_log)
        print(f"Tuning took {time.time() - t0:.1f}s")
        best_model = search.best_estimator_
        print(f"Best params: {search.best_params_}")
    elif best_name == "RandomForest":
        print("\nTuning Random Forest with RandomizedSearchCV...")
        param_dist = {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        search = RandomizedSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            param_distributions=param_dist,
            n_iter=15,
            cv=kfold,
            scoring="neg_root_mean_squared_error",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train_log)
        best_model = search.best_estimator_
        print(f"Best params: {search.best_params_}")
    else:
        best_model = lr

    final_metrics = evaluate(best_model, X_test, y_test_log)
    print(f"\n--- Final tuned model ({best_name}) test metrics ---")
    print(json.dumps(final_metrics, indent=2))

    # Residual std in log space -> used by predict.py to build an approximate price range
    pred_log = best_model.predict(X_test)
    residual_std_log = float(np.std(y_test_log - pred_log))

    # Save everything
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(
            {
                "comparison": comparison_df.round(4).to_dict(orient="index"),
                "best_model": best_name,
                "final_test_metrics": final_metrics,
                "residual_std_log": residual_std_log,
            },
            f,
            indent=2,
        )

    print(f"\nSaved: models/best_model.pkl, models/encoders.json, "
          f"models/feature_columns.json, models/metrics.json")


if __name__ == "__main__":
    main()
