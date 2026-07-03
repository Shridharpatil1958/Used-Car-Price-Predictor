"""
Preprocessing pipeline for the CarDekho Used Car Price Prediction project.

This module is the single source of truth for data cleaning and feature
engineering. It is imported by both train.py (to build the training set)
and predict.py / the Streamlit app (to transform a single new car the same way).

Design decisions:
  - Outliers (bad km_driven, seats==0, extreme prices) are DROPPED at
    training time only. At inference time we do NOT drop the incoming row;
    we just clip absurd values so the model doesn't extrapolate wildly.
  - Categorical encoding: brand/model are high-cardinality -> target
    (mean) encoding, fit on TRAIN ONLY to avoid leakage. Low-cardinality
    columns (fuel_type, seller_type, transmission_type) -> one-hot.
  - Target (selling_price) is modeled in log1p space; predict.py inverts
    this with expm1 before returning a price to the user.
"""
import os
import json
import numpy as np
import pandas as pd

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cardekho_dataset.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "encoders.json")

CURRENT_YEAR_REFERENCE = 2024  # dataset appears to be scraped around this year; vehicle_age is already provided

NUMERIC_FEATURES = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats", "km_per_year"]
ONEHOT_FEATURES = ["fuel_type", "seller_type", "transmission_type"]
TARGET_ENCODE_FEATURES = ["brand", "model"]
TARGET_COL = "selling_price"


def load_raw(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and drop the stray pandas index column if present."""
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def clean(df: pd.DataFrame, drop_outliers: bool = True) -> pd.DataFrame:
    """Remove duplicates and (optionally) drop rows with clearly invalid values.

    drop_outliers=True is used for training data.
    drop_outliers=False is used for a single inference row (we clip instead, see clip_inputs).
    """
    df = df.drop_duplicates().reset_index(drop=True)

    if drop_outliers:
        before = len(df)
        df = df[df["seats"] > 0]
        df = df[df["km_driven"] <= 300_000]  # beyond this is almost certainly a data-entry error
        df = df[df["max_power"] > 0]
        # Keep prices up to the 99.5th percentile to avoid ultra-luxury cars dominating the loss
        price_cap = df["selling_price"].quantile(0.995)
        df = df[df["selling_price"] <= price_cap]
        after = len(df)
        print(f"clean(): dropped {before - after} rows ({before} -> {after})")

    return df.reset_index(drop=True)


def clip_inputs(row: dict) -> dict:
    """Clip a single inference input to sane ranges instead of rejecting it."""
    row = dict(row)
    row["km_driven"] = min(max(row.get("km_driven", 0), 0), 300_000)
    row["vehicle_age"] = min(max(row.get("vehicle_age", 0), 0), 25)
    row["seats"] = min(max(row.get("seats", 5), 2), 9)
    row["max_power"] = max(row.get("max_power", 1), 1.0)
    row["engine"] = max(row.get("engine", 500), 500)
    row["mileage"] = max(row.get("mileage", 5), 5.0)
    return row


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Assumes vehicle_age and km_driven already exist."""
    df = df.copy()
    # Avoid divide-by-zero for brand-new cars (age 0)
    df["km_per_year"] = df["km_driven"] / df["vehicle_age"].replace(0, 1)
    return df


def fit_target_encoders(df: pd.DataFrame, cols: list, target_col: str = TARGET_COL) -> dict:
    """Compute mean log-target per category for each high-cardinality column.

    Fit on the TRAIN split only. Returns a dict: {column: {category: mean_log_price}}
    plus a global fallback mean for unseen categories at inference time.
    """
    encoders = {}
    log_target = np.log1p(df[target_col])
    global_mean = float(log_target.mean())
    for col in cols:
        means = df.assign(_log_target=log_target).groupby(col)["_log_target"].mean()
        encoders[col] = {"map": means.to_dict(), "fallback": global_mean}
    return encoders


def apply_target_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col, enc in encoders.items():
        fallback = enc["fallback"]
        df[f"{col}_enc"] = df[col].map(enc["map"]).fillna(fallback)
    return df


def save_encoders(encoders: dict, path: str = ENCODERS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(encoders, f, indent=2)


def load_encoders(path: str = ENCODERS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def build_model_matrix(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Turn a cleaned, feature-engineered dataframe into the final numeric
    matrix the model expects: numeric features + target-encoded brand/model
    + one-hot encoded low-cardinality categoricals.
    """
    df = apply_target_encoders(df, encoders)

    onehot = pd.get_dummies(df[ONEHOT_FEATURES], prefix=ONEHOT_FEATURES)

    encoded_cat_cols = [f"{c}_enc" for c in TARGET_ENCODE_FEATURES]
    X = pd.concat([df[NUMERIC_FEATURES], df[encoded_cat_cols], onehot], axis=1)
    return X


def align_columns(X: pd.DataFrame, reference_columns: list) -> pd.DataFrame:
    """Ensure a feature matrix has exactly the columns the model was trained on
    (adds missing one-hot columns as 0, drops extras, fixes order). Needed at
    inference time since a single row won't naturally produce every dummy column.
    """
    X = X.reindex(columns=reference_columns, fill_value=0)
    return X


def full_training_pipeline(path: str = RAW_DATA_PATH):
    """Convenience function: raw CSV -> cleaned, feature-engineered df ready to split."""
    df = load_raw(path)
    df = clean(df, drop_outliers=True)
    df = engineer_features(df)
    return df
