"""
Inference module for the CarDekho Used Car Price Prediction project.

Loads the trained model + encoders once, and exposes predict_price() which
takes a raw dict of car attributes (as a user would fill in a form) and
returns a predicted price plus an approximate range.

Usage:
    from predict import predict_price
    result = predict_price({
        "brand": "Maruti", "model": "Swift", "vehicle_age": 4,
        "km_driven": 35000, "seller_type": "Dealer", "fuel_type": "Petrol",
        "transmission_type": "Manual", "mileage": 20.5, "engine": 1197,
        "max_power": 85.0, "seats": 5,
    })
"""
import os
import json
import numpy as np
import pandas as pd
import joblib

from preprocess import (
    clip_inputs,
    engineer_features,
    load_encoders,
    build_model_matrix,
    align_columns,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_model = None
_encoders = None
_feature_columns = None
_residual_std_log = None


def _load_artifacts():
    """Lazy-load model artifacts once per process."""
    global _model, _encoders, _feature_columns, _residual_std_log
    if _model is None:
        _model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
        _encoders = load_encoders(os.path.join(MODELS_DIR, "encoders.json"))
        with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
            _feature_columns = json.load(f)
        with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
            metrics = json.load(f)
            _residual_std_log = metrics["residual_std_log"]
    return _model, _encoders, _feature_columns, _residual_std_log


def _build_row(car: dict) -> pd.DataFrame:
    car = clip_inputs(car)
    df = pd.DataFrame([car])
    df = engineer_features(df)
    return df


def predict_price(car: dict, confidence_z: float = 1.0) -> dict:
    """Predict the resale price for a single car.

    Args:
        car: dict with keys brand, model, vehicle_age, km_driven, seller_type,
             fuel_type, transmission_type, mileage, engine, max_power, seats
        confidence_z: z-score multiplier for the range (1.0 ~= 68% interval,
             1.96 ~= 95% interval), applied in log space then converted back.

    Returns:
        dict with predicted_price, low_estimate, high_estimate (all in rupees)
    """
    model, encoders, feature_columns, residual_std_log = _load_artifacts()

    row_df = _build_row(car)
    X = build_model_matrix(row_df, encoders)
    X = align_columns(X, feature_columns)

    pred_log = model.predict(X)[0]
    price = float(np.expm1(pred_log))

    low_log = pred_log - confidence_z * residual_std_log
    high_log = pred_log + confidence_z * residual_std_log
    low = float(np.expm1(low_log))
    high = float(np.expm1(high_log))

    return {
        "predicted_price": round(price),
        "low_estimate": round(low),
        "high_estimate": round(high),
    }


def get_feature_matrix_for_shap(car: dict) -> pd.DataFrame:
    """Return the aligned feature row (used by the Streamlit app for SHAP explanations)."""
    _, encoders, feature_columns, _ = _load_artifacts()
    row_df = _build_row(car)
    X = build_model_matrix(row_df, encoders)
    X = align_columns(X, feature_columns)
    return X


if __name__ == "__main__":
    example = {
        "brand": "Maruti", "model": "Swift", "vehicle_age": 4,
        "km_driven": 35000, "seller_type": "Dealer", "fuel_type": "Petrol",
        "transmission_type": "Manual", "mileage": 20.5, "engine": 1197,
        "max_power": 85.0, "seats": 5,
    }
    print(predict_price(example))
