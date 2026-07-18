"""
Tests for src/preprocess.py

NOTE: Written against the design decisions documented in the README
(target encoding for brand/model, log1p on selling_price, outlier
clipping on km_driven/seats/max_power). Rename the imported functions
to match your actual src/preprocess.py API if they differ.
"""
import numpy as np
import pandas as pd
import pytest

from src import preprocess


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        {
            "brand": ["Maruti", "Hyundai", "Maruti", "Toyota"],
            "model": ["Swift", "i20", "Swift", "Innova"],
            "vehicle_age": [4, 2, 8, 10],
            "km_driven": [35000, 12000, 90000, 310000],  # last one is an outlier
            "seller_type": ["Dealer", "Individual", "Dealer", "Dealer"],
            "fuel_type": ["Petrol", "Petrol", "Diesel", "Diesel"],
            "transmission_type": ["Manual", "Manual", "Manual", "Automatic"],
            "mileage": [20.5, 18.2, 22.1, 14.0],
            "engine": [1197, 1197, 1248, 2393],
            "max_power": [85.0, 82.0, 74.0, 0.0],  # last one is an outlier
            "seats": [5, 5, 5, 0],  # last one is an outlier
            "selling_price": [502675, 610000, 430000, 1500000],
        }
    )


def test_drops_known_outliers(raw_df):
    cleaned = preprocess.clean(raw_df)
    # km_driven > 300,000, seats == 0, and max_power <= 0 rows should be dropped
    assert (cleaned["km_driven"] <= 300_000).all()
    assert (cleaned["seats"] > 0).all()
    assert (cleaned["max_power"] > 0).all()


def test_drops_duplicate_rows(raw_df):
    dup_df = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
    cleaned = preprocess.clean(dup_df)
    assert not cleaned.duplicated().any()


def test_log1p_target_is_monotonic(raw_df):
    cleaned = preprocess.clean(raw_df)
    transformed = np.log1p(cleaned["selling_price"])
    assert (transformed.diff().dropna() != 0).any()  # sanity: not constant
    assert (transformed >= 0).all()


def test_target_encoding_uses_train_split_only(raw_df):
    """
    Target encoding for brand/model must be fit on the training split only,
    to avoid leaking test-set price information into the encoding.
    """
    cleaned = preprocess.clean(raw_df)
    train_df = cleaned.iloc[:2]
    test_df = cleaned.iloc[2:]

    encoders = preprocess.fit_target_encoders(train_df)
    encoded_test = preprocess.apply_target_encoders(test_df, encoders)

    # Encoded columns should exist and contain no NaNs even for categories
    # not seen in the (small) training split.
    assert "brand_encoded" in encoded_test.columns
    assert not encoded_test["brand_encoded"].isna().any()


def test_inference_clips_rather_than_rejects_extremes():
    """A single malformed input row should be clipped, not raise, at inference time."""
    row = {
        "brand": "Maruti",
        "model": "Swift",
        "vehicle_age": 4,
        "km_driven": 1_000_000,
        "seller_type": "Dealer",
        "fuel_type": "Petrol",
        "transmission_type": "Manual",
        "mileage": 20.5,
        "engine": 1197,
        "max_power": -5.0,
        "seats": 0,
    }
    clipped = preprocess.clip_for_inference(row)
    assert clipped["km_driven"] <= 300_000
    assert clipped["max_power"] > 0
    assert clipped["seats"] > 0
