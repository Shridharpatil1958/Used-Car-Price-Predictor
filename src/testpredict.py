"""
Tests for src/predict.py

NOTE: These tests are written against the public interface documented in the
README (predict_price(dict) -> dict with predicted_price/low_estimate/
high_estimate). Adjust field names/imports if your actual implementation
differs slightly.
"""
import pytest

from src.predict import predict_price

VALID_CAR = {
    "brand": "Maruti",
    "model": "Swift",
    "vehicle_age": 4,
    "km_driven": 35000,
    "seller_type": "Dealer",
    "fuel_type": "Petrol",
    "transmission_type": "Manual",
    "mileage": 20.5,
    "engine": 1197,
    "max_power": 85.0,
    "seats": 5,
}


def test_predict_price_returns_expected_keys():
    result = predict_price(VALID_CAR)
    assert set(result.keys()) >= {"predicted_price", "low_estimate", "high_estimate"}


def test_predict_price_is_positive():
    result = predict_price(VALID_CAR)
    assert result["predicted_price"] > 0


def test_predict_price_range_is_sane():
    result = predict_price(VALID_CAR)
    assert result["low_estimate"] <= result["predicted_price"] <= result["high_estimate"]


def test_predict_price_handles_unknown_brand_gracefully():
    """New/unseen brands should not crash inference (target encoding fallback)."""
    car = dict(VALID_CAR, brand="TotallyUnknownBrandXYZ", model="MadeUpModel")
    result = predict_price(car)
    assert result["predicted_price"] > 0


@pytest.mark.parametrize("km_driven", [0, 500_000, -100])
def test_predict_price_clips_extreme_km_driven(km_driven):
    """Extreme km_driven values should be clipped at inference, not raise."""
    car = dict(VALID_CAR, km_driven=km_driven)
    result = predict_price(car)
    assert result["predicted_price"] > 0


def test_predict_price_handles_zero_max_power():
    """max_power <= 0 is treated as a data error and should be clipped, not crash the app."""
    car = dict(VALID_CAR, max_power=0)
    result = predict_price(car)
    assert result["predicted_price"] > 0


def test_predict_price_missing_field_raises():
    incomplete_car = dict(VALID_CAR)
    del incomplete_car["engine"]
    with pytest.raises((KeyError, ValueError, TypeError)):
        predict_price(incomplete_car)
