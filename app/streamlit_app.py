"""
Streamlit app: Used Car Price Predictor.

Run:
    streamlit run app/streamlit_app.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make src/ importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import predict_price, get_feature_matrix_for_shap  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_resource
def load_model_for_shap():
    return joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))


@st.cache_data
def load_brand_model_map():
    with open(os.path.join(DATA_DIR, "brand_model_map.json")) as f:
        return json.load(f)


st.title("🚗 Used Car Price Predictor")
st.caption("Estimate the fair resale value of a used car based on its specs. "
           "Model: tuned XGBoost trained on the CarDekho dataset (R² ≈ 0.95).")

brand_model_map = load_brand_model_map()

with st.form("car_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", sorted(brand_model_map.keys()), index=sorted(brand_model_map.keys()).index("Maruti"))
        model_options = brand_model_map.get(brand, [])
        model_name = st.selectbox("Model", model_options)
        vehicle_age = st.slider("Vehicle Age (years)", 0, 25, 5)
        km_driven = st.number_input("KM Driven", min_value=0, max_value=300000, value=40000, step=1000)
        seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])

    with col2:
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        transmission_type = st.selectbox("Transmission", ["Manual", "Automatic"])
        mileage = st.number_input("Mileage (km/l or km/kg)", min_value=5.0, max_value=35.0, value=18.5, step=0.1)
        engine = st.number_input("Engine (cc)", min_value=600, max_value=6000, value=1200, step=50)
        max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=650.0, value=80.0, step=1.0)

    seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9], index=2)

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

if submitted:
    car = {
        "brand": brand, "model": model_name, "vehicle_age": vehicle_age,
        "km_driven": km_driven, "seller_type": seller_type, "fuel_type": fuel_type,
        "transmission_type": transmission_type, "mileage": mileage, "engine": engine,
        "max_power": max_power, "seats": seats,
    }

    result = predict_price(car, confidence_z=1.0)

    st.divider()
    st.subheader("Predicted Resale Price")

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated Price", f"₹{result['predicted_price']:,}")
    c2.metric("Lower Bound", f"₹{result['low_estimate']:,}")
    c3.metric("Upper Bound", f"₹{result['high_estimate']:,}")

    st.caption("Range reflects typical model uncertainty (~68% confidence interval), "
               "not a guarantee of sale price. Actual price also depends on condition, "
               "location, and market demand not captured in this dataset.")

    # --- SHAP explanation for this specific prediction ---
    st.subheader("Why this price?")
    with st.spinner("Computing explanation..."):
        model = load_model_for_shap()
        X_row = get_feature_matrix_for_shap(car)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_row)

        fig = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(shap_values[0], max_display=8, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.caption("Each bar shows how much a feature pushed the prediction up (red) "
               "or down (blue) relative to the average predicted car, in log-price space.")

st.divider()
with st.expander("About this model"):
    try:
        with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
            metrics = json.load(f)
        st.write(f"**Best model:** {metrics['best_model']}")
        st.json(metrics["final_test_metrics"])
    except FileNotFoundError:
        st.write("Run `python src/train.py` first to generate model metrics.")
