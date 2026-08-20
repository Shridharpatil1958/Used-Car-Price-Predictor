"""
Streamlit frontend for the Used Car Price Predictor.
Loads the saved model pipeline + reference data, takes user input,
and returns a predicted price.

Run with: streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "car_price_model.pkl")
REF_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "reference_data.pkl")

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef1f8 100%);
    }

    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #1f2937;
    }

    .hero {
        text-align: center;
        padding: 1.5rem 1rem 1rem 1rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #ff512f, #dd2476);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: #6b7280 !important;
        font-size: 1.05rem;
        margin-top: 0;
    }

    .block-card {
        background: white;
        padding: 1.6rem 1.6rem 1.4rem 1.6rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid #eef0f4;
        margin-bottom: 1.2rem;
    }
    .block-card h3 {
        color: #1f2937 !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #6b7280 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #1f2937 !important;
    }

    /* --- Selectbox styling: force WHITE background + dark text + colored border --- */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1.5px solid #e5e7eb !important;
        border-radius: 10px !important;
        color: #1f2937 !important;
    }
    div[data-baseweb="select"] span {
        color: #1f2937 !important;
    }
    div[data-baseweb="select"] svg {
        fill: #6b7280 !important;
    }
    /* dropdown menu (the popup list) */
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background-color: #ffffff !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li {
        color: #1f2937 !important;
        background-color: #ffffff !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
        background-color: #fdeef2 !important;
    }

    /* --- Number input styling --- */
    div[data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border: 1.5px solid #e5e7eb !important;
        border-radius: 10px !important;
    }
    div[data-testid="stNumberInput"] button {
        background-color: #f3f4f6 !important;
        color: #1f2937 !important;
        border: 1.5px solid #e5e7eb !important;
    }

    /* --- Slider styling --- */
    div[data-testid="stSlider"] > div > div > div > div {
        background-color: #dd2476 !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, #ff512f, #dd2476);
        color: white !important;
        font-weight: 700;
        font-size: 1.05rem;
        border-radius: 12px;
        border: none;
        padding: 0.7rem 0;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 14px rgba(221, 36, 118, 0.35);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(221, 36, 118, 0.45);
        color: white !important;
    }
    .stButton > button p {
        color: white !important;
    }

    .price-card {
        text-align: center;
        background: linear-gradient(135deg, #10b981, #059669);
        border-radius: 18px;
        padding: 1.8rem 1rem;
        margin-top: 1rem;
        box-shadow: 0 8px 24px rgba(5, 150, 105, 0.3);
    }
    .price-card * {
        color: white !important;
    }
    .price-card .label {
        font-size: 1rem;
        opacity: 0.9;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .price-card .amount {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }

    section[data-testid="stSidebar"] { display: none; }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_reference():
    model = joblib.load(MODEL_PATH)
    reference = joblib.load(REF_PATH)
    return model, reference


model, ref = load_model_and_reference()

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🚗 Used Car Price Predictor</h1>
        <p>Get an instant resale price estimate, trained on real Quikr listing data</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Quick stats strip
# ---------------------------------------------------------------------------
s1, s2, s3 = st.columns(3)
s1.metric("Brands covered", len(ref["companies"]))
s2.metric("Listings trained on", "~700")
s3.metric("Year range", f"{ref['year_min']}–{ref['year_max']}")

st.write("")

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
names_df = pd.DataFrame({"name": ref["names"]})
names_df["company_guess"] = names_df["name"].apply(lambda x: x.split()[0])

st.markdown('<div class="block-card">', unsafe_allow_html=True)
st.subheader("Tell us about the car")

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("🏷️ Brand", ref["companies"])

    filtered_names = names_df[
        names_df["company_guess"].str.lower() == company.lower()
    ]["name"].tolist()
    if not filtered_names:
        filtered_names = ref["names"]

    model_name = st.selectbox("🚘 Model", sorted(filtered_names))
    fuel_type = st.selectbox("⛽ Fuel Type", ref["fuel_types"])

with col2:
    year = st.slider(
        "📅 Year of Manufacture",
        min_value=ref["year_min"],
        max_value=2026,
        value=2018,
    )
    kms_driven = st.number_input(
        "🛣️ Kilometers Driven",
        min_value=0,
        max_value=500000,
        value=40000,
        step=1000,
    )

st.write("")
predict_clicked = st.button("✨ Predict Price", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Prediction output
# ---------------------------------------------------------------------------
if predict_clicked:
    input_df = pd.DataFrame([{
        "name": model_name,
        "company": company,
        "year": year,
        "kms_driven": kms_driven,
        "fuel_type": fuel_type,
    }])

    prediction = model.predict(input_df)[0]
    prediction = max(prediction, 0)

    st.markdown(
        f"""
        <div class="price-card">
            <div class="label">Estimated Price</div>
            <div class="amount">₹{prediction:,.0f}</div>
            <div style="opacity:0.85;">{model_name} · {year} · {kms_driven:,} km · {fuel_type}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if year > ref["year_max"]:
        st.info(
            f"ℹ️ The model was trained on listings up to {ref['year_max']}, "
            "so predictions for newer years are extrapolated and less reliable."
        )
    if kms_driven > ref["kms_max"]:
        st.info(
            f"ℹ️ {kms_driven:,} km is above the max seen in training data "
            f"({ref['kms_max']:,} km) — treat this estimate with caution."
        )

st.write("")

# ---------------------------------------------------------------------------
# About section
# ---------------------------------------------------------------------------
with st.expander("ℹ️ About this model"):
    st.write(
        """
        - Trained on cleaned Quikr used-car listing data (~700 listings after cleaning).
        - Model: Linear Regression on one-hot encoded categorical features
          (name, company, fuel type) + numeric features (year, kms driven).
        - This is a portfolio/learning project — predictions are estimates only
          and shouldn't be used for actual pricing decisions. The dataset is
          small and regionally limited, which limits real-world accuracy.
        """
    )

st.markdown(
    "<p style='text-align:center; color:#9ca3af; font-size:0.85rem; margin-top:2rem;'>"
    "Built with Streamlit · scikit-learn</p>",
    unsafe_allow_html=True,
)