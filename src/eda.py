"""
Exploratory Data Analysis for the CarDekho Used Car dataset.

Generates:
  - Univariate distribution plots (price, mileage/km_driven, age, brand)
  - Bivariate plots (price vs age, price vs km_driven, price vs fuel/transmission)
  - Correlation heatmap for numeric features
  - Console summary of outliers and data quality flags

Run:
    python src/eda.py
Outputs PNGs to reports/figures/
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cardekho_dataset.csv")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CarDekho dataset and drop the stray index column."""
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=110, bbox_inches="tight")
    plt.close(fig)


def univariate_plots(df: pd.DataFrame):
    # Selling price distribution (raw + log)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["selling_price"], bins=60, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Selling Price (raw) — heavily right-skewed")
    sns.histplot(np.log1p(df["selling_price"]), bins=60, ax=axes[1], color="#55A868")
    axes[1].set_title("Selling Price (log1p) — closer to normal")
    save(fig, "01_price_distribution.png")

    # km_driven distribution (raw + log)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["km_driven"], bins=60, ax=axes[0], color="#4C72B0")
    axes[0].set_title("km_driven (raw) — long tail from data-entry outliers")
    sns.histplot(np.log1p(df["km_driven"]), bins=60, ax=axes[1], color="#55A868")
    axes[1].set_title("km_driven (log1p)")
    save(fig, "02_km_driven_distribution.png")

    # Vehicle age
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x="vehicle_age", data=df, color="#4C72B0", ax=ax)
    ax.set_title("Vehicle Age Distribution (years)")
    save(fig, "03_vehicle_age_distribution.png")

    # Brand frequency (top 15)
    fig, ax = plt.subplots(figsize=(9, 5))
    top_brands = df["brand"].value_counts().head(15)
    sns.barplot(x=top_brands.values, y=top_brands.index, ax=ax, color="#4C72B0")
    ax.set_title("Top 15 Brands by Listing Count")
    ax.set_xlabel("Count")
    save(fig, "04_top_brands.png")


def bivariate_plots(df: pd.DataFrame):
    # Price vs age
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="vehicle_age", y="selling_price", data=df, ax=ax, color="#4C72B0")
    ax.set_yscale("log")
    ax.set_title("Selling Price vs Vehicle Age (log scale)")
    save(fig, "05_price_vs_age.png")

    # Price vs km_driven (scatter, log-log)
    fig, ax = plt.subplots(figsize=(7, 5))
    sample = df.sample(min(4000, len(df)), random_state=42)
    ax.scatter(sample["km_driven"], sample["selling_price"], alpha=0.25, s=10, color="#4C72B0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("km_driven (log)")
    ax.set_ylabel("selling_price (log)")
    ax.set_title("Selling Price vs km_driven")
    save(fig, "06_price_vs_km_driven.png")

    # Price vs fuel type
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(x="fuel_type", y="selling_price", data=df, ax=ax, color="#55A868")
    ax.set_yscale("log")
    ax.set_title("Selling Price vs Fuel Type (log scale)")
    save(fig, "07_price_vs_fuel.png")

    # Price vs transmission
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(x="transmission_type", y="selling_price", data=df, ax=ax, color="#C44E52")
    ax.set_yscale("log")
    ax.set_title("Selling Price vs Transmission Type (log scale)")
    save(fig, "08_price_vs_transmission.png")

    # Price vs seller type
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(x="seller_type", y="selling_price", data=df, ax=ax, color="#8172B2")
    ax.set_yscale("log")
    ax.set_title("Selling Price vs Seller Type (log scale)")
    save(fig, "09_price_vs_seller_type.png")


def correlation_heatmap(df: pd.DataFrame):
    numeric_cols = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats", "selling_price"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap — Numeric Features")
    save(fig, "10_correlation_heatmap.png")
    return corr


def outlier_report(df: pd.DataFrame):
    print("\n--- OUTLIER / DATA QUALITY FLAGS ---")

    q1, q99 = df["selling_price"].quantile([0.01, 0.99])
    extreme_price = df[df["selling_price"] > df["selling_price"].quantile(0.995)]
    print(f"Selling price: 1st-99th pct range = {q1:,.0f} to {q99:,.0f}")
    print(f"  -> {len(extreme_price)} rows above 99.5th percentile (max = {df['selling_price'].max():,.0f})")

    high_km = df[df["km_driven"] > 300000]
    print(f"km_driven: {len(high_km)} rows above 300,000 km (max = {df['km_driven'].max():,.0f}) -- likely data-entry errors")

    zero_seats = df[df["seats"] == 0]
    print(f"seats: {len(zero_seats)} rows with seats == 0 -- invalid, should be dropped or imputed")

    zero_power = df[df["max_power"] <= 0]
    print(f"max_power: {len(zero_power)} rows with max_power <= 0")

    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"Missing values total: {df.isnull().sum().sum()}")


if __name__ == "__main__":
    import numpy as np  # noqa: E402  (used inside univariate_plots via globals)
    globals()["np"] = np

    df = load_data()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    univariate_plots(df)
    bivariate_plots(df)
    corr = correlation_heatmap(df)
    outlier_report(df)
    print("\nAll figures saved to reports/figures/")
