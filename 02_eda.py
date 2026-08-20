"""
Step 2: Exploratory Data Analysis
Generates plots into ./eda_plots/ and prints key stats to console.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
os.makedirs("eda_plots", exist_ok=True)

df = pd.read_csv("data/quikr_car_clean.csv")

print("=" * 50)
print("Shape:", df.shape)
print(df.describe(include="all"))

# 1. Price distribution (raw vs log)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["Price"], bins=40, ax=axes[0], color="steelblue")
axes[0].set_title("Price distribution (raw) - right-skewed")
sns.histplot(df["Price"].apply(lambda x: x).apply(lambda x: __import__("numpy").log1p(x)),
             bins=40, ax=axes[1], color="seagreen")
axes[1].set_title("Price distribution (log1p) - closer to normal")
plt.tight_layout()
plt.savefig("eda_plots/01_price_distribution.png", dpi=100)
plt.close()

# 2. Price vs Year
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="year", y="Price")
plt.xticks(rotation=90)
plt.title("Price by Year of Manufacture")
plt.tight_layout()
plt.savefig("eda_plots/02_price_vs_year.png", dpi=100)
plt.close()

# 3. Price vs kms_driven
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="kms_driven", y="Price", hue="fuel_type", alpha=0.6)
plt.title("Price vs Kilometers Driven")
plt.tight_layout()
plt.savefig("eda_plots/03_price_vs_kms.png", dpi=100)
plt.close()

# 4. Price by company (top 10 by count)
top_companies = df["company"].value_counts().head(10).index
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[df["company"].isin(top_companies)], x="company", y="Price")
plt.xticks(rotation=45)
plt.title("Price by Company (Top 10 by listing count)")
plt.tight_layout()
plt.savefig("eda_plots/04_price_by_company.png", dpi=100)
plt.close()

# 5. Price by fuel type
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x="fuel_type", y="Price")
plt.title("Price by Fuel Type")
plt.tight_layout()
plt.savefig("eda_plots/05_price_by_fuel.png", dpi=100)
plt.close()

# 6. Correlation heatmap (numeric features)
plt.figure(figsize=(5, 4))
numeric_df = df[["year", "Price", "kms_driven"]]
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_plots/06_correlation_heatmap.png", dpi=100)
plt.close()

print("\nSaved 6 plots to ./eda_plots/")
print("\nKey correlations with Price:")
print(numeric_df.corr()["Price"].sort_values(ascending=False))
