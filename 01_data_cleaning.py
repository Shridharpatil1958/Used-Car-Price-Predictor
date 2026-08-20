"""
Step 1: Data Cleaning
Dataset: quikr_car.csv (scraped used-car listings from Quikr, India)

Known issues in raw data:
- year: contains junk text values mixed with valid years (e.g. 'TOUR', '150k')
- Price: contains 'Ask For Price' and comma-formatted numbers (Indian lakh format)
- kms_driven: contains ' kms' suffix and commas; some NaN
- fuel_type: has NaN values
- company: first "word" of name is usually company, but a few rows are junk
- Some fully duplicated rows
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/quikr_car.csv"
CLEAN_PATH = "data/quikr_car_clean.csv"


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path)
    return df


def clean(df):
    df = df.copy()

    # ---- 1. year: keep only rows where year is a valid 4-digit number ----
    df = df[df["year"].str.isnumeric()]
    df["year"] = df["year"].astype(int)
    # sanity bound - no used car listing should be from the future or absurdly old
    df = df[(df["year"] >= 1990) & (df["year"] <= 2024)]

    # ---- 2. Price: drop 'Ask For Price', strip commas, convert to int ----
    df = df[df["Price"] != "Ask For Price"]
    df["Price"] = df["Price"].str.replace(",", "", regex=False).astype(int)

    # ---- 3. kms_driven: strip ' kms' and commas, convert to int, drop NaN ----
    df = df[df["kms_driven"].notna()]
    df["kms_driven"] = (
        df["kms_driven"].str.replace(" kms", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    # a few rows may have non-numeric leftovers - drop those defensively
    df = df[df["kms_driven"].str.isnumeric()]
    df["kms_driven"] = df["kms_driven"].astype(int)

    # ---- 4. fuel_type: drop missing (small % of rows) ----
    df = df[df["fuel_type"].notna()]

    # ---- 5. name: keep only first 3 words (brand + model), matches how
    #          these listings are typically compared) and strip whitespace ----
    df["name"] = df["name"].str.strip().apply(lambda x: " ".join(x.split()[:3]))

    # ---- 6. company: drop rows where company is clearly not a real brand
    #          (single-letter codes, generic words like 'Used', 'selling') ----
    junk_companies = {
        "I", "Used", "URJENT", "selling", "Well", "TATA MOTORS",
        "Any", "condition", "Sell", "Only",
    }
    df = df[~df["company"].isin(junk_companies)]
    df = df[df["company"].str.len() > 2]

    # ---- 7. Outlier removal on Price ----
    # Extremely cheap "cars" (< 30k INR) are usually junk/spare-parts listings,
    # not real cars. Extremely expensive outliers can distort a linear model.
    df = df[df["Price"] < 60_00_000]  # < 60 lakh
    df = df[df["Price"] > 30_000]     # > 30k

    # ---- 8. kms_driven outlier removal ----
    df = df[df["kms_driven"] < 400_000]

    # ---- 9. drop exact duplicates ----
    df = df.drop_duplicates()

    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    raw = load_raw()
    print(f"Raw shape: {raw.shape}")

    cleaned = clean(raw)
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Rows dropped: {raw.shape[0] - cleaned.shape[0]}")

    cleaned.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned data to {CLEAN_PATH}")
    print(cleaned.head())
    print(cleaned.dtypes)
