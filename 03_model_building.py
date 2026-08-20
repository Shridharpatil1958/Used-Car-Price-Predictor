"""
Step 3: Model Building
Trains multiple models, compares performance, saves the best pipeline
(preprocessing + model bundled together, so the frontend doesn't need
to reimplement any encoding logic).
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

df = pd.read_csv("data/quikr_car_clean.csv")

X = df[["name", "company", "year", "kms_driven", "fuel_type"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

categorical_features = ["name", "company", "fuel_type"]
numeric_features = ["year", "kms_driven"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="passthrough",  # numeric features pass through unchanged
)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, max_depth=12),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42),
}

results = []
best_score = -np.inf
best_pipeline = None
best_name = None

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")

    results.append({
        "model": name,
        "R2": round(r2, 4),
        "MAE": round(mae, 0),
        "RMSE": round(rmse, 0),
        "CV_R2_mean": round(cv_scores.mean(), 4),
        "CV_R2_std": round(cv_scores.std(), 4),
    })

    if r2 > best_score:
        best_score = r2
        best_pipeline = pipe
        best_name = name

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print(results_df.to_string(index=False))
print(f"\nBest model: {best_name} (R2={best_score:.4f})")

# Feature importance (if tree-based) for interpretability
if best_name in ("RandomForest", "XGBoost"):
    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = best_pipeline.named_steps["model"].feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
    print("\nTop 15 feature importances:")
    print(fi)

# Save the full pipeline (preprocessing + model together)
joblib.dump(best_pipeline, "models/car_price_model.pkl")
print("\nSaved best pipeline to models/car_price_model.pkl")

# Also save reference data the frontend needs: valid company/name/fuel_type
# options and year range, so dropdowns only offer values the model has seen
reference = {
    "companies": sorted(df["company"].unique().tolist()),
    "names": sorted(df["name"].unique().tolist()),
    "fuel_types": sorted(df["fuel_type"].unique().tolist()),
    "year_min": int(df["year"].min()),
    "year_max": int(df["year"].max()),
    "kms_max": int(df["kms_driven"].max()),
}
joblib.dump(reference, "models/reference_data.pkl")
print("Saved reference data to models/reference_data.pkl")
