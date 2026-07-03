"""
Model explainability using SHAP for the CarDekho Used Car Price Prediction project.

Generates:
  - Global feature importance (SHAP summary/bar plot)
  - Waterfall plots for 3 individual test-set predictions

Run:
    python src/explain.py
Outputs PNGs to reports/figures/
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import (
    full_training_pipeline,
    fit_target_encoders,
    build_model_matrix,
    TARGET_COL,
)
from sklearn.model_selection import train_test_split

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_STATE = 42


def main():
    print("Reloading data + trained model for explainability...")
    df = full_training_pipeline()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

    encoders = fit_target_encoders(train_df, cols=["brand", "model"], target_col=TARGET_COL)
    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)

    X_test = build_model_matrix(test_df, encoders).reindex(columns=feature_columns, fill_value=0)
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))

    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # --- Global feature importance (bar) ---
    fig = plt.figure(figsize=(9, 6))
    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.title("Global Feature Importance (mean |SHAP value|, log-price space)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "11_shap_global_importance.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # --- Beeswarm summary ---
    fig = plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title("SHAP Summary — feature value vs impact on prediction")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "12_shap_beeswarm.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # --- Individual waterfall plots for 3 sample predictions ---
    sample_idx = [0, 1, 2]
    interpretations = []
    for i, idx in enumerate(sample_idx):
        fig = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(shap_values[idx], max_display=10, show=False)
        plt.tight_layout()
        fname = f"13_shap_waterfall_sample{i+1}.png"
        fig.savefig(os.path.join(FIG_DIR, fname), dpi=110, bbox_inches="tight")
        plt.close(fig)

        row = test_df.iloc[idx]
        actual_price = row[TARGET_COL]
        pred_log = model.predict(X_test.iloc[[idx]])[0]
        pred_price = np.expm1(pred_log)
        interpretations.append({
            "sample": i + 1,
            "brand": row["brand"], "model": row["model"],
            "vehicle_age": int(row["vehicle_age"]), "km_driven": int(row["km_driven"]),
            "actual_price": int(actual_price), "predicted_price": int(pred_price),
        })

    with open(os.path.join(FIG_DIR, "shap_sample_summary.json"), "w") as f:
        json.dump(interpretations, f, indent=2)

    print("\nSample predictions explained:")
    for item in interpretations:
        print(item)

    print("\nSHAP figures saved to reports/figures/")


if __name__ == "__main__":
    main()
