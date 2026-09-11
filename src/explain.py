"""
SHAP explainability script (enhanced).

Trains XGBoost with SMOTE on the full dataset, then generates:
    1. shap_summary.png      – beeswarm summary plot
    2. shap_bar.png           – global mean |SHAP| bar chart
    3. shap_dependence.png    – dependence plots for top-4 features
    4. shap_waterfall.png     – waterfall plot for a single high-risk patient
"""

from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from data_loader import load_data

warnings.filterwarnings("ignore", category=FutureWarning)

REPORTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
os.makedirs(REPORTS, exist_ok=True)


def _save(name: str) -> None:
    path = os.path.join(REPORTS, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {name}")


def main() -> None:
    print("=" * 60)
    print("SHAP Explainability (Enhanced) - TreeExplainer (XGBoost)")
    print("=" * 60)

    # ── 1. Load & resample ────────────────────────────────
    df = load_data()
    X = df.drop(columns=["caries_or_perio_risk"])
    y = df["caries_or_perio_risk"]

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"\nAfter SMOTE: {X_res.shape[0]} samples  "
          f"(class 0 = {(y_res == 0).sum()}, class 1 = {(y_res == 1).sum()})")

    # ── 2. Train XGBoost ──────────────────────────────────
    model = XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    model.fit(X_res, y_res)
    print("Model trained.\n")

    # ── 3. SHAP values ────────────────────────────────────
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    explanation = explainer(X)
    print(f"SHAP values shape: {shap_values.shape}\n")

    print("Generating SHAP plots ...")

    # ── Plot 1: Beeswarm summary ──────────────────────────
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Beeswarm Summary", fontsize=14, fontweight="bold", pad=14)
    _save("shap_summary.png")

    # ── Plot 2: Global bar chart ──────────────────────────
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("Mean |SHAP| Feature Importance", fontsize=14, fontweight="bold", pad=14)
    _save("shap_bar.png")

    # ── Plot 3: Dependence plots (top 4 features) ────────
    mean_abs = np.abs(shap_values).mean(axis=0)
    top4_idx = np.argsort(mean_abs)[::-1][:4]
    top4_features = [X.columns[i] for i in top4_idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax_obj, feat in zip(axes.ravel(), top4_features):
        shap.dependence_plot(
            feat, shap_values, X, ax=ax_obj, show=False,
        )
        ax_obj.set_title(feat.replace("_", " ").title(), fontweight="bold")
    fig.suptitle("SHAP Dependence Plots (Top-4 Features)", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(REPORTS, "shap_dependence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] shap_dependence.png")

    # ── Plot 4: Waterfall for a high-risk patient ─────────
    high_risk_indices = np.where(y.values == 1)[0]
    sample_idx = high_risk_indices[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation[sample_idx], show=False)
    plt.title(f"SHAP Waterfall - Patient #{sample_idx} (High Risk)", fontsize=13, fontweight="bold")
    _save("shap_waterfall.png")

    print(f"\nAll SHAP plots saved to {REPORTS}")


if __name__ == "__main__":
    main()
