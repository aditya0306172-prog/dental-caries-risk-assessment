"""
Training script – Stratified 5-Fold CV with SMOTE pipelines.

Compares Logistic Regression, Random Forest, and XGBoost on:
    • PR-AUC   (Average Precision)
    • ROC-AUC
    • Recall
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from data_loader import load_data

warnings.filterwarnings("ignore", category=FutureWarning)


# ──────────────────────────────────────────────
# 1.  Load NHANES data
# ──────────────────────────────────────────────
def load_xy() -> tuple[pd.DataFrame, pd.Series]:
    df = load_data()
    X = df.drop(columns=["caries_or_perio_risk"])
    y = df["caries_or_perio_risk"]
    return X, y


# ──────────────────────────────────────────────
# 2.  Build imblearn pipelines (SMOTE → model)
# ──────────────────────────────────────────────
def build_pipelines() -> dict[str, ImbPipeline]:
    return {
        "LogisticRegression": ImbPipeline(
            [
                ("smote", SMOTE(random_state=42)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "RandomForest": ImbPipeline(
            [
                ("smote", SMOTE(random_state=42)),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
            ]
        ),
        "XGBoost": ImbPipeline(
            [
                ("smote", SMOTE(random_state=42)),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=200,
                        learning_rate=0.1,
                        max_depth=5,
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


# ──────────────────────────────────────────────
# 3.  Evaluate with Stratified 5-Fold CV
# ──────────────────────────────────────────────
SCORING = {
    "PR-AUC": "average_precision",
    "ROC-AUC": "roc_auc",
    "Recall": "recall",
}


def evaluate(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipelines = build_pipelines()

    rows: list[dict] = []
    for name, pipe in pipelines.items():
        print(f"  - Evaluating {name} ...")
        scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"Model": name}
        for metric in SCORING:
            vals = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = round(vals.mean(), 4)
            row[f"{metric}_std"] = round(vals.std(), 4)
        rows.append(row)

    results = pd.DataFrame(rows)
    return results


# ──────────────────────────────────────────────
# 4.  Main
# ──────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("Dental Caries & Periodontal Disease Risk – Model Training")
    print("=" * 60)

    X, y = load_xy()
    print(f"\nDataset : {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class 0 (low risk) : {(y == 0).sum()}")
    print(f"Class 1 (high risk): {(y == 1).sum()}\n")

    print("Running Stratified 5-Fold Cross-Validation with SMOTE ...\n")
    results = evaluate(X, y)

    print("\n" + "=" * 60)
    print("Results (mean ± std across 5 folds)")
    print("=" * 60)
    for _, r in results.iterrows():
        print(f"\n  {r['Model']}")
        for metric in SCORING:
            print(f"    {metric:>8s}: {r[f'{metric}_mean']:.4f} ± {r[f'{metric}_std']:.4f}")
    print()


if __name__ == "__main__":
    main()
