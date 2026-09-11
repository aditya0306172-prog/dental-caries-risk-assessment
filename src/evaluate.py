"""
Model evaluation & visualisation script.

Trains each model with Stratified 5-Fold CV and generates:
    1. roc_curves.png           – overlaid ROC curves (mean + std band)
    2. pr_curves.png            – overlaid Precision-Recall curves
    3. confusion_matrices.png   – side-by-side confusion matrices
    4. metric_comparison.png    – grouped bar chart of PR-AUC / ROC-AUC / Recall
    5. learning_curves.png      – learning curves for each model
"""

from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    learning_curve,
)
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from data_loader import load_data

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", font_scale=1.1)

REPORTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
os.makedirs(REPORTS, exist_ok=True)

COLOURS = {"LogisticRegression": "#2ecc71", "RandomForest": "#3498db", "XGBoost": "#e74c3c"}


def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(REPORTS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {name}")


def build_pipelines() -> dict[str, ImbPipeline]:
    return {
        "LogisticRegression": ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "RandomForest": ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ]),
        "XGBoost": ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("clf", XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=5,
                eval_metric="logloss", random_state=42, n_jobs=-1,
            )),
        ]),
    }


# ──────────────────────────────────────────────────────────
# 1 & 2.  ROC + PR curves (per-fold + mean)
# ──────────────────────────────────────────────────────────
def plot_roc_and_pr_curves(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipelines = build_pipelines()

    fig_roc, ax_roc = plt.subplots(figsize=(8, 7))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 7))

    for name, pipe in pipelines.items():
        colour = COLOURS[name]
        tprs, aucs_roc = [], []
        precisions_interp, aucs_pr = [], []
        mean_fpr = np.linspace(0, 1, 200)
        mean_recall_grid = np.linspace(0, 1, 200)

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipe.fit(X_train, y_train)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            # ROC
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs_roc.append(roc_auc_score(y_test, y_prob))

            # PR
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            # Reverse so recall is increasing for interpolation
            prec_rev, rec_rev = prec[::-1], rec[::-1]
            precisions_interp.append(np.interp(mean_recall_grid, rec_rev, prec_rev))
            aucs_pr.append(average_precision_score(y_test, y_prob))

        # ROC mean curve
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(aucs_roc)
        ax_roc.plot(mean_fpr, mean_tpr, color=colour, lw=2.2,
                    label=f"{name}  (AUC = {mean_auc:.3f})")
        ax_roc.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                            color=colour, alpha=0.12)

        # PR mean curve
        mean_prec = np.mean(precisions_interp, axis=0)
        std_prec = np.std(precisions_interp, axis=0)
        mean_ap = np.mean(aucs_pr)
        ax_pr.plot(mean_recall_grid, mean_prec, color=colour, lw=2.2,
                   label=f"{name}  (AP = {mean_ap:.3f})")
        ax_pr.fill_between(mean_recall_grid, mean_prec - std_prec, mean_prec + std_prec,
                           color=colour, alpha=0.12)

    # ROC styling
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Chance")
    ax_roc.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title="ROC Curves (Stratified 5-Fold CV)")
    ax_roc.legend(loc="lower right", fontsize=10)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    fig_roc.tight_layout()
    _save(fig_roc, "roc_curves.png")

    # PR styling
    baseline = y.mean()
    ax_pr.axhline(baseline, color="grey", ls="--", lw=1, alpha=0.5, label=f"Baseline ({baseline:.2f})")
    ax_pr.set(xlabel="Recall", ylabel="Precision",
              title="Precision-Recall Curves (Stratified 5-Fold CV)")
    ax_pr.legend(loc="upper right", fontsize=10)
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(0, 1.05)
    fig_pr.tight_layout()
    _save(fig_pr, "pr_curves.png")


# ──────────────────────────────────────────────────────────
# 3.  Confusion matrices
# ──────────────────────────────────────────────────────────
def plot_confusion_matrices(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipelines = build_pipelines()
    names = list(pipelines.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, name in zip(axes, names):
        y_pred = cross_val_predict(pipelines[name], X, y, cv=cv, n_jobs=-1)
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax, square=True,
            cbar=False, linewidths=1.5, linecolor="white",
            xticklabels=["Low Risk", "High Risk"],
            yticklabels=["Low Risk", "High Risk"],
            annot_kws={"size": 16, "weight": "bold"},
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(name, fontsize=13, fontweight="bold")
    fig.suptitle("Confusion Matrices (Stratified 5-Fold CV)", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "confusion_matrices.png")


# ──────────────────────────────────────────────────────────
# 4.  Metric comparison bar chart
# ──────────────────────────────────────────────────────────
def plot_metric_comparison(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipelines = build_pipelines()

    records = []
    for name, pipe in pipelines.items():
        y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        records.append({
            "Model": name,
            "PR-AUC": average_precision_score(y, y_prob),
            "ROC-AUC": roc_auc_score(y, y_prob),
            "Recall": recall_score(y, y_pred),
        })

    models = [r["Model"] for r in records]
    metrics = ["PR-AUC", "ROC-AUC", "Recall"]
    x = np.arange(len(models))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))
    metric_colours = ["#2ecc71", "#3498db", "#e74c3c"]
    for i, (metric, clr) in enumerate(zip(metrics, metric_colours)):
        vals = [r[metric] for r in records]
        bars = ax.bar(x + i * width, vals, width, label=metric, color=clr, edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison: PR-AUC / ROC-AUC / Recall", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "metric_comparison.png")


# ──────────────────────────────────────────────────────────
# 5.  Learning curves
# ──────────────────────────────────────────────────────────
def plot_learning_curves(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipelines = build_pipelines()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    for ax, (name, pipe) in zip(axes, pipelines.items()):
        train_sizes, train_scores, test_scores = learning_curve(
            pipe, X, y, cv=cv, scoring="roc_auc",
            train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1,
        )
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_mean = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)

        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.12, color="#2ecc71")
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.12, color="#e74c3c")
        ax.plot(train_sizes, train_mean, "o-", color="#2ecc71", lw=2, label="Training")
        ax.plot(train_sizes, test_mean, "o-", color="#e74c3c", lw=2, label="Validation")
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Set Size")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_ylim(0.5, 1.02)
    axes[0].set_ylabel("ROC-AUC")
    fig.suptitle("Learning Curves (Stratified 5-Fold CV)", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "learning_curves.png")


# ──────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("Model Evaluation & Visualisation")
    print("=" * 60)

    X, y = _load()
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features\n")

    print("Generating evaluation plots ...")
    plot_roc_and_pr_curves(X, y)
    plot_confusion_matrices(X, y)
    plot_metric_comparison(X, y)
    plot_learning_curves(X, y)
    print(f"\nAll evaluation plots saved to {REPORTS}")


def _load():
    df = load_data()
    return df.drop(columns=["caries_or_perio_risk"]), df["caries_or_perio_risk"]


if __name__ == "__main__":
    main()
