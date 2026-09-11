"""
Exploratory Data Analysis (EDA) for the dental caries risk dataset.

Generates the following plots in  reports/ :
    1. feature_distributions.png   – histograms for every feature
    2. correlation_heatmap.png     – annotated Pearson correlation matrix
    3. class_distribution.png      – target class balance bar chart
    4. boxplots_by_risk.png        – box plots of each feature split by risk
    5. violin_plots.png            – violin plots for non-binary features
    6. pairplot.png                – pair-wise scatter matrix coloured by risk
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from data_loader import load_data

# ── Global style ──────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = {0: "#3498db", 1: "#e74c3c"}
REPORTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
os.makedirs(REPORTS, exist_ok=True)

TARGET = "caries_or_perio_risk"


def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(REPORTS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {name}")


def _feature_lists(df):
    """Dynamically classify features into continuous / ordinal / binary."""
    features = [c for c in df.columns if c != TARGET]
    continuous = [c for c in features if df[c].nunique() > 10]
    non_binary = [c for c in features if df[c].nunique() > 2]
    return features, continuous, non_binary


# ──────────────────────────────────────────────────────────
# 1. Feature distributions
# ──────────────────────────────────────────────────────────
def plot_feature_distributions(df):
    features = [c for c in df.columns if c != TARGET]
    ncols = min(4, len(features))
    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, feat in enumerate(features):
        ax = axes[i]
        for label, colour in PALETTE.items():
            subset = df[df[TARGET] == label][feat]
            ax.hist(subset, bins=30, alpha=0.55, label=f"Risk {label}",
                    color=colour, edgecolor="white")
        ax.set_title(feat.replace("_", " ").title(), fontweight="bold")
        ax.legend(fontsize=9)
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions by Risk Class",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "feature_distributions.png")


# ──────────────────────────────────────────────────────────
# 2. Correlation heatmap
# ──────────────────────────────────────────────────────────
def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.8, ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14,
                 fontweight="bold", pad=14)
    fig.tight_layout()
    _save(fig, "correlation_heatmap.png")


# ──────────────────────────────────────────────────────────
# 3. Class distribution
# ──────────────────────────────────────────────────────────
def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df[TARGET].value_counts().sort_index()
    bars = ax.bar(
        ["Low Risk (0)", "High Risk (1)"],
        counts.values,
        color=[PALETTE[0], PALETTE[1]],
        edgecolor="white",
        width=0.5,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f"{val}  ({val / len(df) * 100:.1f}%)",
                ha="center", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_title("Target Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    _save(fig, "class_distribution.png")


# ──────────────────────────────────────────────────────────
# 4. Box plots by risk
# ──────────────────────────────────────────────────────────
def plot_boxplots(df):
    features = [c for c in df.columns if c != TARGET]
    ncols = min(4, len(features))
    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, feat in enumerate(features):
        sns.boxplot(
            data=df, x=TARGET, y=feat, ax=axes[i],
            hue=TARGET, palette=PALETTE, legend=False,
            width=0.45, fliersize=3,
        )
        axes[i].set_title(feat.replace("_", " ").title(), fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["Low Risk", "High Risk"])
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Box Plots by Risk Class",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "boxplots_by_risk.png")


# ──────────────────────────────────────────────────────────
# 5. Violin plots (non-binary features only)
# ──────────────────────────────────────────────────────────
def plot_violins(df):
    _, _, non_binary = _feature_lists(df)
    if not non_binary:
        print("  [SKIP] violin_plots.png (no non-binary features)")
        return
    ncols = min(4, len(non_binary))
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    axes = np.atleast_1d(axes).ravel()
    for i, feat in enumerate(non_binary):
        sns.violinplot(
            data=df, x=TARGET, y=feat, ax=axes[i],
            hue=TARGET, palette=PALETTE, legend=False,
            inner="quartile", cut=0,
        )
        axes[i].set_title(feat.replace("_", " ").title(), fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["Low Risk", "High Risk"])
    fig.suptitle("Violin Plots for Non-Binary Features",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "violin_plots.png")


# ──────────────────────────────────────────────────────────
# 6. Pair plot
# ──────────────────────────────────────────────────────────
def plot_pairplot(df):
    features = [c for c in df.columns if c != TARGET]
    subset = df[features + [TARGET]]
    g = sns.pairplot(
        subset, hue=TARGET, palette=PALETTE,
        diag_kind="kde",
        plot_kws={"alpha": 0.4, "s": 18, "edgecolor": "none"},
        height=2.4,
    )
    g.figure.suptitle("Pair Plot (All Features)",
                      fontsize=15, fontweight="bold", y=1.02)
    _save(g.figure, "pairplot.png")


# ──────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("Exploratory Data Analysis")
    print("=" * 60)

    df = load_data()
    print(f"\nDataset: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"\nDescriptive statistics:\n{df.describe().round(2).to_string()}\n")

    print("Generating plots ...")
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_class_distribution(df)
    plot_boxplots(df)
    plot_violins(df)
    plot_pairplot(df)
    print(f"\nAll EDA plots saved to {REPORTS}")


if __name__ == "__main__":
    main()
