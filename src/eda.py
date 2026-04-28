"""
eda.py
Exploratory Data Analysis — generates all plots saved to outputs/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")
PALETTE = {0: "#2ecc71", 1: "#e74c3c"}


def plot_class_distribution(df: pd.DataFrame):
    counts = df["Class"].value_counts()
    labels = ["Legit (0)", "Fraud (1)"]
    colors = ["#2ecc71", "#e74c3c"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution (Imbalance View)", fontsize=15, fontweight="bold")

    # Bar chart
    axes[0].bar(labels, counts.values, color=colors, edgecolor="black", width=0.5)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, f"{v:,}", ha="center", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Transaction Count")
    axes[0].set_title("Count")

    # Pie chart
    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct="%1.2f%%", startangle=90, wedgeprops=dict(edgecolor="white"))
    axes[1].set_title("Proportion")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "01_class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {path}")


def plot_amount_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transaction Amount by Class", fontsize=15, fontweight="bold")

    for cls, label, color in [(0, "Legit", "#2ecc71"), (1, "Fraud", "#e74c3c")]:
        subset = df[df["Class"] == cls]["Amount"]
        axes[0].hist(subset, bins=60, alpha=0.6, label=label, color=color, edgecolor="none")

    axes[0].set_xlabel("Amount ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Histogram")
    axes[0].legend()

    sns.boxplot(data=df, x="Class", y="Amount", hue="Class",
                palette={0: "#2ecc71", 1: "#e74c3c"}, legend=False, ax=axes[1])
    axes[1].set_xticklabels(["Legit (0)", "Fraud (1)"])
    axes[1].set_title("Boxplot")
    axes[1].set_xlabel("")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "02_amount_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {path}")


def plot_time_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 4))
    for cls, label, color in [(0, "Legit", "#2ecc71"), (1, "Fraud", "#e74c3c")]:
        subset = df[df["Class"] == cls]["Time"]
        ax.hist(subset, bins=80, alpha=0.5, label=label, color=color)
    ax.set_xlabel("Time (seconds from first transaction)")
    ax.set_ylabel("Frequency")
    ax.set_title("Transaction Time Distribution by Class", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "03_time_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {path}")


def plot_feature_correlation(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(20, 16))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                annot=False, linewidths=0.3, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "04_correlation_heatmap.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {path}")


def plot_top_feature_distributions(df: pd.DataFrame, top_n: int = 6):
    """Plot the V-features most correlated with fraud."""
    v_cols = [c for c in df.columns if c.startswith("V")]
    fraud_corr = df[v_cols + ["Class"]].corr()["Class"].drop("Class").abs().sort_values(ascending=False)
    top_features = fraud_corr.head(top_n).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"Top {top_n} Features Most Correlated with Fraud", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        for cls, label, color in [(0, "Legit", "#2ecc71"), (1, "Fraud", "#e74c3c")]:
            axes[i].hist(df[df["Class"] == cls][feat], bins=50,
                         alpha=0.6, label=label, color=color, density=True)
        axes[i].set_title(f"{feat}  (corr={fraud_corr[feat]:.3f})")
        axes[i].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "05_top_features.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {path}")


def run_eda(df: pd.DataFrame):
    print("\n── EDA ─────────────────────────────────────────────")
    print(df.describe().T.to_string())
    plot_class_distribution(df)
    plot_amount_distribution(df)
    plot_time_distribution(df)
    plot_feature_correlation(df)
    plot_top_feature_distributions(df)
    print("── EDA complete ────────────────────────────────────\n")
