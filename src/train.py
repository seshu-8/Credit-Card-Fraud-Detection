"""
train.py
Trains multiple classifiers, compares them, saves the best model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, time

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree            import DecisionTreeClassifier
from xgboost                 import XGBClassifier

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, average_precision_score
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FIG_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR,   exist_ok=True)

MODELS = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=150, class_weight="balanced",
                                                   n_jobs=-1, random_state=42),
    "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost"             : XGBClassifier(n_estimators=150, scale_pos_weight=50,
                                          eval_metric="logloss", random_state=42, verbosity=0),
}


def train_all(X_train, y_train, X_test, y_test) -> dict:
    results = {}
    print("\n── Model Training ──────────────────────────────────")

    for name, model in MODELS.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        f1      = f1_score(y_test, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        auc_pr  = average_precision_score(y_test, y_prob) if y_prob is not None else None

        results[name] = {
            "model"  : model,
            "y_pred" : y_pred,
            "y_prob" : y_prob,
            "f1"     : f1,
            "auc_roc": auc_roc,
            "auc_pr" : auc_pr,
            "time"   : elapsed,
        }
        print(f"  {name:<24} F1={f1:.4f}  AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  ({elapsed:.1f}s)")

    print("────────────────────────────────────────────────────\n")
    return results


def pick_best(results: dict) -> str:
    """Best model by F1 score (most important for imbalanced fraud data)."""
    best = max(results, key=lambda k: results[k]["f1"])
    print(f"[train] Best model → {best}  (F1={results[best]['f1']:.4f})")
    return best


def save_model(model, name: str):
    safe_name = name.replace(" ", "_").lower()
    path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
    joblib.dump(model, path)
    print(f"[train] Model saved → {path}")
    return path


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, model_name: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"], ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "06_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {path}")


def plot_roc_curves(results: dict, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    for name, res in results.items():
        if res["y_prob"] is not None:
            fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
            ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={res['auc_roc']:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "07_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {path}")


def plot_precision_recall(results: dict, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        if res["y_prob"] is not None:
            prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
            ax.plot(rec, prec, linewidth=2, label=f"{name} (AP={res['auc_pr']:.3f})")

    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "08_precision_recall.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {path}")


def plot_model_comparison(results: dict):
    names   = list(results.keys())
    f1s     = [results[n]["f1"]     for n in names]
    auc_rocs= [results[n]["auc_roc"] for n in names]
    auc_prs = [results[n]["auc_pr"]  for n in names]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - w,  f1s,     w, label="F1 Score",  color="#3498db", edgecolor="black")
    bars2 = ax.bar(x,      auc_rocs, w, label="AUC-ROC",   color="#2ecc71", edgecolor="black")
    bars3 = ax.bar(x + w,  auc_prs,  w, label="AUC-PR",    color="#e74c3c", edgecolor="black")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "09_model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {path}")


def plot_feature_importance(model, feature_names: list, model_name: str):
    if not hasattr(model, "feature_importances_"):
        print("[plot] Feature importance not available for this model.")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top20 = importances.nlargest(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    top20.sort_values().plot(kind="barh", ax=ax, color="#3498db", edgecolor="black")
    ax.set_title(f"Feature Importance — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "10_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {path}")


def print_classification_report(y_test, y_pred, model_name: str):
    print(f"\n── Classification Report: {model_name} ──────────────")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], zero_division=0))
