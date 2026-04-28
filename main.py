"""
main.py — Credit Card Fraud Detection System
=============================================
Run:  python main.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generate_data import generate_transactions
from preprocess    import full_pipeline
from eda           import run_eda
from train         import (train_all, pick_best, save_model,
                            plot_confusion_matrix, plot_roc_curves,
                            plot_precision_recall, plot_model_comparison,
                            plot_feature_importance, print_classification_report)
from predict       import load_model, load_scaler, simulate_realtime

DATA_PATH = os.path.join("data", "creditcard_synthetic.csv")


def banner(text: str):
    print("\n" + "═" * 60)
    print(f"  {text}")
    print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1  Generate / Load Data
# ─────────────────────────────────────────────────────────────────────────────
banner("PHASE 1 — Data Generation")

if not os.path.exists(DATA_PATH):
    print("[main] Generating synthetic dataset …")
    df_raw = generate_transactions(n_samples=10000)
    os.makedirs("data", exist_ok=True)
    df_raw.to_csv(DATA_PATH, index=False)
    print(f"[main] Saved → {DATA_PATH}")
else:
    import pandas as pd
    df_raw = pd.read_csv(DATA_PATH)
    print(f"[main] Loaded existing dataset: {df_raw.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2  EDA
# ─────────────────────────────────────────────────────────────────────────────
banner("PHASE 2 — Exploratory Data Analysis")
run_eda(df_raw)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3  Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
banner("PHASE 3 — Preprocessing & SMOTE")
X_train, X_test, y_train, y_test = full_pipeline(DATA_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4  Training
# ─────────────────────────────────────────────────────────────────────────────
banner("PHASE 4 — Model Training")
results    = train_all(X_train, y_train, X_test, y_test)
best_name  = pick_best(results)
best_res   = results[best_name]
best_model = best_res["model"]

save_model(best_model, best_name)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5  Evaluation
# ─────────────────────────────────────────────────────────────────────────────
banner("PHASE 5 — Evaluation")
print_classification_report(y_test, best_res["y_pred"], best_name)
plot_confusion_matrix(y_test, best_res["y_pred"], best_name)
plot_roc_curves(results, y_test)
plot_precision_recall(results, y_test)
plot_model_comparison(results)
plot_feature_importance(best_model, list(X_test.columns), best_name)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6  Real-Time Simulation
# ─────────────────────────────────────────────────────────────────────────────
banner("PHASE 6 — Real-Time Fraud Simulation")
model  = load_model(best_name.replace(" ", "_").lower())
scaler = load_scaler()
simulate_realtime(model, scaler, n_transactions=25, delay=0.05)

banner("✅  Pipeline Complete — check outputs/figures/ for all plots")
