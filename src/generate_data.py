"""
generate_data.py
Generates synthetic credit card transaction data with realistic fraud patterns.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

np.random.seed(42)

def generate_transactions(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic credit card transaction data.
    Fraud rate ≈ 1.7% (mirrors real-world imbalance).
    """
    n_fraud   = int(n_samples * 0.017)
    n_legit   = n_samples - n_fraud

    # ── Legitimate transactions ──────────────────────────────────────────────
    legit = {
        "Time"   : np.random.exponential(scale=3600, size=n_legit).cumsum(),
        "Amount" : np.random.lognormal(mean=3.0, sigma=1.2, size=n_legit),
    }
    for i in range(1, 29):
        legit[f"V{i}"] = np.random.normal(0, 1, size=n_legit)
    legit["Class"] = 0

    # ── Fraudulent transactions ──────────────────────────────────────────────
    fraud = {
        "Time"   : np.random.exponential(scale=1800, size=n_fraud).cumsum(),
        "Amount" : np.random.lognormal(mean=4.5, sigma=1.8, size=n_fraud),
    }
    # Fraud has noticeably different feature distributions
    for i in range(1, 29):
        mean_shift = np.random.uniform(-3, 3)
        fraud[f"V{i}"] = np.random.normal(mean_shift, 1.5, size=n_fraud)
    fraud["Class"] = 1

    df = pd.concat([
        pd.DataFrame(legit),
        pd.DataFrame(fraud)
    ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    df["Time"]   = df["Time"].astype(float)
    df["Amount"] = df["Amount"].round(2)
    df["Class"]  = df["Class"].astype(int)

    return df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "creditcard_synthetic.csv")

    df = generate_transactions(10000)
    df.to_csv(out_path, index=False)

    print(f"✅  Dataset saved → {out_path}")
    print(f"    Total rows  : {len(df):,}")
    print(f"    Fraud rows  : {df['Class'].sum():,}  ({df['Class'].mean()*100:.2f}%)")
    print(f"    Legit rows  : {(df['Class']==0).sum():,}")
    print(f"\n    Columns: {list(df.columns)}")
