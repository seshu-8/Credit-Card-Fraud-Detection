"""
predict.py
Real-time fraud prediction simulation with alert system.
"""

import numpy as np
import pandas as pd
import joblib, os, time

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(model_name: str = "random_forest"):
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}\nRun main.py first.")
    return joblib.load(path)


def load_scaler():
    path = os.path.join(MODEL_DIR, "scaler.pkl")
    return joblib.load(path)


def preprocess_transaction(transaction: dict, scaler) -> pd.DataFrame:
    """Scale a single raw transaction dict into model-ready format."""
    row = pd.DataFrame([transaction])
    row[["Amount", "Time"]] = scaler.transform(row[["Amount", "Time"]])
    return row


def predict_single(model, scaler, transaction: dict, threshold: float = 0.5) -> dict:
    """Predict fraud probability for one transaction."""
    X = preprocess_transaction(transaction, scaler)
    prob  = model.predict_proba(X)[0][1]
    label = int(prob >= threshold)

    risk_level = (
        "🔴 HIGH RISK"   if prob >= 0.75 else
        "🟠 MEDIUM RISK" if prob >= 0.40 else
        "🟢 LOW RISK"
    )

    return {
        "fraud_probability" : round(prob, 4),
        "prediction"        : "FRAUD" if label else "LEGIT",
        "risk_level"        : risk_level,
        "alert_triggered"   : label == 1,
    }


def simulate_realtime(model, scaler, n_transactions: int = 20, delay: float = 0.15):
    """Simulate a live transaction stream with fraud alerts."""
    np.random.seed(99)

    print("\n" + "═" * 60)
    print("    💳  REAL-TIME FRAUD DETECTION SIMULATION")
    print("═" * 60)
    print(f"{'TXN':>4}  {'AMOUNT':>9}  {'PREDICTION':>8}  {'PROB':>6}  RISK")
    print("─" * 60)

    alerts      = []
    predictions = []

    for i in range(1, n_transactions + 1):
        # Randomly decide if this is a fraud attempt
        is_fraud = (np.random.rand() < 0.2)

        txn = {
            "Time"  : float(np.random.randint(0, 172800)),
            "Amount": round(np.random.lognormal(4.5, 1.8) if is_fraud
                            else np.random.lognormal(3.0, 1.2), 2),
        }
        for j in range(1, 29):
            shift = np.random.uniform(-3, 3) if is_fraud else 0
            txn[f"V{j}"] = np.random.normal(shift, 1.5 if is_fraud else 1.0)

        result = predict_single(model, scaler, txn)
        predictions.append(result["prediction"])

        alert_tag = "  ⚠️  ALERT!" if result["alert_triggered"] else ""
        print(f"#{i:>3}  ${txn['Amount']:>8.2f}  {result['prediction']:>8}  "
              f"{result['fraud_probability']:>5.2%}  {result['risk_level']}{alert_tag}")

        if result["alert_triggered"]:
            alerts.append({**txn, **result, "txn_id": i})

        time.sleep(delay)

    print("─" * 60)
    total_fraud = predictions.count("FRAUD")
    print(f"\n  Summary: {n_transactions} transactions | {total_fraud} flagged as FRAUD")
    print(f"  Fraud rate detected: {total_fraud/n_transactions*100:.1f}%")

    if alerts:
        alerts_df = pd.DataFrame(alerts)[["txn_id", "Amount", "fraud_probability", "prediction", "risk_level"]]
        out_path  = os.path.join(OUTPUT_DIR, "fraud_alerts.csv")
        alerts_df.to_csv(out_path, index=False)
        print(f"\n  🚨 {len(alerts)} alert(s) logged → {out_path}")
        print(alerts_df.to_string(index=False))
    else:
        print("\n  ✅ No fraud detected.")

    print("═" * 60 + "\n")
    return alerts


def batch_predict(model, scaler, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Run predictions on an entire DataFrame."""
    X = df.copy()
    X[["Amount", "Time"]] = scaler.transform(X[["Amount", "Time"]])
    probs  = model.predict_proba(X)[:, 1]
    labels = (probs >= threshold).astype(int)
    df = df.copy()
    df["fraud_probability"] = probs.round(4)
    df["prediction"]        = labels
    return df
