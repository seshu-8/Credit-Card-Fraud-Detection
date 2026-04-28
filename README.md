# 💳 Credit Card Fraud Detection System

> **Industry-grade ML pipeline** for detecting fraudulent transactions — built for Data Science & Banking Analytics portfolios.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Problem Statement

Credit card fraud costs financial institutions **billions of dollars annually**. Banks need automated systems that can:
- Detect fraud in **real-time** with high recall
- Handle **severe class imbalance** (fraud ≈ 0.17% of all transactions)
- Minimize false positives (blocking legitimate users)
- Generate **instant alerts** on suspicious activity

---

## 🧠 Solution Overview

A complete end-to-end ML pipeline:

```
Raw Transactions
      │
      ▼
┌─────────────┐
│ Preprocessing│  ← cleaning, scaling, SMOTE oversampling
└─────────────┘
      │
      ▼
┌─────────────┐
│Feature Eng. │  ← 28 PCA-derived features + Amount + Time
└─────────────┘
      │
      ▼
┌─────────────┐
│ ML Models   │  ← LR, DT, RF, GBM, XGBoost (5 models compared)
└─────────────┘
      │
      ▼
┌─────────────┐
│  Prediction │  ← fraud probability + risk tier
└─────────────┘
      │
      ▼
┌─────────────┐
│  Alert Log  │  ← CSV log of all flagged transactions
└─────────────┘
```

---

## 🛠️ Tech Stack

| Layer            | Tools                                      |
|------------------|--------------------------------------------|
| Language         | Python 3.10+                               |
| Data             | Pandas, NumPy                              |
| ML Models        | Scikit-learn, XGBoost                      |
| Imbalance Fix    | imbalanced-learn (SMOTE)                   |
| Visualization    | Matplotlib, Seaborn                        |
| Serialization    | Joblib                                     |
| Notebook Support | Jupyter Notebook                           |

---

## 📁 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── data/
│   └── creditcard_synthetic.csv     # Auto-generated synthetic dataset
│
├── src/
│   ├── generate_data.py             # Synthetic data generator
│   ├── preprocess.py                # Cleaning, scaling, SMOTE
│   ├── eda.py                       # EDA + visualizations
│   ├── train.py                     # Model training & evaluation
│   └── predict.py                   # Prediction + live simulation
│
├── models/
│   ├── scaler.pkl                   # Fitted StandardScaler
│   └── *.pkl                        # Best trained model
│
├── outputs/
│   ├── figures/                     # All generated charts
│   └── fraud_alerts.csv             # Simulation alert log
│
├── notebooks/
│   └── analysis.ipynb               # Interactive Jupyter notebook
│
├── main.py                          # 🚀 Single-command pipeline entry
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline
```bash
python main.py
```

That's it. One command runs everything: data generation → EDA → training → evaluation → simulation.

---

## 📊 Results

| Model               | F1 Score | AUC-ROC | AUC-PR  |
|---------------------|----------|---------|---------|
| Logistic Regression | ~0.82    | ~0.95   | ~0.72   |
| Decision Tree       | ~0.84    | ~0.93   | ~0.75   |
| Random Forest       | **~0.91**| **~0.98**| **~0.87**|
| Gradient Boosting   | ~0.89    | ~0.97   | ~0.85   |
| XGBoost             | ~0.90    | ~0.98   | ~0.86   |

> Results vary slightly due to synthetic data. Random Forest selected as best by F1 score.

### Key Charts Generated

- `01_class_distribution.png` — Class imbalance visualization
- `02_amount_distribution.png` — Fraud vs legit transaction amounts
- `03_time_distribution.png` — Temporal transaction patterns
- `04_correlation_heatmap.png` — Feature correlation matrix
- `05_top_features.png` — Top fraud-correlated features
- `06_confusion_matrix.png` — Prediction accuracy breakdown
- `07_roc_curves.png` — ROC comparison across all models
- `08_precision_recall.png` — Precision-Recall tradeoff
- `09_model_comparison.png` — Side-by-side metric comparison
- `10_feature_importance.png` — Top features by importance

---

## 🔴 Real-Time Simulation

The simulation mimics a live transaction feed:

```
═══════════════════════════════════════════════════════════
    💳  REAL-TIME FRAUD DETECTION SIMULATION
═══════════════════════════════════════════════════════════
TXN     AMOUNT  PREDICTION    PROB  RISK
─────────────────────────────────────────────────────────
# 1    $124.30       LEGIT   4.20%  🟢 LOW RISK
# 2   $1832.44       FRAUD  91.30%  🔴 HIGH RISK  ⚠️  ALERT!
# 3     $43.17       LEGIT   2.10%  🟢 LOW RISK
...
```

Alerts are saved to `outputs/fraud_alerts.csv`.

---

## 🎯 Why This Project Matters for Placements

- **Data Science roles**: Complete EDA, feature engineering, model comparison
- **ML Engineering roles**: Pipeline design, model serialization, deployment-ready structure
- **Banking Analytics roles**: Domain-relevant problem, real-world fraud patterns
- **Shows**: You can handle class imbalance, evaluation metrics beyond accuracy, and productionize ML

---

## 📚 Key Concepts Demonstrated

- **SMOTE** (Synthetic Minority Oversampling Technique) for imbalance handling
- **Threshold tuning** for precision/recall tradeoff in fraud detection
- **F1 Score & AUC-PR** as primary metrics (not accuracy — misleading on imbalanced data)
- **Model serialization** with Joblib for deployment readiness
- **Alert system** design (risk tiering: High / Medium / Low)

---

## 🔗 Dataset

This project uses **synthetically generated data** that mirrors the structure of the famous [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (28 PCA components + Time + Amount + Class label).

---

## 👤 Author

**Seshu** —| VLSI Learner  
📍 Andhra Pradesh, India  
🔗 [GitHub](https://github.com/seshu-8) | [LinkedIn](www.linkedin.com/in/seshu-babu-konijeti)

---

## 📄 License

MIT License — free to use, modify, and distribute.
