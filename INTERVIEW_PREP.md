# 🎯 Interview Preparation — Credit Card Fraud Detection

---

## 🔵 Basic Level (Fresher / Intern rounds)

**Q: What is credit card fraud detection?**
A: It's a binary classification problem where a model predicts whether a given transaction is fraudulent (Class=1) or legitimate (Class=0), based on features like amount, time, and anonymized behavioral patterns.

**Q: What dataset did you use?**
A: I generated synthetic data mirroring the real Kaggle Credit Card Fraud Dataset structure — 28 PCA-derived features (V1–V28), Amount, Time, and Class label. Fraud rate was kept at ~1.7% to match real-world imbalance.

**Q: Why is accuracy a bad metric here?**
A: Because of class imbalance — if 99.8% of transactions are legit, a model predicting everything as legit gets 99.8% accuracy but catches zero fraud. F1 Score and AUC-PR are far more meaningful.

**Q: What is SMOTE?**
A: Synthetic Minority Oversampling Technique. Instead of just duplicating fraud samples, SMOTE creates new synthetic fraud samples by interpolating between existing ones. This prevents the model from ignoring the minority class.

---

## 🟡 Intermediate Level (Data Science / ML roles)

**Q: Why use F1 score over precision or recall alone?**
A: Both matter. Precision = "of predicted frauds, how many are real?" Recall = "of real frauds, how many did we catch?" F1 is the harmonic mean — it penalizes if either is too low. In fraud, missing fraud (low recall) is costly, but too many false alarms (low precision) destroys user experience.

**Q: How does XGBoost handle imbalanced data?**
A: Using `scale_pos_weight` parameter — set to the ratio of negatives to positives. This tells XGBoost to penalize missing positive (fraud) predictions more heavily during training.

**Q: What is AUC-ROC vs AUC-PR?**
A: AUC-ROC measures separability across all thresholds. AUC-PR (Average Precision) is more informative for imbalanced datasets because ROC can be misleadingly optimistic — the PR curve focuses specifically on positive class performance.

**Q: Why did you scale Amount and Time but not V1–V28?**
A: V1–V28 are already PCA-transformed and centered. Amount and Time are raw and have high variance, so StandardScaler is applied to normalize them before model training.

**Q: What is threshold tuning?**
A: Default classification threshold is 0.5. By lowering it (e.g., to 0.3), we catch more fraud (higher recall) at the cost of more false positives. Threshold is tuned based on business requirements — banks often prefer higher recall.

---

## 🔴 Advanced Level (Senior DS / Banking Analytics roles)

**Q: How would you deploy this model in production?**
A: Package the trained model + scaler with Joblib/ONNX, expose it via a FastAPI REST endpoint, containerize with Docker, and deploy on AWS/GCP. Transactions hit the API in real-time, and the response returns fraud probability + risk tier. Monitoring (data drift, model decay) handled via MLflow or Evidently AI.

**Q: How do you handle concept drift in fraud detection?**
A: Fraudsters adapt — patterns change over time. Strategies: sliding window retraining (retrain on recent N days), champion-challenger model testing, and monitoring KS statistic / PSI (Population Stability Index) to detect drift early.

**Q: What's the difference between online and batch fraud detection?**
A: Online = real-time, transaction-by-transaction prediction before approval (< 200ms latency). Batch = post-processing a day's transactions overnight to flag suspicious patterns. Most banks use both: online for blocking, batch for investigation.

**Q: Why is recall more important than precision in fraud detection?**
A: Missing a fraud ($10,000 loss) is far more expensive than a false alarm (customer inconvenience). Business cost matrix: False Negative cost >> False Positive cost. So models are typically tuned for recall ≥ 0.85.

**Q: How does SMOTE compare to class_weight balancing?**
A: SMOTE creates new synthetic samples, changing the data distribution. `class_weight='balanced'` adjusts the loss function to penalize minority misclassification more — no new samples created. SMOTE often performs better on complex boundaries; class_weight is faster and simpler. Using both together can overfit.

---

## 📋 Quick Metrics to Remember

| Metric              | Your Model (approx) | Why it matters             |
|---------------------|---------------------|----------------------------|
| Accuracy            | ~99%                | Misleading (imbalance)     |
| F1 Score (Fraud)    | ~0.91               | Primary metric             |
| AUC-ROC             | ~0.98               | Model discrimination power |
| AUC-PR              | ~0.87               | Imbalance-aware metric     |
| Recall (Fraud)      | ~0.89               | Catching actual fraud      |
| Precision (Fraud)   | ~0.93               | Reducing false alarms      |
