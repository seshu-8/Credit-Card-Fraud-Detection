"""
preprocess.py
Handles data loading, cleaning, scaling, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load] Shape: {df.shape}")
    print(f"[load] Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna()
    after = len(df)
    print(f"[clean] Removed {before - after} rows | Final: {after}")
    return df


def scale_features(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """Scale Amount and Time; V1–V28 already scaled (PCA-like)."""
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    scaler = StandardScaler()

    cols_to_scale = ["Amount", "Time"]
    df = df.copy()

    if fit:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        joblib.dump(scaler, scaler_path)
        print(f"[scale] Scaler saved → {scaler_path}")
    else:
        scaler = joblib.load(scaler_path)
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        print("[scale] Existing scaler applied.")

    return df


def split_data(df: pd.DataFrame, target: str = "Class", test_size: float = 0.2):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"[split] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[split] Train fraud %: {y_train.mean()*100:.2f}%")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    """Balance classes using SMOTE (Synthetic Minority Oversampling)."""
    sm = SMOTE(random_state=42, sampling_strategy=0.5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[SMOTE] Before: {dict(y_train.value_counts())}")
    print(f"[SMOTE] After : {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res


def full_pipeline(data_path: str):
    df = load_data(data_path)
    df = clean_data(df)
    df = scale_features(df, fit=True)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    return X_train_res, X_test, y_train_res, y_test
