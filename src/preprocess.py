"""
Data Preprocessing Pipeline for NF-ToN-IoT-v2 (Refined with robust numerical bounds)
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
import joblib

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

DROP_COLUMNS = [
    "IPV4_SRC_ADDR", 
    "IPV4_DST_ADDR", 
    "L4_SRC_PORT", 
    "L4_DST_PORT"
]

FLOAT32_MAX = np.finfo(np.float32).max * 0.95
FLOAT32_MIN = np.finfo(np.float32).min * 0.95

def load_and_preprocess(csv_path=None, test_size=0.2, random_state=42):
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "nf_ton_iot_sample.csv")
    
    print("\n" + "="*60)
    print("PHASE 2: PREPROCESSING PIPELINE (NUMERICAL BOUND FIX)")
    print("="*60)
    print(f"Loading raw dataset from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Drop non-generalizable network identifiers
    drop_existing = [c for c in DROP_COLUMNS if c in df.columns]
    if drop_existing:
        print(f"Dropping non-generalizable network identifiers: {drop_existing}")
        df = df.drop(columns=drop_existing)

    target_binary = "Label" if "Label" in df.columns else "label"
    target_multi = "Attack" if "Attack" in df.columns else "attack"

    # Replace infs with nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    drop_from_X = [target_binary]
    if target_multi in df.columns:
        drop_from_X.append(target_multi)
    
    X = df.drop(columns=drop_from_X)
    y_binary = df[target_binary].astype(int)

    # Encode any categorical columns in X if present
    for col in X.columns:
        if X[col].dtype == "object":
            print(f"Encoding categorical column: {col}")
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Stratified Train-Test Split (80% Train, 20% Test)
    y_multi = None
    if target_multi in df.columns:
        le = LabelEncoder()
        y_multi = le.fit_transform(df[target_multi].astype(str))
        joblib.dump(le, os.path.join(PROCESSED_DIR, "label_encoder_multi.pkl"))
        print(f"Multi-class classes found: {list(le.classes_)}")

    stratify_target = y_multi if y_multi is not None else y_binary
    if y_multi is not None:
        X_train, X_test, y_train, y_test, y_train_multi, y_test_multi = train_test_split(
            X, y_binary, y_multi, test_size=test_size, random_state=random_state, stratify=stratify_target
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=random_state, stratify=stratify_target
        )
        y_train_multi, y_test_multi = None, None

    print(f"Training set: {X_train.shape[0]:,} samples x {X_train.shape[1]} features")
    print(f"Testing set:  {X_test.shape[0]:,} samples x {X_test.shape[1]} features")

    # Fit RobustScaler
    print("\nApplying RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Clean clip to valid float32 range and replace any NaN/inf produced by 0 IQR
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=FLOAT32_MAX, neginf=FLOAT32_MIN)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=FLOAT32_MAX, neginf=FLOAT32_MIN)

    X_train_scaled = np.clip(X_train_scaled, FLOAT32_MIN, FLOAT32_MAX).astype(np.float32)
    X_test_scaled = np.clip(X_test_scaled, FLOAT32_MIN, FLOAT32_MAX).astype(np.float32)

    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Save artifacts
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "robust_scaler.pkl"))
    joblib.dump((X_train_df, X_test_df, y_train, y_test), os.path.join(PROCESSED_DIR, "train_test_data.pkl"))
    if y_train_multi is not None:
        y_train_multi_s = pd.Series(y_train_multi, index=X_train.index)
        y_test_multi_s = pd.Series(y_test_multi, index=X_test.index)
        joblib.dump((y_train_multi_s, y_test_multi_s), os.path.join(PROCESSED_DIR, "train_test_multi_targets.pkl"))
    print("Preprocessing completed and numerical bounds verified!")

    return X_train_df, X_test_df, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess()
