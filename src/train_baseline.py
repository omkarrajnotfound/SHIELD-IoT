"""
Phase 3: Baseline Model Training (Full Feature Space)
Trains standard benchmark classifiers on all 39 network flow features to establish comparative metrics:
Accuracy, Precision, Recall, Macro F1, Inference Latency (ms/1k packets), and Model Size.
"""
import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\n--- Training {name} on Full Feature Set ({X_train.shape[1]} features) ---")
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    print(f"  Training finished in: {train_time:.2f} seconds")

    # Measure pure inference latency over test set
    start_infer = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - start_infer
    
    # Latency per 1,000 packets (in milliseconds)
    latency_per_1k_ms = (infer_time / len(X_test)) * 1000 * 1000
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Save model artifact and measure disk footprint
    model_path = os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}_full.joblib")
    joblib.dump(model, model_path)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    metrics = {
        "model_name": name,
        "feature_count": int(X_train.shape[1]),
        "accuracy": round(float(acc) * 100, 2),
        "precision": round(float(prec) * 100, 2),
        "recall": round(float(rec) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "train_time_sec": round(float(train_time), 2),
        "infer_time_sec": round(float(infer_time), 4),
        "latency_ms_per_1k_pkts": round(float(latency_per_1k_ms), 2),
        "model_size_mb": round(float(model_size_mb), 2)
    }

    print(f"  Accuracy:         {metrics['accuracy']}%")
    print(f"  F1-Score:         {metrics['f1_score']}%")
    print(f"  Latency / 1k:     {metrics['latency_ms_per_1k_pkts']} ms")
    print(f"  Model Size:       {metrics['model_size_mb']} MB")
    return metrics

def run_baselines(sample_train_size=100000, sample_test_size=30000):
    print("=" * 60)
    print("PHASE 3: BASELINE MODEL BENCHMARKING (FULL FEATURES)")
    print("=" * 60)
    
    data_path = os.path.join(PROCESSED_DIR, "train_test_data.pkl")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}. Run preprocess.py first.")

    X_train, X_test, y_train, y_test = joblib.load(data_path)
    
    # Subsample for fast and reliable baseline benchmarking
    if len(X_train) > sample_train_size:
        print(f"Subsampling {sample_train_size:,} train flows for CPU benchmark efficiency...")
        idx_train = np.random.RandomState(42).choice(len(X_train), sample_train_size, replace=False)
        X_train_sub = X_train.iloc[idx_train]
        y_train_sub = y_train.iloc[idx_train]
    else:
        X_train_sub, y_train_sub = X_train, y_train

    if len(X_test) > sample_test_size:
        print(f"Subsampling {sample_test_size:,} test flows for exact latency timing...")
        idx_test = np.random.RandomState(42).choice(len(X_test), sample_test_size, replace=False)
        X_test_sub = X_test.iloc[idx_test]
        y_test_sub = y_test.iloc[idx_test]
    else:
        X_test_sub, y_test_sub = X_test, y_test

    all_metrics = []

    # 1. Random Forest Baseline
    rf = RandomForestClassifier(n_estimators=60, max_depth=16, random_state=42, n_jobs=-1)
    rf_metrics = evaluate_model("Random Forest (Full Features)", rf, X_train_sub, y_train_sub, X_test_sub, y_test_sub)
    all_metrics.append(rf_metrics)

    # 2. XGBoost Baseline
    xgb = XGBClassifier(n_estimators=80, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric="logloss")
    xgb_metrics = evaluate_model("XGBoost (Full Features)", xgb, X_train_sub, y_train_sub, X_test_sub, y_test_sub)
    all_metrics.append(xgb_metrics)

    # 3. LightGBM Baseline
    lgb = LGBMClassifier(n_estimators=80, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    lgb_metrics = evaluate_model("LightGBM (Full Features)", lgb, X_train_sub, y_train_sub, X_test_sub, y_test_sub)
    all_metrics.append(lgb_metrics)

    # Save summary table
    df_results = pd.DataFrame(all_metrics)
    summary_csv = os.path.join(RESULTS_DIR, "baseline_full_feature_metrics.csv")
    df_results.to_csv(summary_csv, index=False)
    print(f"\nBaseline results successfully saved to: {summary_csv}")
    print("\nSummary Comparison Table:")
    print(df_results[["model_name", "feature_count", "accuracy", "f1_score", "latency_ms_per_1k_pkts", "model_size_mb"]].to_string(index=False))

if __name__ == "__main__":
    run_baselines()

