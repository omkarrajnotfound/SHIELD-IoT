"""
Phase 5: Lightweight Edge Model Optimization & Comparative Benchmarking
Trains classifiers on the selected 14 critical features and benchmarks them directly
against the full 39-feature baselines (Accuracy, F1, Latency, and Memory Footprint).
"""
import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def evaluate_pruned_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\n--- Training {name} on Pruned Subset ({X_train.shape[1]} features) ---")
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    # Measure pure inference latency over test set
    start_infer = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - start_infer
    
    latency_per_1k_ms = (infer_time / len(X_test)) * 1000 * 1000
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    model_path = os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}_top14.joblib")
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
    return metrics, model

def run_comparative_benchmark(sample_train_size=100000, sample_test_size=30000):
    print("=" * 60)
    print("PHASE 5: LIGHTWEIGHT MODEL BENCHMARKING (14 FEATURES)")
    print("=" * 60)

    # Load data & selected features
    data_path = os.path.join(PROCESSED_DIR, "train_test_data.pkl")
    feat_path = os.path.join(PROCESSED_DIR, "selected_features.json")
    
    X_train, X_test, y_train, y_test = joblib.load(data_path)
    with open(feat_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    top14_features = meta["final_features"]

    X_train_top14 = X_train[top14_features]
    X_test_top14 = X_test[top14_features]

    # Subsample for consistency
    idx_train = np.random.RandomState(42).choice(len(X_train), sample_train_size, replace=False)
    idx_test = np.random.RandomState(42).choice(len(X_test), sample_test_size, replace=False)

    X_tr = X_train_top14.iloc[idx_train]
    y_tr = y_train.iloc[idx_train]
    X_te = X_test_top14.iloc[idx_test]
    y_te = y_test.iloc[idx_test]

    pruned_metrics = []

    # 1. Random Forest (14 Features)
    rf = RandomForestClassifier(n_estimators=60, max_depth=16, random_state=42, n_jobs=-1)
    rf_m, _ = evaluate_pruned_model("Random Forest (Proposed 14-Feat)", rf, X_tr, y_tr, X_te, y_te)
    pruned_metrics.append(rf_m)

    # 2. XGBoost (14 Features)
    xgb = XGBClassifier(n_estimators=80, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric="logloss")
    xgb_m, _ = evaluate_pruned_model("XGBoost (Proposed 14-Feat)", xgb, X_tr, y_tr, X_te, y_te)
    pruned_metrics.append(xgb_m)

    # 3. LightGBM (14 Features) - Our primary candidate
    lgb = LGBMClassifier(n_estimators=80, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    lgb_m, best_lgb = evaluate_pruned_model("LightGBM (Proposed 14-Feat)", lgb, X_tr, y_tr, X_te, y_te)
    pruned_metrics.append(lgb_m)

    # Combine with baseline full-feature results
    baseline_csv = os.path.join(RESULTS_DIR, "baseline_full_feature_metrics.csv")
    df_base = pd.read_csv(baseline_csv)
    df_pruned = pd.DataFrame(pruned_metrics)
    
    df_combined = pd.concat([df_base, df_pruned], ignore_index=True)
    comp_csv = os.path.join(RESULTS_DIR, "comparative_benchmark_results.csv")
    df_combined.to_csv(comp_csv, index=False)
    
    print("\n" + "="*60)
    print("FINAL HEAD-TO-HEAD COMPARATIVE BENCHMARK")
    print("="*60)
    print(df_combined[["model_name", "feature_count", "accuracy", "f1_score", "latency_ms_per_1k_pkts", "model_size_mb"]].to_string(index=False))

    # Generate Comparative Visualization
    plot_comparisons(df_combined)

def plot_comparisons(df):
    plt.figure(figsize=(12, 5), dpi=300)
    
    # Subplot 1: Accuracy comparison
    plt.subplot(1, 2, 1)
    colors = ["#4c72b0", "#4c72b0", "#4c72b0", "#55a868", "#55a868", "#2ca02c"]
    bars = plt.bar(range(len(df)), df["accuracy"], color=colors, width=0.6)
    plt.xticks(range(len(df)), [n.replace(" (Full Features)", "\n(39-F)").replace(" (Proposed 14-Feat)", "\n(14-F)") for n in df["model_name"]], fontsize=8, rotation=25)
    plt.ylabel("Accuracy (%)", fontweight="bold")
    plt.ylim(95, 100)
    plt.title("Classification Accuracy Retention", fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.1, f"{h:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Subplot 2: Inference Latency comparison
    plt.subplot(1, 2, 2)
    lat_colors = ["#c44e52", "#c44e52", "#c44e52", "#8172b3", "#8172b3", "#4c72b0"]
    bars2 = plt.bar(range(len(df)), df["latency_ms_per_1k_pkts"], color=lat_colors, width=0.6)
    plt.xticks(range(len(df)), [n.replace(" (Full Features)", "\n(39-F)").replace(" (Proposed 14-Feat)", "\n(14-F)") for n in df["model_name"]], fontsize=8, rotation=25)
    plt.ylabel("Latency per 1k Packets (ms)", fontweight="bold")
    plt.title("Edge Inference Latency Comparison", fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.2f}ms", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    comp_fig = os.path.join(FIGURES_DIR, "head_to_head_comparison.png")
    plt.savefig(comp_fig)
    plt.close()
    print(f"\nComparative figure saved to: {comp_fig}")

if __name__ == "__main__":
    run_comparative_benchmark()
    # Execute rigorous controlled 39 vs 14 feature ablation
    try:
        from feature_ablation import run_ablation
    except ImportError:
        from src.feature_ablation import run_ablation
    run_ablation()

