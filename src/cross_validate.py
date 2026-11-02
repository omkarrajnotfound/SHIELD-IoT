"""
Phase C: 5-Fold Stratified Cross Validation
Validates the statistical stability and generalizability of Our Proposed
14-Feature Stealth-Aware LightGBM (Borderline-SMOTE + Cost-Sensitive)
across the full 801,519 NF-ToN-IoT-v2 dataset.

Outputs:
  - results/cross_validation_results.csv
  - figures/cross_validation_stability.png
"""
import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from lightgbm import LGBMClassifier
from imblearn.over_sampling import BorderlineSMOTE

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def run_cross_validation(n_splits=5):
    print("=" * 80)
    print("PHASE C: 5-FOLD STRATIFIED CROSS VALIDATION (STATISTICAL RIGOR)")
    print("=" * 80)

    # 1. Load Data, Targets, Encoder & Selected 14 Features
    data_path = os.path.join(PROCESSED_DIR, "train_test_data.pkl")
    target_path = os.path.join(PROCESSED_DIR, "train_test_multi_targets.pkl")
    encoder_path = os.path.join(PROCESSED_DIR, "label_encoder_multi.pkl")
    features_path = os.path.join(PROCESSED_DIR, "selected_features.json")

    print("[Step C1] Loading complete dataset and 14 selected features...")
    X_train_df, X_test_df, _, _ = joblib.load(data_path)
    y_train_s, y_test_s = joblib.load(target_path)
    label_encoder = joblib.load(encoder_path)
    classes = list(label_encoder.classes_)

    with open(features_path, "r", encoding="utf-8") as f:
        selected_features = json.load(f)["final_features"]

    # Combine into full 801,519 dataset with 14 features
    X_full = pd.concat([X_train_df[selected_features], X_test_df[selected_features]], axis=0).reset_index(drop=True)
    y_full = pd.concat([y_train_s, y_test_s], axis=0).reset_index(drop=True)

    print(f"  Total Benchmark Records: {len(X_full):,} network flows")
    print(f"  Feature Dimension:       {X_full.shape[1]} features")
    print(f"  Class Categories:        {len(classes)} ({', '.join(classes[:5])}...)")

    # Indices of minority classes
    idx_backdoor = classes.index("backdoor")
    idx_mitm = classes.index("mitm")
    idx_ransomware = classes.index("ransomware")

    oversample_strategy = {
        idx_backdoor: 4000,
        idx_mitm: 3500,
        idx_ransomware: 3000
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    print("\n" + "-" * 80)
    print(f"Executing {n_splits}-Fold Stratified Cross Validation...")
    print("-" * 80)

    total_cv_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full), start=1):
        t0_fold = time.time()
        print(f"\n[Fold {fold}/{n_splits}]")
        X_tr = X_full.iloc[train_idx]
        y_tr = y_full.iloc[train_idx]
        X_val = X_full.iloc[val_idx]
        y_val = y_full.iloc[val_idx]

        print(f"  Training partition:   {len(X_tr):,} flows | Validation: {len(X_val):,} flows")
        print(f"  Applying Borderline-SMOTE strictly to training fold (zero-leakage)...")
        t0_smote = time.time()
        bsmote = BorderlineSMOTE(sampling_strategy=oversample_strategy, random_state=42 + fold)
        X_tr_res, y_tr_res = bsmote.fit_resample(X_tr, y_tr)
        smote_time = time.time() - t0_smote
        print(f"  SMOTE completed in {smote_time:.2f}s (New train size: {len(X_tr_res):,} flows)")

        print(f"  Training Cost-Sensitive LightGBM (class_weight='balanced')...")
        t0_fit = time.time()
        lgb = LGBMClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1
        )
        lgb.fit(X_tr_res, y_tr_res)
        fit_time = time.time() - t0_fit

        # Predict on pure validation fold
        t0_pred = time.time()
        y_pred = lgb.predict(X_val)
        pred_time = time.time() - t0_pred
        latency_1k = (pred_time / len(X_val)) * 1000 * 1000

        acc = accuracy_score(y_val, y_pred) * 100
        macro_f1 = f1_score(y_val, y_pred, average="macro") * 100
        weighted_f1 = f1_score(y_val, y_pred, average="weighted") * 100
        rep = classification_report(y_val, y_pred, target_names=classes, output_dict=True, zero_division=0)

        ransomware_rec = rep["ransomware"]["recall"] * 100
        backdoor_rec = rep["backdoor"]["recall"] * 100
        mitm_rec = rep["mitm"]["recall"] * 100
        benign_rec = rep["Benign"]["recall"] * 100
        ddos_rec = rep["ddos"]["recall"] * 100

        fold_time = time.time() - t0_fold
        print(f"  Fold {fold} Results: Acc={acc:.2f}%, Macro-F1={macro_f1:.2f}%, Ransomware Recall={ransomware_rec:.2f}%, Time={fold_time:.1f}s")

        fold_metrics.append({
            "fold": f"Fold {fold}",
            "accuracy": round(acc, 2),
            "macro_f1": round(macro_f1, 2),
            "weighted_f1": round(weighted_f1, 2),
            "ransomware_recall": round(ransomware_rec, 2),
            "backdoor_recall": round(backdoor_rec, 2),
            "mitm_recall": round(mitm_rec, 2),
            "benign_recall": round(benign_rec, 2),
            "ddos_recall": round(ddos_rec, 2),
            "latency_ms_per_1k": round(latency_1k, 3),
            "fold_time_sec": round(fold_time, 2)
        })

    total_cv_time = time.time() - total_cv_start
    print(f"\nAll {n_splits} folds completed in {total_cv_time:.2f}s ({total_cv_time/60:.2f} minutes)!")

    # -------------------------------------------------------------
    # Compute Mean and Standard Deviation (Statistical Rigor)
    # -------------------------------------------------------------
    df_folds = pd.DataFrame(fold_metrics)

    metric_cols = [
        "accuracy", "macro_f1", "weighted_f1",
        "ransomware_recall", "backdoor_recall", "mitm_recall",
        "benign_recall", "ddos_recall", "latency_ms_per_1k"
    ]

    mean_row = {"fold": "MEAN"}
    std_row = {"fold": "STD (±)"}

    for col in metric_cols:
        mean_row[col] = round(float(df_folds[col].mean()), 2)
        std_row[col] = round(float(df_folds[col].std()), 2)

    mean_row["fold_time_sec"] = round(float(df_folds["fold_time_sec"].mean()), 2)
    std_row["fold_time_sec"] = round(float(df_folds["fold_time_sec"].std()), 2)

    df_full = pd.concat([df_folds, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    # Save to CSV
    csv_path = os.path.join(RESULTS_DIR, "cross_validation_results.csv")
    df_full.to_csv(csv_path, index=False)
    print(f"\nSaved cross validation results to: {csv_path}")

    # Display clean table
    print("\n" + "=" * 105)
    print("5-FOLD STRATIFIED CROSS VALIDATION STATISTICAL SUMMARY")
    print("=" * 105)
    disp_cols = ["fold", "accuracy", "macro_f1", "ransomware_recall", "backdoor_recall", "benign_recall", "ddos_recall", "latency_ms_per_1k"]
    print(df_full[disp_cols].to_string(index=False))
    print("=" * 105)

    print(f"\nStatistical Consistency Report for IEEE:")
    print(f"  Accuracy:          {mean_row['accuracy']:.2f}% ± {std_row['accuracy']:.2f}%")
    print(f"  Macro-F1 Score:    {mean_row['macro_f1']:.2f}% ± {std_row['macro_f1']:.2f}%")
    print(f"  Ransomware Recall: {mean_row['ransomware_recall']:.2f}% ± {std_row['ransomware_recall']:.2f}%")
    print(f"  Backdoor Recall:   {mean_row['backdoor_recall']:.2f}% ± {std_row['backdoor_recall']:.2f}%")

    # Generate Chart
    chart_path = os.path.join(FIGURES_DIR, "cross_validation_stability.png")
    generate_cv_chart(df_folds, mean_row, std_row, chart_path)

    return df_full


def generate_cv_chart(df_folds, mean_row, std_row, save_path):
    plt.figure(figsize=(13, 5), dpi=300)

    # Subplot 1: Fold-wise Accuracy & Macro-F1 with Mean Line
    plt.subplot(1, 2, 1)
    folds = [f"F{i+1}" for i in range(len(df_folds))]
    x = np.arange(len(folds))
    width = 0.35

    plt.bar(x - width/2, df_folds["accuracy"], width, label="Accuracy (%)", color="#4c72b0", alpha=0.85)
    plt.bar(x + width/2, df_folds["macro_f1"], width, label="Macro-F1 (%)", color="#55a868", alpha=0.9)

    plt.axhline(mean_row["accuracy"], color="#1f4e79", linestyle="--", linewidth=1.5,
                label=f"Mean Acc: {mean_row['accuracy']:.2f}% ± {std_row['accuracy']:.2f}%")
    plt.axhline(mean_row["macro_f1"], color="#20603d", linestyle=":", linewidth=1.5,
                label=f"Mean Macro-F1: {mean_row['macro_f1']:.2f}% ± {std_row['macro_f1']:.2f}%")

    plt.xlabel("Stratified Fold", fontweight="bold")
    plt.ylabel("Score (%)", fontweight="bold")
    plt.ylim(80, 102)
    plt.xticks(x, folds, fontweight="bold")
    plt.title("A: Classification Metric Stability Across 5 Folds", fontweight="bold", fontsize=11)
    plt.legend(loc="lower right", fontsize=8, frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Stealth Attack Recall Across Folds
    plt.subplot(1, 2, 2)
    plt.bar(x - width/2, df_folds["ransomware_recall"], width, label="Ransomware Recall (%)", color="#c44e52", alpha=0.85)
    plt.bar(x + width/2, df_folds["backdoor_recall"], width, label="Backdoor Recall (%)", color="#8172b3", alpha=0.9)

    plt.axhline(mean_row["ransomware_recall"], color="#800000", linestyle="--", linewidth=1.5,
                label=f"Mean Ransomware: {mean_row['ransomware_recall']:.2f}% ± {std_row['ransomware_recall']:.2f}%")
    plt.axhline(mean_row["backdoor_recall"], color="#4b0082", linestyle=":", linewidth=1.5,
                label=f"Mean Backdoor: {mean_row['backdoor_recall']:.2f}% ± {std_row['backdoor_recall']:.2f}%")

    plt.xlabel("Stratified Fold", fontweight="bold")
    plt.ylabel("Detection Recall Rate (%)", fontweight="bold")
    plt.ylim(70, 105)
    plt.xticks(x, folds, fontweight="bold")
    plt.title("B: Rare Attack Recall Stability (Zero-Leakage)", fontweight="bold", fontsize=11)
    plt.legend(loc="lower right", fontsize=8, frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Cross-validation stability chart saved to: {save_path}")


if __name__ == "__main__":
    run_cross_validation()

