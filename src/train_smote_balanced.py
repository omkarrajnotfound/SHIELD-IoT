"""
Phase A: Rare Attack Mitigation via Borderline-SMOTE and Cost-Sensitive LightGBM
Addresses extreme class imbalance on NF-ToN-IoT-v2 (e.g., Ransomware 0.02%, MITM 0.04%, Backdoor 0.10%).
Trains and compares:
  1. Standard Unbalanced Baseline LightGBM (14 features)
  2. Proposed Borderline-SMOTE + Cost-Sensitive LightGBM (14 features)
Generates:
  - models/lightgbm_smote_balanced_top14.joblib
  - results/per_class_metrics.csv
  - results/phase_a_summary.json
  - figures/rare_attack_recall_comparison.png
"""
import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from imblearn.over_sampling import BorderlineSMOTE

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def run_phase_a():
    print("=" * 70)
    print("PHASE A: RARE ATTACK FIX (BORDERLINE-SMOTE + COST-SENSITIVE LIGHTGBM)")
    print("=" * 70)

    # 1. Load Data, Targets, Encoder & Selected 14 Features
    data_path = os.path.join(PROCESSED_DIR, "train_test_data.pkl")
    target_path = os.path.join(PROCESSED_DIR, "train_test_multi_targets.pkl")
    encoder_path = os.path.join(PROCESSED_DIR, "label_encoder_multi.pkl")
    features_path = os.path.join(PROCESSED_DIR, "selected_features.json")

    print("[Step 1] Loading preprocessed data and selected 14 critical features...")
    X_train_full, X_test_full, _, _ = joblib.load(data_path)
    y_train, y_test = joblib.load(target_path)
    label_encoder = joblib.load(encoder_path)
    classes = list(label_encoder.classes_)

    with open(features_path, "r", encoding="utf-8") as f:
        selected_features = json.load(f)["final_features"]

    X_train = X_train_full[selected_features]
    X_test = X_test_full[selected_features]

    print(f"  Training Flows:  {len(X_train):,}")
    print(f"  Testing Flows:   {len(X_test):,}")
    print(f"  Feature Subset:  {len(selected_features)} features ({', '.join(selected_features[:4])}...)")
    print(f"  Target Classes:  {len(classes)} categories: {classes}")

    # 2. Train Unbalanced Baseline (Before SMOTE & Cost-Weighting)
    print("\n" + "-" * 70)
    print("[Step 2] Training Baseline LightGBM (Before: Standard 14-Feature)...")
    print("-" * 70)
    t0_base = time.time()
    lgb_base = LGBMClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_base.fit(X_train, y_train)
    train_time_base = time.time() - t0_base

    t0_infer_base = time.time()
    y_pred_base = lgb_base.predict(X_test)
    infer_time_base = time.time() - t0_infer_base
    latency_base_1k = (infer_time_base / len(X_test)) * 1000 * 1000

    report_base = classification_report(y_test, y_pred_base, target_names=classes, output_dict=True)
    acc_base = accuracy_score(y_test, y_pred_base) * 100
    macro_f1_base = f1_score(y_test, y_pred_base, average="macro") * 100
    print(f"  Baseline Training Time:  {train_time_base:.2f}s")
    print(f"  Baseline Test Accuracy:  {acc_base:.2f}%")
    print(f"  Baseline Macro-F1:       {macro_f1_base:.2f}%")
    print(f"  Ransomware Recall:       {report_base['ransomware']['recall']*100:.2f}%")
    print(f"  Backdoor Recall:         {report_base['backdoor']['recall']*100:.2f}%")
    print(f"  MITM Recall:             {report_base['mitm']['recall']*100:.2f}%")

    # 3. Apply Borderline-SMOTE on Rare Minority Classes
    print("\n" + "-" * 70)
    print("[Step 3] Applying Borderline-SMOTE on Minority Attack Classes...")
    print("-" * 70)
    # Rare class indices
    idx_backdoor = classes.index("backdoor")
    idx_mitm = classes.index("mitm")
    idx_ransomware = classes.index("ransomware")

    print(f"  Original Training Counts:")
    print(f"    - Ransomware (class {idx_ransomware}): {(y_train == idx_ransomware).sum()} flows")
    print(f"    - MITM (class {idx_mitm}):       {(y_train == idx_mitm).sum()} flows")
    print(f"    - Backdoor (class {idx_backdoor}):   {(y_train == idx_backdoor).sum()} flows")

    oversample_strategy = {
        idx_backdoor: 4000,
        idx_mitm: 3500,
        idx_ransomware: 3000
    }

    t0_smote = time.time()
    bsmote = BorderlineSMOTE(sampling_strategy=oversample_strategy, random_state=42)
    X_train_resampled, y_train_resampled = bsmote.fit_resample(X_train, y_train)
    smote_time = time.time() - t0_smote
    print(f"  Borderline-SMOTE finished in {smote_time:.2f}s")
    print(f"  New Resampled Training Set: {len(X_train_resampled):,} flows")
    print(f"    - Ransomware: {(y_train_resampled == idx_ransomware).sum():,} synthetic + real flows")
    print(f"    - MITM:       {(y_train_resampled == idx_mitm).sum():,} synthetic + real flows")
    print(f"    - Backdoor:   {(y_train_resampled == idx_backdoor).sum():,} synthetic + real flows")

    # 4. Train Proposed Model (Cost-Sensitive LightGBM on Resampled Data)
    print("\n" + "-" * 70)
    print("[Step 4] Training Proposed Cost-Sensitive LightGBM (class_weight='balanced')...")
    print("-" * 70)
    t0_prop = time.time()
    lgb_prop = LGBMClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_prop.fit(X_train_resampled, y_train_resampled)
    train_time_prop = time.time() - t0_prop

    t0_infer_prop = time.time()
    y_pred_prop = lgb_prop.predict(X_test)
    infer_time_prop = time.time() - t0_infer_prop
    latency_prop_1k = (infer_time_prop / len(X_test)) * 1000 * 1000

    report_prop = classification_report(y_test, y_pred_prop, target_names=classes, output_dict=True)
    acc_prop = accuracy_score(y_test, y_pred_prop) * 100
    macro_f1_prop = f1_score(y_test, y_pred_prop, average="macro") * 100
    print(f"  Proposed Model Training Time: {train_time_prop:.2f}s")
    print(f"  Proposed Model Test Accuracy: {acc_prop:.2f}%")
    print(f"  Proposed Model Macro-F1:      {macro_f1_prop:.2f}%")
    print(f"  Ransomware Recall:            {report_prop['ransomware']['recall']*100:.2f}%")
    print(f"  Backdoor Recall:              {report_prop['backdoor']['recall']*100:.2f}%")
    print(f"  MITM Recall:                  {report_prop['mitm']['recall']*100:.2f}%")

    # 5. Save Trained Model Artifact
    prop_model_path = os.path.join(MODELS_DIR, "lightgbm_smote_balanced_top14.joblib")
    joblib.dump(lgb_prop, prop_model_path)
    model_size_mb = os.path.getsize(prop_model_path) / (1024 * 1024)
    print(f"\nSaved proposed model artifact to: {prop_model_path} ({model_size_mb:.2f} MB)")

    # 6. Generate Comprehensive Per-Class Comparison CSV
    metrics_list = []
    for cls in classes:
        support = report_base[cls]["support"]
        rec_before = report_base[cls]["recall"] * 100
        prec_before = report_base[cls]["precision"] * 100
        f1_before = report_base[cls]["f1-score"] * 100

        rec_after = report_prop[cls]["recall"] * 100
        prec_after = report_prop[cls]["precision"] * 100
        f1_after = report_prop[cls]["f1-score"] * 100

        diff_rec = rec_after - rec_before

        metrics_list.append({
            "attack_class": cls,
            "test_samples": int(support),
            "recall_before": round(rec_before, 2),
            "recall_after": round(rec_after, 2),
            "recall_improvement": round(diff_rec, 2),
            "precision_before": round(prec_before, 2),
            "precision_after": round(prec_after, 2),
            "f1_before": round(f1_before, 2),
            "f1_after": round(f1_after, 2)
        })

    df_metrics = pd.DataFrame(metrics_list)
    csv_metrics_path = os.path.join(RESULTS_DIR, "per_class_metrics.csv")
    df_metrics.to_csv(csv_metrics_path, index=False)
    print(f"Saved per-class comparison to: {csv_metrics_path}")

    # Display comparison table
    print("\n" + "=" * 90)
    print(f"{'Attack Class':<14} | {'Samples':<8} | {'Recall Before':<14} | {'Recall After':<14} | {'Improvement':<12} | {'F1 After':<10}")
    print("-" * 90)
    for row in metrics_list:
        sign = "+" if row["recall_improvement"] >= 0 else ""
        print(f"{row['attack_class']:<14} | {row['test_samples']:<8} | {row['recall_before']:>12.2f}% | {row['recall_after']:>12.2f}% | {sign+str(row['recall_improvement'])+'%':>12} | {row['f1_after']:>8.2f}%")
    print("=" * 90)

    # 7. Save Phase A Summary Metadata
    summary_data = {
        "phase": "Phase A: Rare Attack Fix",
        "features_count": 14,
        "selected_features": selected_features,
        "oversample_strategy": {classes[k]: v for k, v in oversample_strategy.items()},
        "baseline_model": {
            "name": "Standard LightGBM (14-Feat, No SMOTE)",
            "accuracy": round(acc_base, 2),
            "macro_f1": round(macro_f1_base, 2),
            "weighted_f1": round(report_base["weighted avg"]["f1-score"] * 100, 2),
            "latency_ms_per_1k": round(latency_base_1k, 4),
            "train_time_sec": round(train_time_base, 2)
        },
        "proposed_model": {
            "name": "Borderline-SMOTE + Cost-Sensitive LightGBM (14-Feat)",
            "accuracy": round(acc_prop, 2),
            "macro_f1": round(macro_f1_prop, 2),
            "weighted_f1": round(report_prop["weighted avg"]["f1-score"] * 100, 2),
            "latency_ms_per_1k": round(latency_prop_1k, 4),
            "train_time_sec": round(train_time_prop, 2),
            "model_size_mb": round(model_size_mb, 2)
        },
        "minority_attack_recall": {
            "ransomware": {
                "before": round(report_base["ransomware"]["recall"] * 100, 2),
                "after": round(report_prop["ransomware"]["recall"] * 100, 2),
                "f1_after": round(report_prop["ransomware"]["f1-score"] * 100, 2)
            },
            "backdoor": {
                "before": round(report_base["backdoor"]["recall"] * 100, 2),
                "after": round(report_prop["backdoor"]["recall"] * 100, 2),
                "f1_after": round(report_prop["backdoor"]["f1-score"] * 100, 2)
            },
            "mitm": {
                "before": round(report_base["mitm"]["recall"] * 100, 2),
                "after": round(report_prop["mitm"]["recall"] * 100, 2),
                "f1_after": round(report_prop["mitm"]["f1-score"] * 100, 2)
            }
        }
    }
    summary_path = os.path.join(RESULTS_DIR, "phase_a_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved Phase A metadata summary to: {summary_path}")

    # 8. Generate High-Resolution Publication Chart (Before vs After Recall)
    plot_path = os.path.join(FIGURES_DIR, "rare_attack_recall_comparison.png")
    generate_comparison_plot(df_metrics, macro_f1_base, macro_f1_prop, plot_path)

    return df_metrics, summary_data

def generate_comparison_plot(df, macro_f1_base, macro_f1_prop, save_path):
    plt.figure(figsize=(12, 6), dpi=300)
    x = np.arange(len(df))
    width = 0.38

    plt.bar(x - width/2, df["recall_before"], width, label="Standard Baseline LightGBM", color="#4c72b0", alpha=0.85)
    plt.bar(x + width/2, df["recall_after"], width, label="Proposed Borderline-SMOTE + Cost-Sensitive", color="#2ca02c", alpha=0.95)

    plt.xlabel("Attack Category", fontweight="bold", fontsize=11)
    plt.ylabel("Recall Rate (%)", fontweight="bold", fontsize=11)
    plt.title("Per-Class Detection Recall: Standard vs. Proposed Stealth-Aware LightGBM\n(NF-ToN-IoT-v2, 14 Critical Features)", fontweight="bold", fontsize=12)
    plt.xticks(x, df["attack_class"], rotation=30, ha="right", fontsize=10)
    plt.ylim(0, 115)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Highlight rare attack improvements
    for i, row in df.iterrows():
        y_aft = row["recall_after"]
        y_bef = row["recall_before"]
        if row["attack_class"] in ["ransomware", "backdoor", "mitm"]:
            plt.annotate(
                f"+{row['recall_improvement']:.1f}%",
                (i + width/2, y_aft + 2.5),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#b22222"
            )
        else:
            plt.text(i + width/2, y_aft + 1.5, f"{y_aft:.1f}%", ha="center", fontsize=8)

    plt.legend(loc="upper right", frameon=True, fontsize=10)
    
    # Text box with Macro-F1 improvement
    text_str = f"Macro-F1 Improvement:\n  Baseline: {macro_f1_base:.2f}%\n  Proposed: {macro_f1_prop:.2f}% (+{macro_f1_prop - macro_f1_base:.2f}%)"
    props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#2ca02c", alpha=0.9, lw=1.5)
    plt.gca().text(0.02, 0.95, text_str, transform=plt.gca().transAxes, fontsize=10, verticalalignment="top", bbox=props)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Generated IEEE publication chart at: {save_path}")

if __name__ == "__main__":
    run_phase_a()

