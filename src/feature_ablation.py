"""
Controlled Feature Ablation Study: 39 Features vs. 14 Features
Isolates the exact empirical impact of Two-Stage Feature Selection (ANOVA + TreeSHAP)
by evaluating the identical LightGBM algorithm under identical data splits, resampling protocols,
and hyperparameter grids:
  1. Config A: 39 Features | Standard (No SMOTE, Unweighted)
  2. Config B: 14 Features | Standard (No SMOTE, Unweighted)
  3. Config C: 39 Features | Hardened (Borderline-SMOTE + Balanced Class Weights)
  4. Config D: 14 Features | Hardened (SHIELD-IoT: Borderline-SMOTE + Balanced Class Weights)

Outputs:
  - results/feature_ablation_results.csv
  - results/feature_ablation_summary.json
  - figures/feature_ablation_comparison.png
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
from lightgbm import LGBMClassifier
from imblearn.over_sampling import BorderlineSMOTE

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def run_ablation():
    print("=" * 85)
    print("CONTROLLED FEATURE ABLATION STUDY: 39 FEATURES vs. 14 FEATURES")
    print("=" * 85)

    # 1. Load Data, Targets, Encoder & Selected 14 Features
    t0_load = time.time()
    X_train_full, X_test_full, _, _ = joblib.load(os.path.join(PROCESSED_DIR, "train_test_data.pkl"))
    y_train, y_test = joblib.load(os.path.join(PROCESSED_DIR, "train_test_multi_targets.pkl"))
    label_encoder = joblib.load(os.path.join(PROCESSED_DIR, "label_encoder_multi.pkl"))
    classes = list(label_encoder.classes_)

    with open(os.path.join(PROCESSED_DIR, "selected_features.json"), "r", encoding="utf-8") as f:
        top14_features = json.load(f)["final_features"]

    X_train_14 = X_train_full[top14_features]
    X_test_14 = X_test_full[top14_features]

    print(f"Loaded {len(X_train_full):,} training flows and {len(X_test_full):,} test flows in {time.time()-t0_load:.2f}s.")
    print(f"Classes ({len(classes)}): {classes}")

    # Common LightGBM Hyperparameters
    lgb_params = {
        "n_estimators": 100,
        "max_depth": 8,
        "learning_rate": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }

    # Oversampling strategy for rare classes
    idx_backdoor = classes.index("backdoor")
    idx_mitm = classes.index("mitm")
    idx_ransomware = classes.index("ransomware")
    oversample_strategy = {
        idx_backdoor: 4000,
        idx_mitm: 3500,
        idx_ransomware: 3000
    }

    configs = []

    # -----------------------------------------------------------------
    # Configuration 1: 39 Features, Standard (Unweighted, No SMOTE)
    # -----------------------------------------------------------------
    print("\n[Config 1/4] Training LightGBM on 39 Features (Unweighted, No SMOTE)...")
    clf_39_std = LGBMClassifier(**lgb_params)
    t0 = time.time()
    clf_39_std.fit(X_train_full, y_train)
    fit_time_39_std = time.time() - t0

    t0_inf = time.time()
    preds_39_std = clf_39_std.predict(X_test_full)
    infer_sec_39_std = time.time() - t0_inf
    lat_39_std = (infer_sec_39_std / len(y_test)) * 1000 * 1000
    rep_39_std = classification_report(y_test, preds_39_std, target_names=classes, output_dict=True)

    configs.append({
        "configuration": "39 Features (Standard Unweighted)",
        "feature_count": 39,
        "defense_resampling": "None (Original Imbalanced)",
        "class_weighting": "None (Uniform)",
        "accuracy": round(accuracy_score(y_test, preds_39_std) * 100, 2),
        "macro_f1": round(f1_score(y_test, preds_39_std, average="macro") * 100, 2),
        "weighted_f1": round(f1_score(y_test, preds_39_std, average="weighted") * 100, 2),
        "ransomware_recall": round(rep_39_std["ransomware"]["recall"] * 100, 2),
        "backdoor_recall": round(rep_39_std["backdoor"]["recall"] * 100, 2),
        "mitm_recall": round(rep_39_std["mitm"]["recall"] * 100, 2),
        "train_time_sec": round(fit_time_39_std, 2),
        "latency_ms_per_1k": round(lat_39_std, 3)
    })
    print(f"  Acc: {configs[-1]['accuracy']}% | Macro-F1: {configs[-1]['macro_f1']}% | Ransomware: {configs[-1]['ransomware_recall']}%")

    # -----------------------------------------------------------------
    # Configuration 2: 14 Features, Standard (Unweighted, No SMOTE)
    # -----------------------------------------------------------------
    print("\n[Config 2/4] Training LightGBM on 14 Features (Unweighted, No SMOTE)...")
    clf_14_std = LGBMClassifier(**lgb_params)
    t0 = time.time()
    clf_14_std.fit(X_train_14, y_train)
    fit_time_14_std = time.time() - t0

    t0_inf = time.time()
    preds_14_std = clf_14_std.predict(X_test_14)
    infer_sec_14_std = time.time() - t0_inf
    lat_14_std = (infer_sec_14_std / len(y_test)) * 1000 * 1000
    rep_14_std = classification_report(y_test, preds_14_std, target_names=classes, output_dict=True)

    configs.append({
        "configuration": "14 Features (Standard Unweighted)",
        "feature_count": 14,
        "defense_resampling": "None (Original Imbalanced)",
        "class_weighting": "None (Uniform)",
        "accuracy": round(accuracy_score(y_test, preds_14_std) * 100, 2),
        "macro_f1": round(f1_score(y_test, preds_14_std, average="macro") * 100, 2),
        "weighted_f1": round(f1_score(y_test, preds_14_std, average="weighted") * 100, 2),
        "ransomware_recall": round(rep_14_std["ransomware"]["recall"] * 100, 2),
        "backdoor_recall": round(rep_14_std["backdoor"]["recall"] * 100, 2),
        "mitm_recall": round(rep_14_std["mitm"]["recall"] * 100, 2),
        "train_time_sec": round(fit_time_14_std, 2),
        "latency_ms_per_1k": round(lat_14_std, 3)
    })
    print(f"  Acc: {configs[-1]['accuracy']}% | Macro-F1: {configs[-1]['macro_f1']}% | Ransomware: {configs[-1]['ransomware_recall']}%")

    # -----------------------------------------------------------------
    # Configuration 3: 39 Features, Hardened (Borderline-SMOTE, Balanced)
    # -----------------------------------------------------------------
    print("\n[Config 3/4] Training LightGBM on 39 Features (Borderline-SMOTE + Cost-Sensitive)...")
    bsmote_39 = BorderlineSMOTE(sampling_strategy=oversample_strategy, random_state=42)
    X_train_res_39, y_train_res_39 = bsmote_39.fit_resample(X_train_full, y_train)

    clf_39_hard = LGBMClassifier(**lgb_params, class_weight="balanced")
    t0 = time.time()
    clf_39_hard.fit(X_train_res_39, y_train_res_39)
    fit_time_39_hard = time.time() - t0

    t0_inf = time.time()
    preds_39_hard = clf_39_hard.predict(X_test_full)
    infer_sec_39_hard = time.time() - t0_inf
    lat_39_hard = (infer_sec_39_hard / len(y_test)) * 1000 * 1000
    rep_39_hard = classification_report(y_test, preds_39_hard, target_names=classes, output_dict=True)

    configs.append({
        "configuration": "39 Features (Hardened Defense)",
        "feature_count": 39,
        "defense_resampling": "Borderline-SMOTE",
        "class_weighting": "Cost-Sensitive (Balanced)",
        "accuracy": round(accuracy_score(y_test, preds_39_hard) * 100, 2),
        "macro_f1": round(f1_score(y_test, preds_39_hard, average="macro") * 100, 2),
        "weighted_f1": round(f1_score(y_test, preds_39_hard, average="weighted") * 100, 2),
        "ransomware_recall": round(rep_39_hard["ransomware"]["recall"] * 100, 2),
        "backdoor_recall": round(rep_39_hard["backdoor"]["recall"] * 100, 2),
        "mitm_recall": round(rep_39_hard["mitm"]["recall"] * 100, 2),
        "train_time_sec": round(fit_time_39_hard, 2),
        "latency_ms_per_1k": round(lat_39_hard, 3)
    })
    print(f"  Acc: {configs[-1]['accuracy']}% | Macro-F1: {configs[-1]['macro_f1']}% | Ransomware: {configs[-1]['ransomware_recall']}%")

    # -----------------------------------------------------------------
    # Configuration 4: 14 Features, Hardened (SHIELD-IoT: SMOTE, Balanced)
    # -----------------------------------------------------------------
    print("\n[Config 4/4] Training LightGBM on 14 Features (SHIELD-IoT: Borderline-SMOTE + Cost-Sensitive)...")
    bsmote_14 = BorderlineSMOTE(sampling_strategy=oversample_strategy, random_state=42)
    X_train_res_14, y_train_res_14 = bsmote_14.fit_resample(X_train_14, y_train)

    clf_14_hard = LGBMClassifier(**lgb_params, class_weight="balanced")
    t0 = time.time()
    clf_14_hard.fit(X_train_res_14, y_train_res_14)
    fit_time_14_hard = time.time() - t0

    t0_inf = time.time()
    preds_14_hard = clf_14_hard.predict(X_test_14)
    infer_sec_14_hard = time.time() - t0_inf
    lat_14_hard = (infer_sec_14_hard / len(y_test)) * 1000 * 1000
    rep_14_hard = classification_report(y_test, preds_14_hard, target_names=classes, output_dict=True)

    configs.append({
        "configuration": "14 Features (Hardened SHIELD-IoT)",
        "feature_count": 14,
        "defense_resampling": "Borderline-SMOTE",
        "class_weighting": "Cost-Sensitive (Balanced)",
        "accuracy": round(accuracy_score(y_test, preds_14_hard) * 100, 2),
        "macro_f1": round(f1_score(y_test, preds_14_hard, average="macro") * 100, 2),
        "weighted_f1": round(f1_score(y_test, preds_14_hard, average="weighted") * 100, 2),
        "ransomware_recall": round(rep_14_hard["ransomware"]["recall"] * 100, 2),
        "backdoor_recall": round(rep_14_hard["backdoor"]["recall"] * 100, 2),
        "mitm_recall": round(rep_14_hard["mitm"]["recall"] * 100, 2),
        "train_time_sec": round(fit_time_14_hard, 2),
        "latency_ms_per_1k": round(lat_14_hard, 3)
    })
    print(f"  Acc: {configs[-1]['accuracy']}% | Macro-F1: {configs[-1]['macro_f1']}% | Ransomware: {configs[-1]['ransomware_recall']}%")

    # -----------------------------------------------------------------
    # Save Results CSV and JSON
    # -----------------------------------------------------------------
    df_ablation = pd.DataFrame(configs)
    csv_path = os.path.join(RESULTS_DIR, "feature_ablation_results.csv")
    df_ablation.to_csv(csv_path, index=False)

    summary_data = {
        "study": "Controlled 39-Feature vs. 14-Feature Ablation",
        "dataset": "NF-ToN-IoT-v2",
        "total_training_flows": len(X_train_full),
        "total_testing_flows": len(X_test_full),
        "configurations": configs,
        "delta_analysis": {
            "feature_reduction_under_standard": {
                "accuracy_delta": round(configs[1]["accuracy"] - configs[0]["accuracy"], 2),
                "macro_f1_delta": round(configs[1]["macro_f1"] - configs[0]["macro_f1"], 2),
                "feature_compression_pct": round((39 - 14) / 39 * 100, 1)
            },
            "feature_reduction_under_hardened": {
                "accuracy_delta": round(configs[3]["accuracy"] - configs[2]["accuracy"], 2),
                "macro_f1_delta": round(configs[3]["macro_f1"] - configs[2]["macro_f1"], 2),
                "weighted_f1_delta": round(configs[3]["weighted_f1"] - configs[2]["weighted_f1"], 2),
                "ransomware_recall_delta": round(configs[3]["ransomware_recall"] - configs[2]["ransomware_recall"], 2),
                "backdoor_recall_delta": round(configs[3]["backdoor_recall"] - configs[2]["backdoor_recall"], 2),
                "mitm_recall_delta": round(configs[3]["mitm_recall"] - configs[2]["mitm_recall"], 2),
                "latency_reduction_pct": round((configs[2]["latency_ms_per_1k"] - configs[3]["latency_ms_per_1k"]) / configs[2]["latency_ms_per_1k"] * 100, 1) if configs[2]["latency_ms_per_1k"] > 0 else 0
            },
            "defense_gain_on_14_features": {
                "accuracy_gain": round(configs[3]["accuracy"] - configs[1]["accuracy"], 2),
                "macro_f1_gain": round(configs[3]["macro_f1"] - configs[1]["macro_f1"], 2),
                "backdoor_recall_gain": round(configs[3]["backdoor_recall"] - configs[1]["backdoor_recall"], 2),
                "mitm_recall_gain": round(configs[3]["mitm_recall"] - configs[1]["mitm_recall"], 2)
            }
        }
    }
    json_path = os.path.join(RESULTS_DIR, "feature_ablation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 85)
    print("CONTROLLED FEATURE ABLATION SUMMARY TABLE")
    print("=" * 85)
    print(df_ablation[["configuration", "feature_count", "accuracy", "macro_f1", "weighted_f1", "ransomware_recall", "backdoor_recall", "mitm_recall"]].to_string(index=False))
    print(f"\nSaved CSV to:  {csv_path}")
    print(f"Saved JSON to: {json_path}")

    # -----------------------------------------------------------------
    # Generate Publication Figure
    # -----------------------------------------------------------------
    plot_ablation_figure(df_ablation)

    return summary_data


def plot_ablation_figure(df):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Subplot 1: Macro-F1 and Overall Accuracy across all 4 configurations
    x = np.arange(len(df))
    width = 0.35

    ax1 = axes[0]
    rects1 = ax1.bar(x - width/2, df["accuracy"], width, label="Accuracy (%)", color="#1f77b4", alpha=0.9)
    rects2 = ax1.bar(x + width/2, df["macro_f1"], width, label="Macro-F1 (%)", color="#ff7f0e", alpha=0.9)

    ax1.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Overall Accuracy & Macro-F1 Across Configurations\n(Controlled Feature Count vs. Defense Mechanism)",
                  fontsize=12, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    labels = [
        "Config A\n39-Feat\n(Standard)",
        "Config B\n14-Feat\n(Standard)",
        "Config C\n39-Feat\n(Hardened)",
        "Config D (Ours)\n14-Feat\n(Hardened)"
    ]
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.legend(loc="lower right", frameon=True, fontsize=10)
    ax1.set_ylim(50, 105)

    for bar in rects1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in rects2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Subplot 2: Rare Attack Recall Retention
    ax2 = axes[1]
    w_sub = 0.25
    x_sub = np.arange(len(df))
    r_ransom = ax2.bar(x_sub - w_sub, df["ransomware_recall"], w_sub, label="Ransomware Recall", color="#d62728", alpha=0.85)
    r_backdoor = ax2.bar(x_sub, df["backdoor_recall"], w_sub, label="Backdoor Recall", color="#2ca02c", alpha=0.85)
    r_mitm = ax2.bar(x_sub + w_sub, df["mitm_recall"], w_sub, label="MITM Recall", color="#9467bd", alpha=0.85)

    ax2.set_ylabel("Detection Recall (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Rare Attack Recall: 39 Features vs. 14 Features\n(Isolating Minority Detection Trade-offs)",
                  fontsize=12, fontweight="bold", pad=10)
    ax2.set_xticks(x_sub)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax2.set_ylim(0, 115)

    for bar in r_ransom:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar in r_backdoor:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar in r_mitm:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "feature_ablation_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved publication figure to: {fig_path}")


if __name__ == "__main__":
    run_ablation()

