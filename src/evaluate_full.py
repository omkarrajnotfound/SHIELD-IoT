"""
Phase D: Comprehensive Evaluation Suite
Generates exhaustive experimental evaluation metrics, figures, and tables for IEEE paper:
  - Multi-Class One-vs-Rest AUC-ROC curves & Macro-AUC score
  - False Positive Rate (FPR), Detection Rate (DR), and Specificity analysis
  - 10x10 Multi-Class Confusion Matrix Heatmap (Row-normalized percentage & counts)
  - Complete Per-Class Metrics Table (TP, FP, FN, TN, Precision, Recall, F1, FPR, Specificity, AUC)

Outputs:
  - figures/auc_roc_curves.png
  - figures/confusion_matrix_heatmap.png
  - results/per_class_full_metrics.csv
  - results/evaluation_full_summary.json
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

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.preprocessing import label_binarize

# Directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def run_evaluation_suite():
    print("=" * 80)
    print("PHASE D: COMPREHENSIVE EXPERIMENTAL EVALUATION SUITE")
    print("=" * 80)

    # 1. Load Model, Test Data & Metadata
    model_path = os.path.join(MODELS_DIR, "lightgbm_smote_balanced_top14.joblib")
    data_path = os.path.join(PROCESSED_DIR, "train_test_data.pkl")
    target_path = os.path.join(PROCESSED_DIR, "train_test_multi_targets.pkl")
    encoder_path = os.path.join(PROCESSED_DIR, "label_encoder_multi.pkl")
    features_path = os.path.join(PROCESSED_DIR, "selected_features.json")

    print("[Step 1] Loading model artifact, test flows (160,304), and selected 14 features...")
    model = joblib.load(model_path)
    _, X_test_df, _, _ = joblib.load(data_path)
    _, y_test = joblib.load(target_path)
    label_encoder = joblib.load(encoder_path)
    classes = list(label_encoder.classes_)

    with open(features_path, "r", encoding="utf-8") as f:
        selected_features = json.load(f)["final_features"]

    X_test_14 = X_test_df[selected_features]
    print(f"  Test partition: {len(X_test_14):,} flows x {len(selected_features)} features")
    print(f"  Target Classes ({len(classes)}): {classes}")

    # 2. Model Inference & Probability Estimates
    print("\n[Step 2] Computing predictions and posterior probabilities on test set...")
    t0_inf = time.time()
    y_pred = model.predict(X_test_14)
    y_probs = model.predict_proba(X_test_14)
    infer_time = time.time() - t0_inf
    latency_per_1k = (infer_time / len(X_test_14)) * 1000 * 1000

    acc = accuracy_score(y_test, y_pred) * 100
    macro_f1 = f1_score(y_test, y_pred, average="macro") * 100
    weighted_f1 = f1_score(y_test, y_pred, average="weighted") * 100
    print(f"  Test Accuracy:     {acc:.2f}%")
    print(f"  Macro-F1 Score:    {macro_f1:.2f}%")
    print(f"  Inference Latency: {latency_per_1k:.3f} ms / 1,000 flows")

    # 3. Step D1: Multi-Class AUC-ROC Analysis
    print("\n[Step D1] Computing Multi-Class One-vs-Rest ROC curves and AUC scores...")
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    n_classes = len(classes)

    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}

    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

    # Compute micro and macro average ROC
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(y_test_bin.ravel(), y_probs.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    # Macro-average AUC
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    print(f"  Macro-Average ROC-AUC: {roc_auc_dict['macro']:.4f}")
    print(f"  Micro-Average ROC-AUC: {roc_auc_dict['micro']:.4f}")
    for i, c in enumerate(classes):
        print(f"    - {c:<12}: AUC = {roc_auc_dict[i]:.4f}")

    # Plot AUC-ROC Curves
    auc_fig_path = os.path.join(FIGURES_DIR, "auc_roc_curves.png")
    plot_roc_curves(fpr_dict, tpr_dict, roc_auc_dict, classes, auc_fig_path)

    # 4. Step D2: False Positive Rate (FPR) & Detection Rate (DR) Analysis
    print("\n[Step D2] Analyzing False Positive Rate (FPR) & Attack Detection Rate (DR)...")
    is_benign_true = (y_test == 0)
    is_attack_true = (y_test > 0)
    is_benign_pred = (y_pred == 0)
    is_attack_pred = (y_pred > 0)

    fp_benign = int((is_benign_true & is_attack_pred).sum())
    tn_benign = int((is_benign_true & is_benign_pred).sum())
    tp_attack = int((is_attack_true & is_attack_pred).sum())
    fn_attack = int((is_attack_true & is_benign_pred).sum())

    benign_fpr = (fp_benign / (fp_benign + tn_benign)) * 100
    attack_dr = (tp_attack / (tp_attack + fn_attack)) * 100
    attack_fnr = (fn_attack / (tp_attack + fn_attack)) * 100

    print(f"  True Benign Flows:                  {is_benign_true.sum():,}")
    print(f"  Benign False Alarms (Normal->Attack): {fp_benign:,} flows")
    print(f"  False Positive Rate (FPR):          {benign_fpr:.2f}% (Target: < 2.0%)")
    print(f"  True Attack Flows:                  {is_attack_true.sum():,}")
    print(f"  Attacks Correctly Intercepted:      {tp_attack:,} flows")
    print(f"  Attack Detection Rate (DR):         {attack_dr:.2f}%")
    print(f"  Attack False Negative Rate (FNR):   {attack_fnr:.2f}%")

    # 5. Step D3: Multi-Class 10x10 Confusion Matrix Heatmap
    print("\n[Step D3] Generating 10x10 Multi-Class Confusion Matrix Heatmap...")
    cm_raw = confusion_matrix(y_test, y_pred)
    cm_norm = cm_raw.astype("float") / cm_raw.sum(axis=1)[:, np.newaxis] * 100

    cm_fig_path = os.path.join(FIGURES_DIR, "confusion_matrix_heatmap.png")
    plot_confusion_matrix_heatmap(cm_norm, cm_raw, classes, cm_fig_path)

    # 6. Step D4: Detailed Per-Class Evaluation Metrics Table
    print("\n[Step D4] Compiling Detailed Per-Class Metrics Table...")
    per_class_data = []

    for i, c in enumerate(classes):
        # One-vs-Rest metrics for class i
        tp = int(cm_raw[i, i])
        fn = int(cm_raw[i, :].sum() - tp)
        fp = int(cm_raw[:, i].sum() - tp)
        tn = int(cm_raw.sum() - (tp + fp + fn))

        rec = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        prec = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        spec = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0
        fpr = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0

        per_class_data.append({
            "attack_class": c,
            "test_samples": int(cm_raw[i, :].sum()),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision_pct": round(prec, 2),
            "recall_detection_rate_pct": round(rec, 2),
            "f1_score_pct": round(f1, 2),
            "specificity_pct": round(spec, 2),
            "false_positive_rate_pct": round(fpr, 2),
            "auc_roc": round(roc_auc_dict[i], 4)
        })

    df_full_metrics = pd.DataFrame(per_class_data)
    csv_metrics_path = os.path.join(RESULTS_DIR, "per_class_full_metrics.csv")
    df_full_metrics.to_csv(csv_metrics_path, index=False)
    print(f"Saved full per-class metrics to: {csv_metrics_path}")

    # Display clean table
    print("\n" + "=" * 115)
    print("DETAILED PER-CLASS EVALUATION METRICS (PROPOSED STEALTH-AWARE LIGHTGBM)")
    print("=" * 115)
    disp_cols = ["attack_class", "test_samples", "precision_pct", "recall_detection_rate_pct", "f1_score_pct", "false_positive_rate_pct", "auc_roc"]
    print(df_full_metrics[disp_cols].to_string(index=False))
    print("=" * 115)

    # 7. Save Comprehensive JSON Summary
    summary_dict = {
        "evaluation_phase": "Phase D: Full Evaluation Suite",
        "dataset": "NF-ToN-IoT-v2 (160,304 Test Flows)",
        "features_count": 14,
        "selected_features": selected_features,
        "overall_performance": {
            "accuracy_pct": round(acc, 2),
            "macro_f1_pct": round(macro_f1, 2),
            "weighted_f1_pct": round(weighted_f1, 2),
            "macro_auc_roc": round(roc_auc_dict["macro"], 4),
            "micro_auc_roc": round(roc_auc_dict["micro"], 4),
            "inference_latency_ms_per_1k": round(latency_per_1k, 3)
        },
        "false_alarm_analysis": {
            "benign_total_flows": int(is_benign_true.sum()),
            "benign_false_alarms": fp_benign,
            "benign_false_positive_rate_pct": round(benign_fpr, 2),
            "attack_total_flows": int(is_attack_true.sum()),
            "attack_detection_rate_pct": round(attack_dr, 2),
            "attack_false_negative_rate_pct": round(attack_fnr, 2)
        },
        "stealth_attack_detection": {
            "ransomware": {
                "samples": int(cm_raw[classes.index("ransomware"), :].sum()),
                "recall_pct": round(df_full_metrics[df_full_metrics["attack_class"]=="ransomware"]["recall_detection_rate_pct"].values[0], 2),
                "precision_pct": round(df_full_metrics[df_full_metrics["attack_class"]=="ransomware"]["precision_pct"].values[0], 2),
                "f1_pct": round(df_full_metrics[df_full_metrics["attack_class"]=="ransomware"]["f1_score_pct"].values[0], 2),
                "auc": round(roc_auc_dict[classes.index("ransomware")], 4)
            },
            "backdoor": {
                "samples": int(cm_raw[classes.index("backdoor"), :].sum()),
                "recall_pct": round(df_full_metrics[df_full_metrics["attack_class"]=="backdoor"]["recall_detection_rate_pct"].values[0], 2),
                "precision_pct": round(df_full_metrics[df_full_metrics["attack_class"]=="backdoor"]["precision_pct"].values[0], 2),
                "f1_pct": round(df_full_metrics[df_full_metrics["attack_class"]=="backdoor"]["f1_score_pct"].values[0], 2),
                "auc": round(roc_auc_dict[classes.index("backdoor")], 4)
            },
            "mitm": {
                "samples": int(cm_raw[classes.index("mitm"), :].sum()),
                "recall_pct": round(df_full_metrics[df_full_metrics["attack_class"]=="mitm"]["recall_detection_rate_pct"].values[0], 2),
                "f1_pct": round(df_full_metrics[df_full_metrics["attack_class"]=="mitm"]["f1_score_pct"].values[0], 2),
                "auc": round(roc_auc_dict[classes.index("mitm")], 4)
            }
        }
    }

    summary_json_path = os.path.join(RESULTS_DIR, "evaluation_full_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Saved evaluation summary JSON to: {summary_json_path}")

    return df_full_metrics, summary_dict


def plot_roc_curves(fpr_dict, tpr_dict, roc_auc_dict, classes, save_path):
    plt.figure(figsize=(9, 7), dpi=300)

    # Macro & Micro average lines
    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label=f"Macro-Average ROC (AUC = {roc_auc_dict['macro']:.4f})",
             color="navy", linestyle=":", linewidth=2.5)

    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label=f"Micro-Average ROC (AUC = {roc_auc_dict['micro']:.4f})",
             color="deeppink", linestyle="--", linewidth=2.0)

    # Color palette for 10 classes
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    for i, (c, color) in enumerate(zip(classes, colors)):
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=1.5,
                 label=f"{c} (AUC = {roc_auc_dict[i]:.4f})")

    # Baseline diagonal
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)

    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold", fontsize=11)
    plt.title("Multi-Class One-vs-Rest ROC Curves: Proposed Stealth-Aware LightGBM\n(NF-ToN-IoT-v2, 14 Critical Features)", fontweight="bold", fontsize=12)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.9, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve figure saved to: {save_path}")


def plot_confusion_matrix_heatmap(cm_norm, cm_raw, classes, save_path):
    plt.figure(figsize=(10, 8), dpi=300)

    # Format annotations showing both row-percentage and sample count
    annot_matrix = np.empty_like(cm_norm, dtype=object)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val_pct = cm_norm[i, j]
            val_cnt = cm_raw[i, j]
            if val_pct >= 1.0:
                annot_matrix[i, j] = f"{val_pct:.1f}%\n({val_cnt:,})"
            elif val_cnt > 0:
                annot_matrix[i, j] = f"<1%\n({val_cnt})"
            else:
                annot_matrix[i, j] = "0"

    sns.heatmap(
        cm_norm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Normalized True Class Recall Rate (%)"},
        linewidths=0.5,
        linecolor="#e0e0e0"
    )

    plt.xlabel("Predicted Class Label", fontweight="bold", fontsize=11)
    plt.ylabel("Actual True Class Label", fontweight="bold", fontsize=11)
    plt.title("Multi-Class Confusion Matrix Heatmap (Row-Normalized Detection Rates)\nProposed Stealth-Aware LightGBM (14 Features)", fontweight="bold", fontsize=12)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix heatmap saved to: {save_path}")


if __name__ == "__main__":
    run_evaluation_suite()

