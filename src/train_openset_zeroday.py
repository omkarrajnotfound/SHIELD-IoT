"""
Phase G: Open-Set Zero-Day Discovery Engine with Energy-Based Epistemic Uncertainty Estimation
Strict Class-Held-Out Implementation (Zero Leakage Across Upstream Feature Selection & Scaling)

Key Integrity Guarantees:
  1. Strict Zero-Leakage Holdout Protocol:
     - Withholds Ransomware and Backdoor completely from EVERY upstream step.
     - RobustScaler is fit strictly on the known-class fit partition.
     - Two-stage feature selection (ANOVA F-test 39->22 and TreeSHAP 22->14) is run
       strictly on the known-class fit partition with zero ransomware/backdoor exposure.
     - Trains exclusively on known classes (Benign, Scanning, DDoS, DoS, Password, Injection, XSS, MITM).
  2. Free Energy & Distance-Aware Epistemic Uncertainty Estimator:
     - E(x; T) = -T * log sum exp(f_k(x) / T)
     - S(x) = d_proto(x) - w * E_norm(x)
  3. Validation-Calibrated Decision Threshold:
     - tau_OOD is fixed strictly at the 95th percentile of known validation traffic.
  4. Generates:
     - models/lightgbm_openset_top14.joblib
     - models/openset_detector_params.joblib
     - results/zero_day_detection_metrics.csv
     - results/phase_g_summary.json
     - figures/zero_day_openset_auroc.png
     - figures/energy_score_distribution.png
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def run_phase_g():
    print("=" * 80)
    print("PHASE G: OPEN-SET ZERO-DAY RECOGNITION (OSR) - STRICT CLASS-HELD-OUT PROTOCOL")
    print("=" * 80)

    # 1. Load Raw Unscaled Data & Multi-Class Targets
    raw_path = os.path.join(PROCESSED_DIR, "raw_features_and_targets.pkl")
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found. Please create raw features first.")
        sys.exit(1)

    print("[Step 1] Loading raw unscaled benchmark dataset...")
    X_raw, y_binary, y_multi = joblib.load(raw_path)
    print(f"  Total Benchmark Records: {len(X_raw):,} network flows")
    print(f"  Total Raw Features:      {X_raw.shape[1]}")

    global_le = LabelEncoder()
    global_le.fit(y_multi)
    classes = list(global_le.classes_)
    print(f"  Full Class Inventory ({len(classes)}): {classes}")

    # 80/20 Stratified Train-Test Split
    X_train_raw, X_test_raw, y_tr_bin, y_te_bin, y_tr_multi, y_te_multi = train_test_split(
        X_raw, y_binary, y_multi, test_size=0.20, random_state=42, stratify=y_multi
    )
    print(f"  Training Split (Raw): {len(X_train_raw):,} flows")
    print(f"  Testing Split (Raw):  {len(X_test_raw):,} flows")

    # 2. Strict Zero-Day Holdout Protocol: Quarantine Ransomware & Backdoor Upfront
    zero_day_names = ["ransomware", "backdoor"]
    print(f"\n[Step 2] Executing Strict Zero-Day Quarantine Upfront...")
    print(f"  Held-Out Zero-Day Attack Classes: {zero_day_names}")

    train_mask_known = ~y_tr_multi.isin(zero_day_names)
    X_train_known_raw = X_train_raw[train_mask_known]
    y_train_known_multi = y_tr_multi[train_mask_known]
    y_train_known_bin = y_tr_bin[train_mask_known]

    train_zd_count = int(np.sum(~train_mask_known))
    # CRITICAL LEAKAGE ASSERTION: Zero-day classes must be strictly excluded
    assert not any(y_train_known_multi.isin(zero_day_names)), "CRITICAL ERROR: Held-out attack flows detected in training partition!"
    print(f"  Quarantined {train_zd_count:,} zero-day flows from training split (Zero Leakage Guaranteed).")
    print(f"  Remaining Known Training Flows: {len(X_train_known_raw):,}")

    # Encode known classes
    known_le = LabelEncoder()
    y_train_known_encoded = known_le.fit_transform(y_train_known_multi)
    known_classes = list(known_le.classes_)
    print(f"  Known Classes Trained ({len(known_classes)}): {known_classes}")

    # 3-Way Partitioning: 80% Fit / 20% Validation Strictly on Known Training Flows
    print("\n[Step 2b] Creating 3-Way Split: 80% Fit / 20% Validation on Known Data...")
    X_fit_raw, X_val_raw, y_fit_multi, y_val_multi, y_fit_bin, y_val_bin, y_fit_encoded, y_val_encoded = train_test_split(
        X_train_known_raw, y_train_known_multi, y_train_known_bin, y_train_known_encoded,
        test_size=0.20, random_state=42, stratify=y_train_known_multi
    )
    print(f"  Known Fit Partition:        {len(X_fit_raw):,} flows")
    print(f"  Known Validation Partition: {len(X_val_raw):,} flows (for blind tau_OOD calibration)")

    # 3. Fit RobustScaler Strictly on Known Fit Partition
    print("\n[Step 3] Fitting RobustScaler Strictly on Known Fit Partition (Zero Preprocessing Leakage)...")
    CLIP_BOUND = 10000.0

    scaler = RobustScaler()
    X_fit_s = scaler.fit_transform(X_fit_raw)
    X_val_s = scaler.transform(X_val_raw)
    X_test_s = scaler.transform(X_test_raw)

    X_fit_s = np.nan_to_num(X_fit_s, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_s = np.nan_to_num(X_val_s, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)

    X_fit_scaled = pd.DataFrame(np.clip(X_fit_s, -CLIP_BOUND, CLIP_BOUND).astype(np.float32), columns=X_raw.columns, index=X_fit_raw.index)
    X_val_scaled = pd.DataFrame(np.clip(X_val_s, -CLIP_BOUND, CLIP_BOUND).astype(np.float32), columns=X_raw.columns, index=X_val_raw.index)
    X_test_scaled = pd.DataFrame(np.clip(X_test_s, -CLIP_BOUND, CLIP_BOUND).astype(np.float32), columns=X_raw.columns, index=X_test_raw.index)

    # 4. Rerun Two-Stage Feature Selection Strictly on Known Fit Partition
    print("\n[Step 4] Rerunning Two-Stage Feature Selection Strictly on Known Fit Data...")
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(X_fit_scaled), min(50000, len(X_fit_scaled)), replace=False)
    X_sub = X_fit_scaled.iloc[sub_idx]
    y_sub_bin = y_fit_bin.iloc[sub_idx]

    # Stage 1: ANOVA F-test on Known Fit (39 -> 22)
    t0_fs = time.time()
    f_vals, _ = f_classif(np.clip(X_sub.values, -1e4, 1e4), y_sub_bin.values)
    f_vals = np.nan_to_num(f_vals, nan=0.0)
    s1_features = X_raw.columns[np.argsort(-f_vals)[:22]].tolist()
    print(f"  Stage 1 (ANOVA F-test) selected 22 candidate features in {time.time()-t0_fs:.2f}s")

    # Stage 2: TreeSHAP Attribution on Known Fit (22 -> 14)
    t0_shap = time.time()
    clf_fs = LGBMClassifier(
        n_estimators=60, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1
    )
    clf_fs.fit(X_sub[s1_features], y_sub_bin)
    shap_contribs = clf_fs.booster_.predict(X_sub[s1_features].iloc[:5000], pred_contrib=True)[:, :-1]
    mean_abs_shap = np.mean(np.abs(shap_contribs), axis=0)
    selected_features = [s1_features[i] for i in np.argsort(-mean_abs_shap)[:14]]
    print(f"  Stage 2 (TreeSHAP) selected 14 final features in {time.time()-t0_shap:.2f}s:")
    for idx_f, f_name in enumerate(selected_features, start=1):
        print(f"    {idx_f:2d}. {f_name}")

    # 5. Train LightGBM OSR Classifier on Known Fit Partition
    print("\n[Step 5] Training LightGBM on Known Fit Partition (14 Selected Features)...")
    t0_train = time.time()
    lgb_model = LGBMClassifier(
        n_estimators=100,
        max_depth=8,
        min_child_samples=100,
        learning_rate=0.08,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_fit_scaled[selected_features], y_fit_encoded)
    train_duration = time.time() - t0_train
    print(f"  Model training completed in {train_duration:.2f} seconds.")

    # 6. Calculate Epistemic Uncertainty & Prototype Centroids Strictly on Known Fit Partition
    print("\n[Step 6] Calculating Class Prototypes and Free Energy Statistics on Known Fit Data...")
    X_fit_14 = X_fit_scaled[selected_features].values
    known_centroids = np.array([
        X_fit_14[y_fit_encoded == k].mean(axis=0)
        for k in range(len(known_le.classes_))
    ])
    feat_variance = np.var(X_fit_14, axis=0) + 1e-6

    # Subsample fit data for baseline normalization parameters
    calib_idx = rng.choice(len(X_fit_14), min(50000, len(X_fit_14)), replace=False)
    X_train_calib = X_fit_14[calib_idx]

    diffs_tr = X_train_calib[:, None, :] - known_centroids[None, :, :]
    norm_diffs_tr = diffs_tr / np.sqrt(feat_variance)[None, None, :]
    dists_proto_tr = np.min(np.sqrt(np.sum(norm_diffs_tr ** 2, axis=2)), axis=1)
    dist_mean = float(np.mean(dists_proto_tr))
    dist_std = float(np.std(dists_proto_tr) + 1e-8)

    T = 10.0
    raw_logits_tr = lgb_model.predict(X_train_calib, raw_score=True)
    raw_max_tr = np.max(raw_logits_tr, axis=1, keepdims=True)
    free_energy_tr = -T * np.log(np.sum(np.exp((raw_logits_tr - raw_max_tr) / T), axis=1) + 1e-12) - np.max(raw_logits_tr, axis=1)
    energy_mean = float(np.mean(free_energy_tr))
    energy_std = float(np.std(free_energy_tr) + 1e-8)

    print(f"  Scaling parameters (Fit partition):")
    print(f"    - Prototype Distance: mean = {dist_mean:.4f}, std = {dist_std:.4f}")
    print(f"    - Free Energy:        mean = {energy_mean:.4f}, std = {energy_std:.4f}")

    # 7. Blind Validation Calibration of tau_OOD & tau_MSP
    print("\n[Step 7] Calibrating Decision Thresholds on Known Validation Partition (q_0.95)...")
    X_val_14 = X_val_scaled[selected_features].values
    diffs_val = X_val_14[:, None, :] - known_centroids[None, :, :]
    norm_diffs_val = diffs_val / np.sqrt(feat_variance)[None, None, :]
    dists_proto_val = np.min(np.sqrt(np.sum(norm_diffs_val ** 2, axis=2)), axis=1)

    raw_logits_val = lgb_model.predict(X_val_14, raw_score=True)
    raw_max_val = np.max(raw_logits_val, axis=1, keepdims=True)
    free_energy_val = -T * np.log(np.sum(np.exp((raw_logits_val - raw_max_val) / T), axis=1) + 1e-12) - np.max(raw_logits_val, axis=1)

    dists_norm_val = (dists_proto_val - dist_mean) / dist_std
    energy_norm_val = (free_energy_val - energy_mean) / energy_std
    w_energy = 0.25
    val_epistemic_scores = dists_norm_val - w_energy * energy_norm_val
    tau_ood = float(np.percentile(val_epistemic_scores, 95.0))

    # Baseline Softmax MSP calibration
    val_probs = lgb_model.predict_proba(X_val_14)
    val_msp_scores = 1.0 - np.max(val_probs, axis=1)
    tau_msp = float(np.percentile(val_msp_scores, 95.0))

    print(f"  Calibrated tau_OOD (q_0.95): {tau_ood:.4f}")
    print(f"  Calibrated tau_MSP (q_0.95): {tau_msp:.4f}")

    # 8. Evaluation on Untouched Holdout Test Partition
    print("\n[Step 8] Evaluating on Untouched Holdout Test Partition (160,304 Flows)...")
    X_test_14 = X_test_scaled[selected_features].values
    test_is_zd = y_te_multi.isin(zero_day_names).values.astype(int)
    num_test_zd = int(np.sum(test_is_zd))
    num_test_known = int(len(y_te_multi) - num_test_zd)

    print(f"  Test partition composition: {num_test_known:,} Known Flows vs {num_test_zd:,} Novel Zero-Day Flows")
    for zd in zero_day_names:
        cnt_zd = int(np.sum(y_te_multi == zd))
        print(f"    - Unseen {zd}: {cnt_zd} flows")

    # Epistemic Uncertainty on Test Data
    diffs_te = X_test_14[:, None, :] - known_centroids[None, :, :]
    norm_diffs_te = diffs_te / np.sqrt(feat_variance)[None, None, :]
    dists_proto_te = np.min(np.sqrt(np.sum(norm_diffs_te ** 2, axis=2)), axis=1)

    raw_logits_te = lgb_model.predict(X_test_14, raw_score=True)
    raw_max_te = np.max(raw_logits_te, axis=1, keepdims=True)
    free_energy_te = -T * np.log(np.sum(np.exp((raw_logits_te - raw_max_te) / T), axis=1) + 1e-12) - np.max(raw_logits_te, axis=1)

    dists_norm_te = (dists_proto_te - dist_mean) / dist_std
    energy_norm_te = (free_energy_te - energy_mean) / energy_std
    test_epistemic_scores = dists_norm_te - w_energy * energy_norm_te

    # Softmax MSP on Test Data
    test_probs = lgb_model.predict_proba(X_test_14)
    test_msp_scores = 1.0 - np.max(test_probs, axis=1)

    # AUROC
    osr_auroc = float(roc_auc_score(test_is_zd, test_epistemic_scores))
    baseline_msp_auroc = float(roc_auc_score(test_is_zd, test_msp_scores))
    print(f"  Proposed Open-Set AUROC: {osr_auroc * 100:.2f}%")
    print(f"  Closed-World MSP AUROC:  {baseline_msp_auroc * 100:.2f}%")

    fpr_curve, tpr_curve, _ = roc_curve(test_is_zd, test_epistemic_scores)

    # Operational Decision Flags
    osr_flags = (test_epistemic_scores >= tau_ood).astype(int)
    msp_flags = (test_msp_scores >= tau_msp).astype(int)

    mask_known = (test_is_zd == 0)
    fpr_test = float(np.mean(osr_flags[mask_known]))
    msp_fpr_test = float(np.mean(msp_flags[mask_known]))

    overall_zd_recall = float(np.mean(osr_flags[test_is_zd == 1]))
    msp_overall_zd_recall = float(np.mean(msp_flags[test_is_zd == 1]))

    # Per-Class Detection Rates
    per_class_zd_results = {}
    for zd_name in zero_day_names:
        mask = (y_te_multi == zd_name).values
        det_cnt = int(np.sum(osr_flags[mask]))
        tot_cnt = int(np.sum(mask))
        det_rate = det_cnt / tot_cnt if tot_cnt > 0 else 0.0
        per_class_zd_results[zd_name] = {
            "total_samples": tot_cnt,
            "detected_samples": det_cnt,
            "detection_rate": round(det_rate * 100, 2)
        }
        print(f"  Unseen {zd_name.capitalize()} Detection Rate: {det_rate * 100:.2f}% ({det_cnt}/{tot_cnt})")

    # Closed-World Blindness demonstration
    closed_world_preds_encoded = lgb_model.predict(X_test_14)
    baseline_misclass = {}
    for zd_name in zero_day_names:
        mask = (y_te_multi == zd_name).values
        preds_for_zd = known_le.inverse_transform(closed_world_preds_encoded[mask])
        unique_p, counts_p = np.unique(preds_for_zd, return_counts=True)
        top_misclass = sorted(zip(unique_p, counts_p), key=lambda x: -x[1])
        baseline_misclass[zd_name] = {cls: int(cnt) for cls, cnt in top_misclass}

    # Known traffic decomposed metrics
    known_preds_raw = lgb_model.predict(X_test_14[mask_known])
    known_true_encoded = known_le.transform(y_te_multi[mask_known])

    raw_classifier_acc = float(accuracy_score(known_true_encoded, known_preds_raw))
    mask_accepted = (osr_flags[mask_known] == 0)
    known_coverage = float(np.mean(mask_accepted))
    acc_accepted_known = float(accuracy_score(known_true_encoded[mask_accepted], known_preds_raw[mask_accepted]))
    ccr_known = float(np.mean(mask_accepted & (known_preds_raw == known_true_encoded)))
    total_correct_osr = int(np.sum(mask_accepted & (known_preds_raw == known_true_encoded)) + np.sum(osr_flags[test_is_zd == 1] == 1))
    overall_osr_acc = float(total_correct_osr / len(y_te_multi))

    # MSP baseline operational metrics
    msp_mask_accepted = (msp_flags[mask_known] == 0)
    msp_coverage = float(np.mean(msp_mask_accepted))
    msp_acc_accepted = float(accuracy_score(known_true_encoded[msp_mask_accepted], known_preds_raw[msp_mask_accepted]))
    msp_ccr_known = float(np.mean(msp_mask_accepted & (known_preds_raw == known_true_encoded)))
    msp_ransom_recall = float(np.mean(msp_flags[(y_te_multi == zero_day_names[0]).values]))
    msp_backdoor_recall = float(np.mean(msp_flags[(y_te_multi == zero_day_names[1]).values]))
    msp_total_correct = int(np.sum(msp_mask_accepted & (known_preds_raw == known_true_encoded)) + np.sum(msp_flags[test_is_zd == 1] == 1))
    msp_overall_decision_acc = float(msp_total_correct / len(y_te_multi))

    print(f"  Underlying Classifier Accuracy (Pre-Rejection):   {raw_classifier_acc * 100:.2f}%")
    print(f"  Known Flow Acceptance Rate (Coverage):           {known_coverage * 100:.2f}%")
    print(f"  Classification Accuracy on Accepted Known Flows: {acc_accepted_known * 100:.2f}%")
    print(f"  Known-Traffic Joint CCR (Accepted & Correct):    {ccr_known * 100:.2f}%")
    print(f"  Overall Open-Set Decision Accuracy:              {overall_osr_acc * 100:.2f}%")

    # 9. Save CSV Table
    metrics_table = pd.DataFrame([
        {
            "Metric": "Zero-Day Detection AUROC",
            "Closed-World Baseline": f"{baseline_msp_auroc * 100:.2f}%",
            "Proposed Energy OSR Detector": f"{osr_auroc * 100:.2f}%",
            "Target Threshold": "> 90.0%",
            "Status": "MET" if osr_auroc >= 0.90 else "UNMET"
        },
        {
            "Metric": "Out-of-Sample FPR (Known Test Flows)",
            "Closed-World Baseline": f"{msp_fpr_test * 100:.2f}%",
            "Proposed Energy OSR Detector": f"{fpr_test * 100:.2f}%",
            "Target Threshold": "< 5.5%",
            "Status": "MET" if fpr_test <= 0.055 else "UNMET"
        },
        {
            "Metric": "Known Flow Acceptance Rate (Coverage)",
            "Closed-World Baseline": f"{msp_coverage * 100:.2f}%",
            "Proposed Energy OSR Detector": f"{known_coverage * 100:.2f}%",
            "Target Threshold": "> 94.5%",
            "Status": "MET" if known_coverage >= 0.945 else "UNMET"
        },
        {
            "Metric": "Underlying Classifier Accuracy (Pre-Rejection)",
            "Closed-World Baseline": f"{raw_classifier_acc * 100:.2f}%",
            "Proposed Energy OSR Detector": f"{raw_classifier_acc * 100:.2f}%",
            "Target Threshold": "> 96.0%",
            "Status": "MET" if raw_classifier_acc >= 0.96 else "UNMET"
        },
        {
            "Metric": "Accuracy on Accepted Known Flows",
            "Closed-World Baseline": f"{msp_acc_accepted * 100:.2f}%",
            "Proposed Energy OSR Detector": f"{acc_accepted_known * 100:.2f}%",
            "Target Threshold": "> 96.0%",
            "Status": "MET" if acc_accepted_known >= 0.96 else "UNMET"
        },
        {
            "Metric": "Known-Traffic Joint CCR (Accepted & Correct)",
            "Closed-World Baseline": f"{msp_ccr_known * 100:.2f}%",
            "Proposed Energy OSR Detector": f"{ccr_known * 100:.2f}%",
            "Target Threshold": "> 90.0%",
            "Status": "MET" if ccr_known >= 0.90 else "UNMET"
        },
        {
            "Metric": "Unseen Ransomware Detection Rate",
            "Closed-World Baseline": f"{msp_ransom_recall * 100:.2f}% (0.0% raw)",
            "Proposed Energy OSR Detector": f"{per_class_zd_results['ransomware']['detection_rate']:.2f}%",
            "Target Threshold": "> 75.0%",
            "Status": "MET" if per_class_zd_results['ransomware']['detection_rate'] >= 75.0 else "UNMET"
        },
        {
            "Metric": "Unseen Backdoor Detection Rate",
            "Closed-World Baseline": f"{msp_backdoor_recall * 100:.2f}% (0.0% raw)",
            "Proposed Energy OSR Detector": f"{per_class_zd_results['backdoor']['detection_rate']:.2f}%",
            "Target Threshold": "> 60.0%",
            "Status": "MET" if per_class_zd_results['backdoor']['detection_rate'] >= 60.0 else "UNMET"
        },
        {
            "Metric": "Overall Zero-Day Detection Recall",
            "Closed-World Baseline": f"{msp_overall_zd_recall * 100:.2f}% (0.0% raw)",
            "Proposed Energy OSR Detector": f"{overall_zd_recall * 100:.2f}%",
            "Target Threshold": "> 60.0%",
            "Status": "MET" if overall_zd_recall >= 0.60 else "UNMET"
        },
        {
            "Metric": "Overall Open-Set Decision Accuracy",
            "Closed-World Baseline": f"{msp_overall_decision_acc * 100:.2f}%",
            "Proposed Energy OSR Detector": f"{overall_osr_acc * 100:.2f}%",
            "Target Threshold": "> 90.0%",
            "Status": "MET" if overall_osr_acc >= 0.90 else "UNMET"
        },
        {
            "Metric": "Known Traffic Accuracy Retention",
            "Closed-World Baseline": f"{raw_classifier_acc * 100:.2f}%",
            "Proposed Energy OSR Detector": f"{acc_accepted_known * 100:.2f}%",
            "Target Threshold": "> 96.0%",
            "Status": "MET" if acc_accepted_known >= 0.96 else "UNMET"
        }
    ])

    metrics_csv_path = os.path.join(RESULTS_DIR, "zero_day_detection_metrics.csv")
    metrics_table.to_csv(metrics_csv_path, index=False)
    print(f"\nSaved metrics comparison table to: {metrics_csv_path}")

    # 10. Save Models and Parameters
    model_save_path = os.path.join(MODELS_DIR, "lightgbm_openset_top14.joblib")
    joblib.dump(lgb_model, model_save_path)

    detector_params = {
        "features": selected_features,
        "known_classes": known_classes,
        "known_label_encoder": known_le,
        "global_label_encoder": global_le,
        "zero_day_classes": zero_day_names,
        "scaler": scaler,
        "centroids": known_centroids,
        "feat_variance": feat_variance,
        "dist_mean": float(dist_mean),
        "dist_std": float(dist_std),
        "energy_mean": float(energy_mean),
        "energy_std": float(energy_std),
        "temperature_T": float(T),
        "weight_energy": float(w_energy),
        "tau_ood": float(tau_ood),
        "fpr_at_95": float(fpr_test),
        "tpr_at_95": float(overall_zd_recall),
        "osr_auroc": float(osr_auroc),
        "known_raw_accuracy": float(raw_classifier_acc),
        "known_coverage": float(known_coverage),
        "known_accepted_accuracy": float(acc_accepted_known),
        "known_ccr": float(ccr_known),
        "overall_open_set_acc": float(overall_osr_acc)
    }
    params_save_path = os.path.join(MODELS_DIR, "openset_detector_params.joblib")
    joblib.dump(detector_params, params_save_path)
    print(f"  Saved trained model to: {model_save_path}")
    print(f"  Saved OSR parameters to: {params_save_path}")

    # 11. Generate Publication Figures
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: AUROC
    fig1, ax1 = plt.subplots(figsize=(7, 6), dpi=300)
    fpr_msp, tpr_msp, _ = roc_curve(test_is_zd, test_msp_scores)

    ax1.plot(fpr_curve, tpr_curve, color="#0052cc", lw=2.5,
             label=f"Proposed Distance-Energy OSR (AUROC = {osr_auroc*100:.2f}%)")
    ax1.plot(fpr_msp, tpr_msp, color="#ff5630", lw=2.0, linestyle="--",
             label=f"Closed-World MSP Baseline (AUROC = {baseline_msp_auroc*100:.2f}%)")
    ax1.plot([0, 1], [0, 1], color="#999999", lw=1.2, linestyle=":", label="Random Guess (AUROC = 50.0%)")

    ax1.scatter([fpr_test], [overall_zd_recall], color="#ffab00", s=100, zorder=5, edgecolor="black",
                label=f"Calibrated Operating Point (Recall={overall_zd_recall*100:.1f}%, FPR={fpr_test*100:.2f}%)")
    ax1.axvline(x=fpr_test, color="#ffab00", linestyle=":", alpha=0.6)
    ax1.axhline(y=overall_zd_recall, color="#ffab00", linestyle=":", alpha=0.6)

    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel("False Positive Rate (FPR)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (TPR / Zero-Day Recall)", fontsize=12, fontweight="bold")
    ax1.set_title("Open-Set Zero-Day Intrusion Discovery AUROC\n(Strict Class-Held-Out Evaluation on Ransomware & Backdoor)",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="lower right", frameon=True, fontsize=10)
    fig1.tight_layout()

    fig1_path = os.path.join(FIGURES_DIR, "zero_day_openset_auroc.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"  Generated ROC curve: {fig1_path}")

    # Figure 2: Score distribution
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=300)
    known_scores = test_epistemic_scores[mask_known]
    zd_ransom_scores = test_epistemic_scores[(y_te_multi == zero_day_names[0]).values]
    zd_backdoor_scores = test_epistemic_scores[(y_te_multi == zero_day_names[1]).values]

    bins = np.linspace(-3, 6, 90)
    ax2.hist(known_scores, bins=bins, density=True, alpha=0.55, color="#0052cc", label=f"Known Network Flows (N={len(known_scores):,})")
    ax2.hist(zd_ransom_scores, bins=bins, density=True, alpha=0.75, color="#de350b", label=f"Unseen Ransomware (N={len(zd_ransom_scores)})")
    ax2.hist(zd_backdoor_scores, bins=bins, density=True, alpha=0.75, color="#ff8b00", label=f"Unseen Backdoor (N={len(zd_backdoor_scores)})")

    ax2.axvline(x=tau_ood, color="#36b37e", lw=2.5, linestyle="--",
                label=f"Decision Threshold tau_OOD = {tau_ood:.2f} (Val Calibrated)")

    ax2.set_xlabel("Epistemic Uncertainty Score S(x)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Probability Density", fontsize=12, fontweight="bold")
    ax2.set_title("Epistemic Uncertainty Distribution Separation\n(Known In-Distribution vs. Novel Zero-Day Exploits)",
                  fontsize=13, fontweight="bold", pad=12)
    ax2.legend(loc="upper right", frameon=True, fontsize=10)
    fig2.tight_layout()

    fig2_path = os.path.join(FIGURES_DIR, "energy_score_distribution.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"  Generated distribution density plot: {fig2_path}")

    # 12. Save JSON Summary
    summary_data = {
        "phase": "Phase G - Open-Set Zero-Day Discovery Engine (Strict Class-Held-Out Protocol)",
        "dataset": "NF-ToN-IoT-v2",
        "features_used": selected_features,
        "feature_count": len(selected_features),
        "holdout_protocol": {
            "held_out_classes": zero_day_names,
            "quarantined_training_partition_flows": int(train_zd_count),
            "known_training_flows": int(len(X_train_known_raw)),
            "known_fit_flows": int(len(X_fit_raw)),
            "known_val_flows": int(len(X_val_raw)),
            "known_testing_flows": int(num_test_known),
            "unseen_testing_flows": int(num_test_zd)
        },
        "performance": {
            "baseline_msp_auroc": round(baseline_msp_auroc * 100, 2),
            "proposed_osr_auroc": round(osr_auroc * 100, 2),
            "fpr_at_95": round(fpr_test * 100, 2),
            "tpr_at_95": round(overall_zd_recall * 100, 2),
            "tau_ood_threshold": round(tau_ood, 4),
            "known_accuracy_retention": round(acc_accepted_known * 100, 2),
            "known_traffic_metrics": {
                "raw_classifier_accuracy": round(raw_classifier_acc * 100, 2),
                "known_flow_acceptance_rate": round(known_coverage * 100, 2),
                "accuracy_on_accepted_flows": round(acc_accepted_known * 100, 2),
                "classification_cum_rejection_rate": round(ccr_known * 100, 2)
            },
            "baseline_msp_calibrated_metrics": {
                "tau_msp_threshold": round(tau_msp, 4),
                "fpr_on_known_flows": round(msp_fpr_test * 100, 2),
                "known_flow_acceptance_rate": round(msp_coverage * 100, 2),
                "accuracy_on_accepted_flows": round(msp_acc_accepted * 100, 2),
                "classification_cum_rejection_rate": round(msp_ccr_known * 100, 2),
                "unseen_zero_day_recall": round(msp_overall_zd_recall * 100, 2),
                "overall_open_set_decision_rate": round(msp_overall_decision_acc * 100, 2)
            },
            "per_class_zero_day_detection": per_class_zd_results,
            "overall_open_set_decision_rate": round(overall_osr_acc * 100, 2)
        },
        "baseline_closed_world_blindness": baseline_misclass,
        "criteria_met": {
            "auroc_over_90": bool(osr_auroc >= 0.90),
            "fpr95_under_5_5": bool(fpr_test <= 0.055),
            "ransomware_over_75": bool(per_class_zd_results["ransomware"]["detection_rate"] >= 75.0),
            "backdoor_over_60": bool(per_class_zd_results["backdoor"]["detection_rate"] >= 60.0),
            "known_raw_accuracy_over_96": bool(raw_classifier_acc >= 0.96),
            "accepted_known_accuracy_over_96": bool(acc_accepted_known >= 0.96),
            "known_ccr_over_90": bool(ccr_known >= 0.90)
        }
    }
    summary_json_path = os.path.join(RESULTS_DIR, "phase_g_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  Saved JSON summary: {summary_json_path}")
    print("\n" + "=" * 80)
    print("PHASE G COMPLETED SUCCESSFULLY WITH STRICT CLASS-HELD-OUT PROTOCOL")
    print("=" * 80)
    return summary_data


if __name__ == "__main__":
    run_phase_g()
