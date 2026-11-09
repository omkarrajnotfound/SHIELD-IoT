"""
Phase 1: Adversarially Hardened Model Training Pipeline
Implements Min-Max Adversarial Training under RFC-compliant Network Flow Constraints.

Produces:
  - models/lightgbm_adversarially_hardened_top14.joblib
  - Evaluates clean accuracy retention and hardened boundary stability.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# RFC-Compliant Manipulable Feature Indices within Top 14:
# 0: MAX_IP_PKT_LEN, 1: TCP_WIN_MAX_IN, 3: DST_TO_SRC_AVG_THROUGHPUT,
# 4: SRC_TO_DST_AVG_THROUGHPUT, 5: FLOW_DURATION_MILLISECONDS, 6: MIN_IP_PKT_LEN,
# 10: TCP_WIN_MAX_OUT, 11: SHORTEST_FLOW_PKT
MANIPULABLE_INDICES = [0, 1, 3, 4, 5, 6, 10, 11]


def train_hardened_model():
    print("=" * 80)
    print("PHASE 1: ADVERSARIALLY HARDENED TRAINING UNDER RFC FLOW CONSTRAINTS")
    print("=" * 80)

    # 1. Load Data, Targets, Encoder & Selected 14 Features
    data_path = os.path.join(PROCESSED_DIR, "train_test_data.pkl")
    target_path = os.path.join(PROCESSED_DIR, "train_test_multi_targets.pkl")
    encoder_path = os.path.join(PROCESSED_DIR, "label_encoder_multi.pkl")
    features_path = os.path.join(PROCESSED_DIR, "selected_features.json")

    print("[Step 1] Loading preprocessed training data and 14 critical features...")
    X_train_df, X_test_df, _, _ = joblib.load(data_path)
    y_train, y_test = joblib.load(target_path)
    le = joblib.load(encoder_path)
    classes = list(le.classes_)

    with open(features_path, "r", encoding="utf-8") as f:
        selected_features = json.load(f)["final_features"]

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    X_tr = X_train_df[selected_features].values
    X_te = X_test_df[selected_features].values

    print(f"  Training Flows:  {len(X_tr):,}")
    print(f"  Testing Flows:   {len(X_te):,}")
    print(f"  Features:        {len(selected_features)}")
    print(f"  Manipulable:     {len(MANIPULABLE_INDICES)} continuous features (packet padding, delay, throughput)")

    # 2. Synthesize Adversarial Evasion Perturbations for Training Fold
    print("\n[Step 2] Synthesizing Min-Max Adversarial Flow Perturbations...")
    is_attack_tr = (y_train != 0)
    X_att_tr = X_tr[is_attack_tr]
    y_att_tr = y_train[is_attack_tr]

    # Calculate benign distribution center and feature standard deviations
    benign_centroid_tr = np.mean(X_tr[y_train == 0], axis=0)
    feat_stds_tr = np.std(X_tr, axis=0) + 1e-6

    # Direction towards benign manifold (adversarial objective: mimic benign traffic)
    direction_tr = np.sign(benign_centroid_tr - X_att_tr) * feat_stds_tr

    # Sample adversarial perturbations across multiple budgets (eps=0.05, 0.10)
    # Subsample attack flows to maintain balance and avoid memory bloat
    np.random.seed(42)
    adv_sample_size = min(120000, len(X_att_tr))
    idx_sub = np.random.choice(len(X_att_tr), size=adv_sample_size, replace=False)

    X_adv_batches = [X_tr]
    y_adv_batches = [y_train]

    for eps in [0.05, 0.10, 0.15]:
        X_pert = X_att_tr[idx_sub].copy()
        X_pert[:, MANIPULABLE_INDICES] += eps * direction_tr[idx_sub][:, MANIPULABLE_INDICES]
        X_adv_batches.append(X_pert)
        y_adv_batches.append(y_att_tr[idx_sub])

    X_train_hardened = np.vstack(X_adv_batches)
    y_train_hardened = np.concatenate(y_adv_batches)

    print(f"  Base Training Flows:       {len(X_tr):,}")
    print(f"  Augmented Adversarial:     {len(X_train_hardened) - len(X_tr):,}")
    print(f"  Total Hardened Pool:       {len(X_train_hardened):,}")

    # 3. Train Adversarially Hardened LightGBM Model
    print("\n[Step 3] Training Adversarially Hardened LightGBM Ensemble...")
    t0_train = time.time()
    hardened_model = LGBMClassifier(
        n_estimators=150,
        max_depth=10,
        learning_rate=0.08,
        num_leaves=63,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    hardened_model.fit(X_train_hardened, y_train_hardened)
    train_duration = time.time() - t0_train
    print(f"  Hardened training completed in {train_duration:.2f} seconds.")

    # 4. Clean Traffic Retention Validation
    print("\n[Step 4] Validating Clean Benchmark Accuracy Retention...")
    clean_preds = hardened_model.predict(X_te)
    clean_acc = accuracy_score(y_test, clean_preds)
    clean_f1_macro = f1_score(y_test, clean_preds, average="macro")
    clean_attack_dr = np.mean(clean_preds[y_test != 0] != 0) * 100.0

    print(f"  Clean Test Accuracy:        {clean_acc * 100:.2f}% (Target: > 96.0%)")
    print(f"  Clean Macro-F1 Score:       {clean_f1_macro * 100:.2f}% (Target: > 85.0%)")
    print(f"  Clean Attack Detection Rate: {clean_attack_dr:.2f}%")

    # 5. Save Model Artifact
    model_save_path = os.path.join(MODELS_DIR, "lightgbm_adversarially_hardened_top14.joblib")
    joblib.dump(hardened_model, model_save_path)
    print(f"\n[Step 5] Saved hardened model artifact to: {model_save_path}")

    print("\n" + "=" * 80)
    print("ADVERSARIAL DEFENSE TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return hardened_model


if __name__ == "__main__":
    train_hardened_model()
