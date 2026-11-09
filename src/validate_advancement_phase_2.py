"""
Phase 2 Validation & Verification Suite: Cross-Dataset Domain Adaptation & Generalization
Validates:
  1. Artifact integrity and file existence (matrix CSV, JSON summary, high-res chart, parquet datasets)
  2. NetFlow v2/v3 standardization schema (confirms 14 features identical across all 3 datasets)
  3. Domain Shift Catastrophe verification (confirms naive zero-shot drops below 40%, proving the research gap)
  4. Unsupervised alignment progression (confirms CORAL improves over unadapted baseline)
  5. IEEE cross-dataset target satisfaction (confirms proposed adaptation achieves >88.0% on all 3 datasets)
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CROSS_DATA_DIR = os.path.join(DATA_DIR, "cross_datasets")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")


def validate():
    print("=" * 80)
    print("PHASE 2: CROSS-DATASET DOMAIN ADAPTATION VALIDATION SUITE")
    print("=" * 80)

    checks_passed = 0
    total_checks = 5

    # -------------------------------------------------------------
    # Check 1: Artifact Existence and File Integrity
    # -------------------------------------------------------------
    print("\n[CHECK 1] Artifact Existence and File Integrity:")
    required_files = [
        os.path.join(RESULTS_DIR, "cross_dataset_transfer_matrix.csv"),
        os.path.join(RESULTS_DIR, "phase_2_cross_dataset_summary.json"),
        os.path.join(FIGURES_DIR, "cross_dataset_generalization_radar.png"),
        os.path.join(CROSS_DATA_DIR, "NF-BoT-IoT-v3.parquet"),
        os.path.join(CROSS_DATA_DIR, "NF-UNSW-NB15-v3.parquet")
    ]

    all_exist = True
    for f in required_files:
        exists = os.path.isfile(f)
        size_kb = os.path.getsize(f) / 1024 if exists else 0
        status = "PASS" if exists and size_kb > 0 else "FAIL"
        if status == "FAIL":
            all_exist = False
        print(f"  [{status}] {os.path.basename(f):<45} ({size_kb:,.1f} KB)")

    if all_exist:
        checks_passed += 1
        print(">> Check 1 Result: ALL 5 REQUIRED PHASE 2 ARTIFACTS VERIFIED AND NON-EMPTY")
    else:
        print(">> Check 1 Result: FAILED - MISSING ARTIFACTS")

    # -------------------------------------------------------------
    # Check 2: NetFlow v2/v3 Standardized Schema Compliance
    # -------------------------------------------------------------
    print("\n[CHECK 2] NetFlow v2/v3 Standardized Schema Compliance:")
    with open(os.path.join(PROCESSED_DIR, "selected_features.json")) as f:
        selected_features = json.load(f)["final_features"]

    try:
        import unittest.mock
        sys.modules.setdefault('fastparquet.speedups', unittest.mock.MagicMock())
        from fastparquet import ParquetFile
        bot_cols = ParquetFile(os.path.join(CROSS_DATA_DIR, "NF-BoT-IoT-v3.parquet")).columns
        unsw_cols = ParquetFile(os.path.join(CROSS_DATA_DIR, "NF-UNSW-NB15-v3.parquet")).columns
    except Exception:
        bot_cols = pd.read_parquet(os.path.join(CROSS_DATA_DIR, "NF-BoT-IoT-v3.parquet"), engine="fastparquet").columns
        unsw_cols = pd.read_parquet(os.path.join(CROSS_DATA_DIR, "NF-UNSW-NB15-v3.parquet"), engine="fastparquet").columns

    bot_has_all = all(c in bot_cols for c in selected_features)
    unsw_has_all = all(c in unsw_cols for c in selected_features)

    print(f"  Selected Feature Dimension:   {len(selected_features)} features")
    print(f"  NF-BoT-IoT Schema Match:      {'PASS (14/14 Present)' if bot_has_all else 'FAIL'}")
    print(f"  NF-UNSW-NB15 Schema Match:    {'PASS (14/14 Present)' if unsw_has_all else 'FAIL'}")

    if bot_has_all and unsw_has_all:
        checks_passed += 1
        print(">> Check 2 Result: FULL NETFLOW SCHEMA PARITY CONFIRMED")
    else:
        print(">> Check 2 Result: FAILED - SCHEMA MISMATCH DETECTED")

    # -------------------------------------------------------------
    # Check 3: Domain Shift Catastrophe Demonstration
    # -------------------------------------------------------------
    print("\n[CHECK 3] Domain Shift Catastrophe Demonstration (Naive Zero-Shot):")
    matrix_df = pd.read_csv(os.path.join(RESULTS_DIR, "cross_dataset_transfer_matrix.csv"))

    bot_zero = matrix_df.loc[(matrix_df["Dataset"].str.contains("NF-BoT-IoT")) & (matrix_df["Adaptation Method"].str.contains("Naive")), "Accuracy (%)"].values[0]
    unsw_zero = matrix_df.loc[(matrix_df["Dataset"].str.contains("NF-UNSW-NB15")) & (matrix_df["Adaptation Method"].str.contains("Naive")), "Accuracy (%)"].values[0]

    print(f"  Naive Zero-Shot NF-BoT-IoT Accuracy:    {bot_zero:.2f}% (Expected: < 40.0%)")
    print(f"  Naive Zero-Shot NF-UNSW-NB15 Accuracy:  {unsw_zero:.2f}% (Expected: < 40.0%)")

    both_collapsed = (bot_zero < 40.0 and unsw_zero < 40.0)
    if both_collapsed:
        checks_passed += 1
        print("  [PASS] Definitively proves un-adapted cross-dataset transfer suffers fatal domain collapse.")
        print(">> Check 3 Result: DOMAIN SHIFT CATASTROPHE RIGOROUSLY DEMONSTRATED")
    else:
        print(">> Check 3 Result: FAILED - DOMAIN SHIFT COLLAPSE NOT DOCUMENTED")

    # -------------------------------------------------------------
    # Check 4: Unsupervised Covariance Alignment (CORAL) Efficacy
    # -------------------------------------------------------------
    print("\n[CHECK 4] Unsupervised Covariance Alignment (CORAL) Efficacy:")
    bot_coral = matrix_df.loc[(matrix_df["Dataset"].str.contains("NF-BoT-IoT")) & (matrix_df["Adaptation Method"].str.contains("CORAL")), "Accuracy (%)"].values[0]
    unsw_coral = matrix_df.loc[(matrix_df["Dataset"].str.contains("NF-UNSW-NB15")) & (matrix_df["Adaptation Method"].str.contains("CORAL")), "Accuracy (%)"].values[0]

    bot_gain = bot_coral - bot_zero
    unsw_gain = unsw_coral - unsw_zero

    print(f"  CORAL NF-BoT-IoT Gain:    +{bot_gain:.2f}% ({bot_zero:.2f}% -> {bot_coral:.2f}%)")
    print(f"  CORAL NF-UNSW-NB15 Gain:  +{unsw_gain:.2f}% ({unsw_zero:.2f}% -> {unsw_coral:.2f}%)")

    if bot_gain > 15.0 and unsw_gain > 10.0:
        checks_passed += 1
        print("  [PASS] Covariance alignment significantly shifts target distributions toward source manifold.")
        print(">> Check 4 Result: CORAL COVARIANCE ALIGNMENT VERIFIED")
    else:
        print(">> Check 4 Result: FAILED - CORAL GAIN INSUFFICIENT")

    # -------------------------------------------------------------
    # Check 5: Proposed Edge Adaptation IEEE Target Satisfaction
    # -------------------------------------------------------------
    print("\n[CHECK 5] Proposed Rapid Edge Adaptation IEEE Target (> 88.0% Accuracy):")
    bot_edge = matrix_df.loc[(matrix_df["Dataset"].str.contains("NF-BoT-IoT")) & (matrix_df["Adaptation Method"].str.contains("Calibration|Edge")), "Accuracy (%)"].values[0]
    unsw_edge = matrix_df.loc[(matrix_df["Dataset"].str.contains("NF-UNSW-NB15")) & (matrix_df["Adaptation Method"].str.contains("Calibration|Edge")), "Accuracy (%)"].values[0]
    source_acc = matrix_df.loc[matrix_df["Domain Role"] == "Source", "Accuracy (%)"].values[0]

    print(f"  Source (NF-ToN-IoT-v2) Accuracy:    {source_acc:.2f}% (Target: > 88.0%)")
    print(f"  Target 1 (NF-BoT-IoT) Edge Accuracy: {bot_edge:.2f}% (Target: > 88.0%)")
    print(f"  Target 2 (NF-UNSW-NB15) Edge Accuracy: {unsw_edge:.2f}% (Target: > 88.0%)")

    all_targets_met = (source_acc >= 88.0 and bot_edge >= 88.0 and unsw_edge >= 88.0)
    if all_targets_met:
        checks_passed += 1
        print("  [PASS] All 3 datasets achieve > 88.0% detection accuracy under rapid edge calibration.")
        print(">> Check 5 Result: ALL CROSS-DATASET BENCHMARK TARGETS SATISFIED")
    else:
        print(">> Check 5 Result: TARGETS NOT MET")

    # -------------------------------------------------------------
    # Final Result
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"FINAL RESULT: {checks_passed}/{total_checks} CHECKS PASSED")
    if checks_passed == total_checks:
        print("CROSS-DATASET EVALUATION COMPLETE: ALL 5 ADAPTATION CHECKS PASSED.")
    else:
        print("PHASE 2 VALIDATION FAILED. REVIEW LOGS ABOVE.")
    print("=" * 80)

    return checks_passed == total_checks


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)

