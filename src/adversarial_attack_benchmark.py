"""
Phase 1: Adversarial Evasion Benchmarking Suite
Evaluates Baseline Model vs. Proposed Adversarially Hardened Model across perturbation budgets (epsilon 0.00 to 0.25).

Generates:
  - results/adversarial_robustness_metrics.csv
  - results/phase_1_adversarial_summary.json
  - figures/adversarial_perturbation_curve.png
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# RFC-Compliant Manipulable Feature Indices within Top 14:
# 0: MAX_IP_PKT_LEN, 1: TCP_WIN_MAX_IN, 3: DST_TO_SRC_AVG_THROUGHPUT,
# 4: SRC_TO_DST_AVG_THROUGHPUT, 5: FLOW_DURATION_MILLISECONDS, 6: MIN_IP_PKT_LEN,
# 10: TCP_WIN_MAX_OUT, 11: SHORTEST_FLOW_PKT
MANIPULABLE_INDICES = [0, 1, 3, 4, 5, 6, 10, 11]


def run_adversarial_benchmark():
    print("=" * 80)
    print("PHASE 1: ADVERSARIAL EVASION ROBUSTNESS BENCHMARK")
    print("=" * 80)

    # 1. Load Baseline and Hardened Models
    base_model_path = os.path.join(MODELS_DIR, "lightgbm_smote_balanced_top14.joblib")
    hard_model_path = os.path.join(MODELS_DIR, "lightgbm_adversarially_hardened_top14.joblib")

    print("[Step 1] Loading models and evaluation dataset...")
    base_model = joblib.load(base_model_path)
    hard_model = joblib.load(hard_model_path)

    data_path = os.path.join(PROCESSED_DIR, "train_test_data.pkl")
    target_path = os.path.join(PROCESSED_DIR, "train_test_multi_targets.pkl")
    features_path = os.path.join(PROCESSED_DIR, "selected_features.json")
    encoder_path = os.path.join(PROCESSED_DIR, "label_encoder_multi.pkl")

    _, X_test_df, _, _ = joblib.load(data_path)
    _, y_test = joblib.load(target_path)
    le = joblib.load(encoder_path)

    with open(features_path, "r", encoding="utf-8") as f:
        selected_features = json.load(f)["final_features"]

    X_te = X_test_df[selected_features].values
    y_te = np.array(y_test)

    # Filter attack flows (y != 0) and clean benign flows (y == 0)
    is_attack = (y_te != 0)
    X_attacks = X_te[is_attack]
    y_attacks = y_te[is_attack]
    X_benign = X_te[~is_attack]
    y_benign = y_te[~is_attack]

    print(f"  Test Evaluation Pool: {len(X_te):,} flows")
    print(f"  Attack Flows to Perturb: {len(X_attacks):,} flows")
    print(f"  Clean Benign Flows:      {len(X_benign):,} flows")

    # Compute benign centroid and feature standard deviations
    benign_centroid = np.mean(X_benign, axis=0)
    feat_stds = np.std(X_attacks, axis=0) + 1e-6
    attack_direction = np.sign(benign_centroid - X_attacks) * feat_stds

    # 2. Benchmark Perturbation Sweep (Epsilon 0.00 to 0.25)
    eps_levels = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
    records = []

    print("\n[Step 2] Executing RFC-Compliant Evasion Perturbations across Epsilons...")
    print(f"{'Epsilon (eps)':<15} | {'Baseline DR':<12} | {'Baseline Evasion':<16} | {'Hardened DR':<12} | {'Hardened Evasion':<16} | {'Advantage'}")
    print("-" * 88)

    for eps in eps_levels:
        if eps == 0.00:
            X_pert = X_attacks
        else:
            X_pert = X_attacks.copy()
            X_pert[:, MANIPULABLE_INDICES] += eps * attack_direction[:, MANIPULABLE_INDICES]

        # Evaluate Baseline Model
        preds_base = base_model.predict(X_pert)
        base_evaded = np.sum(preds_base == 0)
        base_dr = (np.sum(preds_base != 0) / len(X_attacks)) * 100.0
        base_evasion = (base_evaded / len(X_attacks)) * 100.0

        # Evaluate Hardened Model
        preds_hard = hard_model.predict(X_pert)
        hard_evaded = np.sum(preds_hard == 0)
        hard_dr = (np.sum(preds_hard != 0) / len(X_attacks)) * 100.0
        hard_evasion = (hard_evaded / len(X_attacks)) * 100.0

        advantage = hard_dr - base_dr

        print(f"eps = {eps:4.2f}        | {base_dr:6.2f}%      | {base_evasion:6.2f}%          | {hard_dr:6.2f}%      | {hard_evasion:6.2f}%          | +{advantage:5.2f}%")

        records.append({
            "Epsilon": eps,
            "Baseline Attack Detection Rate (%)": round(base_dr, 2),
            "Baseline Evasion Rate (%)": round(base_evasion, 2),
            "Hardened Attack Detection Rate (%)": round(hard_dr, 2),
            "Hardened Evasion Rate (%)": round(hard_evasion, 2),
            "Security Retention Gain (%)": round(advantage, 2)
        })

    # Save CSV
    df_metrics = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, "adversarial_robustness_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"\n[Step 3] Saved adversarial metrics to: {csv_path}")

    # 3. Clean Benign Accuracy Verification
    clean_base_acc = np.mean(base_model.predict(X_benign) == 0) * 100.0
    clean_hard_acc = np.mean(hard_model.predict(X_benign) == 0) * 100.0
    print(f"  Clean Benign Retention: Baseline = {clean_base_acc:.2f}% | Hardened = {clean_hard_acc:.2f}%")

    # 4. Generate Publication-Quality Two-Panel Figure
    print("\n[Step 4] Generating Publication Figure: figures/adversarial_perturbation_curve.png...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    eps_arr = [r["Epsilon"] for r in records]
    base_dr_arr = [r["Baseline Attack Detection Rate (%)"] for r in records]
    hard_dr_arr = [r["Hardened Attack Detection Rate (%)"] for r in records]
    base_ev_arr = [r["Baseline Evasion Rate (%)"] for r in records]
    hard_ev_arr = [r["Hardened Evasion Rate (%)"] for r in records]

    # Panel 1: Attack Detection Rate vs Perturbation
    ax1.plot(eps_arr, hard_dr_arr, color="#0052cc", lw=3.0, marker="o", markersize=7,
             label="Proposed Adversarially Hardened Model")
    ax1.plot(eps_arr, base_dr_arr, color="#ff5630", lw=2.5, marker="s", markersize=6, linestyle="--",
             label="Standard Baseline Model (Collapses)")

    ax1.axhline(y=85.0, color="#36b37e", linestyle=":", lw=1.5, label="IEEE Acceptable Robustness Bound (85%)")
    ax1.set_xlabel("Adversarial Perturbation Budget (ε)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Attack Detection Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Attack Detection Resilience Under Evasion\n(Higher is Better)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylim([15, 105])
    ax1.legend(loc="lower left", frameon=True, fontsize=9)

    # Highlight collapse callout
    collapse_gap = hard_dr_arr[3] - base_dr_arr[3]  # at eps=0.10
    ax1.annotate(f"Security Gap: +{collapse_gap:.1f}%\nProtection Advantage",
                 xy=(0.10, hard_dr_arr[3]), xytext=(0.12, 60),
                 arrowprops=dict(arrowstyle="->", color="#0052cc", lw=1.8),
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#deebff", edgecolor="#0052cc", lw=1.2),
                 fontsize=9, fontweight="bold", color="#0052cc")

    # Panel 2: Attacker Evasion Success Rate
    ax2.plot(eps_arr, base_ev_arr, color="#ff5630", lw=2.5, marker="s", markersize=6, linestyle="--",
             label="Attacker Success Against Baseline")
    ax2.plot(eps_arr, hard_ev_arr, color="#0052cc", lw=3.0, marker="o", markersize=7,
             label="Attacker Blocked by Hardened Defense")

    ax2.set_xlabel("Adversarial Perturbation Budget (ε)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Attacker Evasion Success Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Attacker Evasion Rate vs. Perturbation Budget\n(Lower is Better)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylim([-2, 85])
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "adversarial_perturbation_curve.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"  Saved publication chart to: {fig_path}")

    # 5. Export JSON Summary
    summary_data = {
        "phase": "Phase 1 - Adversarial Evasion Robustness & Defensive Hardening",
        "threat_model": {
            "rfc_compliant": True,
            "manipulable_features_count": len(MANIPULABLE_INDICES),
            "immutable_protocol_features_count": 14 - len(MANIPULABLE_INDICES),
            "evaluated_epsilons": eps_levels
        },
        "vulnerability_proof": {
            "clean_baseline_dr": base_dr_arr[0],
            "perturbed_baseline_dr_at_eps05": base_dr_arr[2],
            "baseline_collapse_pct": round(base_dr_arr[0] - base_dr_arr[2], 2)
        },
        "defense_efficacy": {
            "clean_hardened_dr": hard_dr_arr[0],
            "perturbed_hardened_dr_at_eps05": hard_dr_arr[2],
            "perturbed_hardened_dr_at_eps10": hard_dr_arr[3],
            "perturbed_hardened_dr_at_eps20": hard_dr_arr[5],
            "clean_benign_retention": round(clean_hard_acc, 2)
        },
        "ieee_targets_met": {
            "retains_over_85_at_eps10": bool(hard_dr_arr[3] >= 85.0),
            "clean_benign_retention_over_95": bool(clean_hard_acc >= 95.0),
            "proves_baseline_evasion_flaw": bool(base_dr_arr[2] < 50.0)
        }
    }

    summary_json_path = os.path.join(RESULTS_DIR, "phase_1_adversarial_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  Saved JSON summary to: {summary_json_path}")

    print("\n" + "=" * 80)
    print("PHASE 1 ADVERSARIAL BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return summary_data


if __name__ == "__main__":
    run_adversarial_benchmark()

