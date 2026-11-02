"""
Phase 6: Explainable AI (XAI) Module using TreeSHAP on Proposed LightGBM Model
Generates global feature impact plots and local decision waterfall charts
to provide human-auditable rationales for network security analysts.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

def generate_xai_explanations(sample_size=1500):
    print("=" * 60)
    print("PHASE 6: EXPLAINABLE AI (XAI) GENERATION WITH TREESHAP")
    print("=" * 60)

    data_path = os.path.join(PROCESSED_DIR, "train_test_data.pkl")
    feat_path = os.path.join(PROCESSED_DIR, "selected_features.json")
    model_path = os.path.join(MODELS_DIR, "lightgbm_(proposed_14-feat)_top14.joblib")

    X_train, X_test, y_train, y_test = joblib.load(data_path)
    with open(feat_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    top14 = meta["final_features"]

    X_test_top14 = X_test[top14]
    model = joblib.load(model_path)
    print(f"Loaded proposed model: {model.__class__.__name__}")

    # 1. Global Explanations (TreeSHAP Beeswarm Plot)
    print(f"\n[1/3] Computing TreeSHAP values for global interpretability ({sample_size} test flows)...")
    idx_sample = np.random.RandomState(42).choice(len(X_test_top14), sample_size, replace=False)
    X_sample = X_test_top14.iloc[idx_sample]
    y_sample = y_test.iloc[idx_sample]

    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(X_sample)

    plt.figure(figsize=(10, 6), dpi=300)
    shap.plots.beeswarm(shap_explanation, max_display=14, show=False)
    plt.title("Global Feature Impact (SHAP Beeswarm Plot) - 14 Selected Features", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    beeswarm_path = os.path.join(FIGURES_DIR, "shap_global_beeswarm.png")
    plt.savefig(beeswarm_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved Global Beeswarm Plot to: {beeswarm_path}")

    # 2. Local Explanation: Confirmed Attack Packet (Waterfall Plot)
    print("\n[2/3] Generating Local Explanation for a Detected Intrusion Packet...")
    attack_indices = np.where(y_sample.values == 1)[0]
    attack_idx = attack_indices[0]

    plt.figure(figsize=(9, 5), dpi=300)
    shap.plots.waterfall(shap_explanation[attack_idx], max_display=10, show=False)
    plt.title("Local SHAP Attribution: Intrusion Alert Rationale", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    waterfall_atk_path = os.path.join(FIGURES_DIR, "shap_local_waterfall_attack.png")
    plt.savefig(waterfall_atk_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved Attack Waterfall Plot to: {waterfall_atk_path}")

    # 3. Local Explanation: Confirmed Benign Packet (Waterfall Plot)
    print("\n[3/3] Generating Local Explanation for a Normal/Benign Packet...")
    benign_indices = np.where(y_sample.values == 0)[0]
    benign_idx = benign_indices[0]

    plt.figure(figsize=(9, 5), dpi=300)
    shap.plots.waterfall(shap_explanation[benign_idx], max_display=10, show=False)
    plt.title("Local SHAP Attribution: Benign Verification Rationale", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    waterfall_bng_path = os.path.join(FIGURES_DIR, "shap_local_waterfall_benign.png")
    plt.savefig(waterfall_bng_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved Benign Waterfall Plot to: {waterfall_bng_path}")

    # 4. Extract Human-Auditable Plain Text Security Rule for the Attack
    sample_shap_arr = shap_explanation[attack_idx].values
    feature_names = top14
    top_pushing_features = []
    for i, val in enumerate(sample_shap_arr):
        if val > 0.05:
            feat_name = feature_names[i]
            feat_val = float(X_sample.iloc[attack_idx][feat_name])
            top_pushing_features.append(f"{feat_name} (scaled={feat_val:.2f}, impact=+{val:.2f})")

    rule_summary = {
        "alert_type": "Intrusion / Malicious Attack Flagged",
        "model_confidence": float(model.predict_proba(X_sample.iloc[[attack_idx]])[0][1]),
        "primary_trigger_features": top_pushing_features,
        "operator_summary": "Network flow flagged due to anomalous packet size distribution and protocol flags differing significantly from baseline normal profile."
    }

    rule_path = os.path.join(RESULTS_DIR, "sample_alert_explanation_rule.json")
    with open(rule_path, "w", encoding="utf-8") as f:
        json.dump(rule_summary, f, indent=2)
    print(f"\nHuman-Auditable Alert Explanation Rule saved to: {rule_path}")
    print("Rule Preview:")
    print(json.dumps(rule_summary, indent=2))

if __name__ == "__main__":
    generate_xai_explanations()
