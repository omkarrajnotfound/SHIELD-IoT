"""
Phase 4: Two-Stage Hybrid Feature Selection Pipeline
Stage 1: Statistical Filter (ANOVA F-test / Variance) -> Cuts 39 features down to ~22
Stage 2: SHAP Attribution Ranking -> Isolate Top 12 to 14 Critical Features
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif, SelectKBest
import shap
from lightgbm import LGBMClassifier

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_feature_selection(top_k_stage1=22, top_k_final=14, sample_shap=5000):
    print("=" * 60)
    print("PHASE 4: TWO-STAGE HYBRID FEATURE SELECTION")
    print("=" * 60)
    
    data_path = os.path.join(PROCESSED_DIR, "train_test_data.pkl")
    X_train, X_test, y_train, y_test = joblib.load(data_path)
    print(f"Initial Feature Count: {X_train.shape[1]}")

    # ---------------------------------------------------------
    # STAGE 1: Fast Statistical Filter (ANOVA F-Value)
    # ---------------------------------------------------------
    print(f"\n[Stage 1] Computing ANOVA F-scores across {X_train.shape[1]} features...")
    # Subsample for speed
    sub_idx = np.random.RandomState(42).choice(len(X_train), min(100000, len(X_train)), replace=False)
    X_sub = X_train.iloc[sub_idx]
    y_sub = y_train.iloc[sub_idx]

    f_values, p_values = f_classif(X_sub, y_sub)
    f_scores_df = pd.DataFrame({
        "feature": X_train.columns,
        "f_score": np.nan_to_num(f_values, nan=0.0)
    }).sort_values(by="f_score", ascending=False)

    stage1_features = f_scores_df.head(top_k_stage1)["feature"].tolist()
    print(f"[Stage 1 Filter Complete] Selected Top {len(stage1_features)} Features:")
    for i, feat in enumerate(stage1_features[:8]):
        print(f"  {i+1}. {feat} (F-score: {f_scores_df.loc[f_scores_df['feature'] == feat, 'f_score'].values[0]:.1f})")
    print(f"  ... and {len(stage1_features) - 8} more.")

    # ---------------------------------------------------------
    # STAGE 2: TreeSHAP Attribution-Guided Ranking
    # ---------------------------------------------------------
    print(f"\n[Stage 2] Training LightGBM on Stage-1 candidate features for SHAP attribution...")
    X_train_s1 = X_train[stage1_features]
    
    lgb_s1 = LGBMClassifier(n_estimators=60, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    lgb_s1.fit(X_train_s1.iloc[sub_idx], y_sub)

    print(f"Calculating exact TreeSHAP values for {sample_shap} flow samples...")
    shap_sample = X_train_s1.iloc[sub_idx[:sample_shap]]
    explainer = shap.TreeExplainer(lgb_s1)
    shap_values = explainer.shap_values(shap_sample)

    # For binary classification, shap_values can be a list or array
    if isinstance(shap_values, list):
        # class 1 (attack)
        vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        vals = shap_values

    # Mean absolute SHAP value per feature: E[|\phi_j|]
    mean_abs_shap = np.mean(np.abs(vals), axis=0)
    shap_df = pd.DataFrame({
        "feature": stage1_features,
        "mean_shap_value": mean_abs_shap
    }).sort_values(by="mean_shap_value", ascending=False)

    final_features = shap_df.head(top_k_final)["feature"].tolist()
    print(f"\n[Stage 2 Complete] Final {top_k_final} Critical Features Selected by SHAP:")
    for i, feat in enumerate(final_features):
        score = shap_df.loc[shap_df['feature'] == feat, 'mean_shap_value'].values[0]
        print(f"  {i+1:2d}. {feat:<30} (SHAP importance: {score:.4f})")

    # Save feature lists
    feature_meta = {
        "initial_feature_count": X_train.shape[1],
        "stage1_feature_count": len(stage1_features),
        "stage1_features": stage1_features,
        "final_feature_count": len(final_features),
        "final_features": final_features
    }
    feat_json_path = os.path.join(PROCESSED_DIR, "selected_features.json")
    with open(feat_json_path, "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=2)
    print(f"\nFeature metadata successfully saved to: {feat_json_path}")

    # Generate and save SHAP Feature Importance Plot
    plt.figure(figsize=(10, 6), dpi=300)
    top_plot_df = shap_df.head(top_k_final).sort_values(by="mean_shap_value", ascending=True)
    palette = sns.color_palette("mako", len(top_plot_df))
    plt.barh(top_plot_df["feature"], top_plot_df["mean_shap_value"], color=palette)
    plt.xlabel("Mean Absolute SHAP Value (Impact on Prediction)", fontsize=11, fontweight="bold")
    plt.title(f"Top {top_k_final} Most Informative Network Features (Stage-2 SHAP Ranking)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    fig_path = os.path.join(FIGURES_DIR, "shap_feature_importance_top14.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved SHAP feature importance chart to: {fig_path}")

    return final_features

if __name__ == "__main__":
    run_feature_selection()

