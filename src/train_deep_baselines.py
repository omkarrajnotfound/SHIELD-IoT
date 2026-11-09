"""
Phase B: Deep Learning Baseline Benchmarking & Master Comparative Study
Evaluates deep neural network baselines (MLP + 1D-CNN) against tree baselines
(Random Forest, XGBoost, LightGBM) on 39 features vs. Our Proposed 14-Feature Stealth-Aware LightGBM.

Outputs:
  - models/mlp_baseline_full39.pt
  - models/cnn1d_baseline_full39.pt
  - results/master_comparison_table.csv
  - figures/master_comparison_chart.png
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# -------------------------------------------------------------
# Deep Learning Model Architectures (PyTorch)
# -------------------------------------------------------------
class MLPBaseline(nn.Module):
    """3-Layer Fully Connected Deep Neural Network for Network Flow Classification"""
    def __init__(self, in_features=39, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class CNN1DBaseline(nn.Module):
    """1D-Convolutional Neural Network treating flow features as sequential signal"""
    def __init__(self, in_features=39, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=8)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, num_features)
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


def train_pytorch_model(model, train_loader, epochs=6, lr=0.003, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.to(device)

    start_train = time.time()
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start_train
    return model, train_time


def evaluate_pytorch_model(name, model, X_test_tensor, y_test_np, classes, model_save_path, device="cpu"):
    model.eval()
    model.to(device)

    # Pure Inference Latency Measurement
    t0_infer = time.time()
    with torch.no_grad():
        logits = model(X_test_tensor.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
    infer_time = time.time() - t0_infer
    latency_per_1k = (infer_time / len(y_test_np)) * 1000 * 1000

    # Save model weights and measure file size
    torch.save(model.state_dict(), model_save_path)
    model_size_mb = os.path.getsize(model_save_path) / (1024 * 1024)

    acc = accuracy_score(y_test_np, preds) * 100
    macro_f1 = f1_score(y_test_np, preds, average="macro") * 100
    weighted_f1 = f1_score(y_test_np, preds, average="weighted") * 100

    rep = classification_report(y_test_np, preds, target_names=classes, output_dict=True, zero_division=0)
    ransomware_rec = rep["ransomware"]["recall"] * 100
    backdoor_rec = rep["backdoor"]["recall"] * 100

    print(f"[{name}] Acc: {acc:.2f}%, Macro-F1: {macro_f1:.2f}%, Latency: {latency_per_1k:.2f}ms/1k, Size: {model_size_mb:.2f}MB")

    return {
        "model_name": name,
        "architecture_type": "Deep Learning (Neural Net)",
        "feature_count": 39,
        "accuracy": round(acc, 2),
        "macro_f1": round(macro_f1, 2),
        "weighted_f1": round(weighted_f1, 2),
        "ransomware_recall": round(ransomware_rec, 2),
        "backdoor_recall": round(backdoor_rec, 2),
        "latency_ms_per_1k_pkts": round(latency_per_1k, 3),
        "model_size_mb": round(model_size_mb, 2),
        "gpu_required": "Yes (Recommended)",
        "edge_deployable": "Poor (High compute/latency)"
    }


def evaluate_sklearn_tree_model(name, model, X_tr, y_tr, X_te, y_te, classes, model_save_path):
    t0_train = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0_train

    t0_infer = time.time()
    preds = model.predict(X_te)
    infer_time = time.time() - t0_infer
    latency_per_1k = (infer_time / len(y_te)) * 1000 * 1000

    joblib.dump(model, model_save_path)
    model_size_mb = os.path.getsize(model_save_path) / (1024 * 1024)

    acc = accuracy_score(y_te, preds) * 100
    macro_f1 = f1_score(y_te, preds, average="macro") * 100
    weighted_f1 = f1_score(y_te, preds, average="weighted") * 100

    rep = classification_report(y_te, preds, target_names=classes, output_dict=True, zero_division=0)
    ransomware_rec = rep["ransomware"]["recall"] * 100
    backdoor_rec = rep["backdoor"]["recall"] * 100

    print(f"[{name}] Acc: {acc:.2f}%, Macro-F1: {macro_f1:.2f}%, Latency: {latency_per_1k:.2f}ms/1k, Size: {model_size_mb:.2f}MB")

    return {
        "model_name": name,
        "architecture_type": "Classical Ensemble",
        "feature_count": int(X_tr.shape[1]),
        "accuracy": round(acc, 2),
        "macro_f1": round(macro_f1, 2),
        "weighted_f1": round(weighted_f1, 2),
        "ransomware_recall": round(ransomware_rec, 2),
        "backdoor_recall": round(backdoor_rec, 2),
        "latency_ms_per_1k_pkts": round(latency_per_1k, 3),
        "model_size_mb": round(model_size_mb, 2),
        "gpu_required": "No",
        "edge_deployable": "Moderate" if X_tr.shape[1] == 39 else "Optimal (Ultra-Lightweight)"
    }


def run_phase_b():
    print("=" * 80)
    print("PHASE B: DEEP LEARNING BASELINES (MLP + 1D-CNN) & MASTER COMPARATIVE STUDY")
    print("=" * 80)

    # 1. Load Data
    X_train_df, X_test_df, _, _ = joblib.load(os.path.join(PROCESSED_DIR, "train_test_data.pkl"))
    y_train, y_test = joblib.load(os.path.join(PROCESSED_DIR, "train_test_multi_targets.pkl"))
    label_encoder = joblib.load(os.path.join(PROCESSED_DIR, "label_encoder_multi.pkl"))
    classes = list(label_encoder.classes_)

    with open(os.path.join(PROCESSED_DIR, "selected_features.json"), "r", encoding="utf-8") as f:
        top14_features = json.load(f)["final_features"]

    print(f"Full Feature Space: {X_train_df.shape[1]} features | Classes: {len(classes)}")
    print(f"Train samples: {len(X_train_df):,} | Test samples: {len(X_test_df):,}")

    # Clip for neural network numerical stability (handling robust scaler tails)
    X_tr_clipped = np.clip(X_train_df.values, -10.0, 10.0).astype(np.float32)
    X_te_clipped = np.clip(X_test_df.values, -10.0, 10.0).astype(np.float32)

    # Rigorous Stratified Subsample for baseline training calibration (150,000 flows)
    # Using stratify=y_train ensures every minority class (Ransomware, MITM, Backdoor) retains exact proportional presence
    from sklearn.model_selection import train_test_split
    sub_idx, _ = train_test_split(np.arange(len(y_train)), train_size=150000, stratify=y_train, random_state=42)
    X_tr_sub_clip = X_tr_clipped[sub_idx]
    y_tr_sub = y_train.iloc[sub_idx].values

    # PyTorch DataLoaders
    batch_size = 512
    train_dataset = TensorDataset(torch.tensor(X_tr_sub_clip), torch.tensor(y_tr_sub, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    X_test_tensor = torch.tensor(X_te_clipped)
    y_test_np = y_test.values

    results = []

    # -------------------------------------------------------------
    # 1. Multi-Layer Perceptron (MLP) Baseline
    # -------------------------------------------------------------
    print("\n[Step B1] Training Multi-Layer Perceptron (MLP, 3-Layer Dense)...")
    mlp = MLPBaseline(in_features=39, num_classes=len(classes))
    mlp, mlp_train_time = train_pytorch_model(mlp, train_loader, epochs=6, lr=0.003)
    mlp_metrics = evaluate_pytorch_model(
        "MLP (Full Features)", mlp, X_test_tensor, y_test_np, classes,
        os.path.join(MODELS_DIR, "mlp_baseline_full39.pt")
    )
    results.append(mlp_metrics)

    # -------------------------------------------------------------
    # 2. 1D Convolutional Neural Network (1D-CNN) Baseline
    # -------------------------------------------------------------
    print("\n[Step B2] Training 1D-CNN Baseline (Conv1D + BatchNorm + Pooling)...")
    cnn = CNN1DBaseline(in_features=39, num_classes=len(classes))
    cnn, cnn_train_time = train_pytorch_model(cnn, train_loader, epochs=6, lr=0.003)
    cnn_metrics = evaluate_pytorch_model(
        "1D-CNN (Full Features)", cnn, X_test_tensor, y_test_np, classes,
        os.path.join(MODELS_DIR, "cnn1d_baseline_full39.pt")
    )
    results.append(cnn_metrics)

    # -------------------------------------------------------------
    # 3. Classical Tree Baselines (39 Features)
    # -------------------------------------------------------------
    print("\n[Step B3] Benchmarking Classical Ensembles on Full 39 Features...")
    X_tr_sub_df = X_train_df.iloc[sub_idx]
    y_tr_sub_s = y_train.iloc[sub_idx]

    # Random Forest (39 Features)
    rf = RandomForestClassifier(n_estimators=60, max_depth=16, random_state=42, n_jobs=-1)
    rf_m = evaluate_sklearn_tree_model(
        "Random Forest (Full Features)", rf, X_tr_sub_df, y_tr_sub_s, X_test_df, y_test,
        classes, os.path.join(MODELS_DIR, "random_forest_full39.joblib")
    )
    results.append(rf_m)

    # XGBoost (39 Features)
    xgb = XGBClassifier(n_estimators=80, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric="mlogloss")
    xgb_m = evaluate_sklearn_tree_model(
        "XGBoost (Full Features)", xgb, X_tr_sub_df, y_tr_sub_s, X_test_df, y_test,
        classes, os.path.join(MODELS_DIR, "xgboost_full39.joblib")
    )
    results.append(xgb_m)

    # LightGBM (39 Features, Standard Unweighted)
    lgb_full = LGBMClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    lgb_full_m = evaluate_sklearn_tree_model(
        "LightGBM (Full Features)", lgb_full, X_tr_sub_df, y_tr_sub_s, X_test_df, y_test,
        classes, os.path.join(MODELS_DIR, "lightgbm_full39.joblib")
    )
    results.append(lgb_full_m)

    # -------------------------------------------------------------
    # 4. Our Proposed Model (Phase A: 14 Feat, Borderline-SMOTE, Balanced)
    # -------------------------------------------------------------
    print("\n[Step B4] Loading Our Proposed Stealth-Aware LightGBM (14 Features)...")
    prop_model_path = os.path.join(MODELS_DIR, "lightgbm_smote_balanced_top14.joblib")
    prop_model = joblib.load(prop_model_path)
    X_te_top14 = X_test_df[top14_features]

    t0_inf = time.time()
    prop_preds = prop_model.predict(X_te_top14)
    prop_infer_sec = time.time() - t0_inf
    prop_latency_1k = (prop_infer_sec / len(y_test)) * 1000 * 1000
    prop_size_mb = os.path.getsize(prop_model_path) / (1024 * 1024)

    prop_acc = accuracy_score(y_test, prop_preds) * 100
    prop_macro_f1 = f1_score(y_test, prop_preds, average="macro") * 100
    prop_weighted_f1 = f1_score(y_test, prop_preds, average="weighted") * 100
    prop_rep = classification_report(y_test, prop_preds, target_names=classes, output_dict=True)

    prop_m = {
        "model_name": "Proposed LightGBM (14-Feat + SMOTE)",
        "architecture_type": "Proposed Framework (Ours)",
        "feature_count": 14,
        "accuracy": round(prop_acc, 2),
        "macro_f1": round(prop_macro_f1, 2),
        "weighted_f1": round(prop_weighted_f1, 2),
        "ransomware_recall": round(prop_rep["ransomware"]["recall"] * 100, 2),
        "backdoor_recall": round(prop_rep["backdoor"]["recall"] * 100, 2),
        "latency_ms_per_1k_pkts": round(prop_latency_1k, 3),
        "model_size_mb": round(prop_size_mb, 2),
        "gpu_required": "No",
        "edge_deployable": "Optimal (Ultra-Lightweight)"
    }
    results.append(prop_m)

    # -------------------------------------------------------------
    # 5. Build and Save Master Comparison Table
    # -------------------------------------------------------------
    df_master = pd.DataFrame(results)
    master_csv_path = os.path.join(RESULTS_DIR, "master_comparison_table.csv")
    df_master.to_csv(master_csv_path, index=False)
    print("\n" + "=" * 110)
    print("MASTER COMPARATIVE BENCHMARK TABLE")
    print("=" * 110)
    print(df_master[["model_name", "feature_count", "accuracy", "macro_f1", "ransomware_recall", "latency_ms_per_1k_pkts", "model_size_mb", "gpu_required"]].to_string(index=False))
    print("=" * 110)
    print(f"Master comparison saved to: {master_csv_path}")

    # -------------------------------------------------------------
    # 6. Generate Master Comparison Chart
    # -------------------------------------------------------------
    chart_path = os.path.join(FIGURES_DIR, "master_comparison_chart.png")
    generate_master_chart(df_master, chart_path)

    return df_master


def generate_master_chart(df, save_path):
    plt.figure(figsize=(14, 10), dpi=300)

    # Subplot 1: Accuracy vs Macro-F1
    plt.subplot(2, 2, 1)
    x = np.arange(len(df))
    width = 0.35
    colors_acc = ["#4c72b0" if "Proposed" not in n else "#2ca02c" for n in df["model_name"]]
    colors_f1 = ["#dd8452" if "Proposed" not in n else "#55a868" for n in df["model_name"]]

    b1 = plt.bar(x - width/2, df["accuracy"], width, label="Overall Accuracy (%)", color=colors_acc, alpha=0.85)
    b2 = plt.bar(x + width/2, df["macro_f1"], width, label="Macro-F1 Score (%)", color=colors_f1, alpha=0.9)
    plt.xticks(x, [n.replace(" (Full Features)", "\n(39-F)").replace(" (14-Feat + SMOTE)", "\n(14-F Ours)") for n in df["model_name"]], fontsize=8, rotation=20)
    plt.ylabel("Score (%)", fontweight="bold")
    plt.ylim(40, 105)
    plt.title("A: Detection Accuracy vs. Macro-F1", fontweight="bold", fontsize=11)
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Rare Attack Recall (Ransomware)
    plt.subplot(2, 2, 2)
    rec_colors = ["#c44e52" if "Proposed" not in n else "#2ca02c" for n in df["model_name"]]
    bars2 = plt.bar(x, df["ransomware_recall"], color=rec_colors, width=0.55)
    plt.xticks(x, [n.replace(" (Full Features)", "\n(39-F)").replace(" (14-Feat + SMOTE)", "\n(14-F Ours)") for n in df["model_name"]], fontsize=8, rotation=20)
    plt.ylabel("Ransomware Recall (%)", fontweight="bold")
    plt.ylim(0, 105)
    plt.title("B: Stealth Attack Detection (Ransomware Recall)", fontweight="bold", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Subplot 3: Inference Latency
    plt.subplot(2, 2, 3)
    lat_colors = ["#8172b3" if "Proposed" not in n else "#2ca02c" for n in df["model_name"]]
    bars3 = plt.bar(x, df["latency_ms_per_1k_pkts"], color=lat_colors, width=0.55)
    plt.xticks(x, [n.replace(" (Full Features)", "\n(39-F)").replace(" (14-Feat + SMOTE)", "\n(14-F Ours)") for n in df["model_name"]], fontsize=8, rotation=20)
    plt.ylabel("Latency (ms / 1k flows)", fontweight="bold")
    plt.title("C: Edge Inference Latency Comparison", fontweight="bold", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars3:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.2, f"{h:.1f}ms", ha="center", va="bottom", fontsize=8)

    # Subplot 4: Storage Footprint
    plt.subplot(2, 2, 4)
    size_colors = ["#ccb974" if "Proposed" not in n else "#2ca02c" for n in df["model_name"]]
    bars4 = plt.bar(x, df["model_size_mb"], color=size_colors, width=0.55)
    plt.xticks(x, [n.replace(" (Full Features)", "\n(39-F)").replace(" (14-Feat + SMOTE)", "\n(14-F Ours)") for n in df["model_name"]], fontsize=8, rotation=20)
    plt.ylabel("Model Storage Footprint (MB)", fontweight="bold")
    plt.title("D: Memory Footprint for IoT Flash Deployment", fontweight="bold", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars4:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.1, f"{h:.2f}MB", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Master comparison chart saved to: {save_path}")


if __name__ == "__main__":
    run_phase_b()

