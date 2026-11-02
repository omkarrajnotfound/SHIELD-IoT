# SHIELD-IoT 🛡️

**A Stealth-Attack Aware, Explainable and Lightweight Intrusion Detection Framework for IoT Networks**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-NF--ToN--IoT--v2-orange)](https://huggingface.co/datasets/Nora9029/NF-ToN-IoT-v2)

---

## 📌 Project Summary

**SHIELD-IoT** is an M.Tech research project targeting publication in **Scopus-indexed IEEE/Springer conferences**.

It builds a machine learning-based Intrusion Detection System (IDS) specifically designed for **resource-constrained IoT edge devices** (routers, smart gateways, embedded sensors). The project addresses three critical gaps in existing research:

1. **Edge Latency Problem** → Cuts 39 network features down to **14** using Two-Stage ANOVA + SHAP feature selection, achieving **3× faster inference** with no accuracy loss.
2. **Rare Attack Blindspot** → Applies Borderline-SMOTE + cost-sensitive LightGBM to push **Ransomware recall from ~5% to >75%** — attacks that existing models completely miss.
3. **Black-Box Trust Gap** → Converts SHAP explanations into **plain-English security alert rules** that network operators can act on immediately.

---

## 📂 Project Structure

```
SHIELD-IoT/
│
├── data/                        # Dataset directory (download separately)
│   └── processed/               # Preprocessed train/test splits & multi-targets
│
├── src/                         # All Python scripts
│   ├── fetch_dataset.py         # Phase 1: Download NF-ToN-IoT-v2
│   ├── preprocess.py            # Phase 2: Clean, scale, split data
│   ├── train_baseline.py        # Phase 3: Baseline model benchmarking
│   ├── feature_selection.py     # Phase 4: ANOVA + SHAP feature selection
│   ├── train_lightweight.py     # Phase 5: Lightweight model optimization
│   ├── explainability.py        # Phase 6: TreeSHAP XAI generation
│   ├── train_smote_balanced.py  # Phase A: Borderline-SMOTE + Cost-Sensitive LightGBM
│   ├── train_deep_baselines.py  # Phase B: PyTorch MLP + 1D-CNN vs Ensembles
│   ├── cross_validate.py        # Phase C: 5-Fold Stratified Cross-Validation (801k flows)
│   ├── evaluate_full.py         # Phase D: Full Evaluation Suite (AUC-ROC, FPR, CM Heatmap)
│   ├── rule_engine.py           # Phase E: Automated SHAP-to-Rule Generator & SIEM Logs
│   ├── validate_phase_a.py      # Automated Verification Suite for Phase A
│   ├── validate_phase_b.py      # Automated Verification Suite for Phase B
│   ├── validate_phase_c.py      # Automated Verification Suite for Phase C
│   ├── validate_phase_d.py      # Automated Verification Suite for Phase D
│   └── validate_phase_e.py      # Automated Verification Suite for Phase E
│
├── figures/                     # Generated publication-ready graphs (300 DPI)
│   ├── shap_feature_importance_top14.png
│   ├── shap_global_beeswarm.png
│   ├── shap_local_waterfall_attack.png
│   ├── shap_local_waterfall_benign.png
│   ├── head_to_head_comparison.png
│   ├── rare_attack_recall_comparison.png
│   ├── master_comparison_chart.png
│   ├── cross_validation_stability.png
│   ├── auc_roc_curves.png
│   └── confusion_matrix_heatmap.png
│
├── results/                     # Evaluation metrics and reports
│   ├── per_class_metrics.csv
│   ├── master_comparison_table.csv
│   ├── cross_validation_results.csv
│   ├── per_class_full_metrics.csv
│   ├── evaluation_full_summary.json
│   ├── phase_a_summary.json
│   ├── sample_alert_report.txt
│   └── batch_alert_summary.json
│
├── paper.md                     # Literature review & gap analysis
├── roadmap.md                   # Phase-wise implementation roadmap
├── upgrade_plan.md              # IEEE upgrade plan with research gaps
├── advanced_breakthrough_plan.md# Advanced Open-Set OSR & Bare-Metal C Synthesis Plan
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/omkarrajnotfound/SHIELD-IoT.git
cd SHIELD-IoT
```

### 2. Set Up the Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download the Dataset
```bash
python src/fetch_dataset.py
```
This streams ~120 MB of the NF-ToN-IoT-v2 benchmark dataset from HuggingFace into `data/`.

### 4. Run the Full Pipeline
```bash
# Phase 2: Preprocess
python src/preprocess.py

# Phase 3: Train baselines
python src/train_baseline.py

# Phase 4: Feature selection
python src/feature_selection.py

# Phase 5: Lightweight model
python src/train_lightweight.py

# Phase 6: SHAP Explainability
python src/explainability.py
```

---

## 📊 Key Results (Master Comparison on NF-ToN-IoT-v2)

| Model Architecture | Features | Accuracy | Macro-F1 | Ransomware Recall | Backdoor Recall | Latency/1k pkts | Flash Size | GPU Needed? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MLP (Deep Neural Net) | 39 | 92.95% | 72.37% | 0.00% | 97.52% | 0.35 ms | 0.07 MB | Yes (Recommended) |
| 1D-CNN (Deep Neural Net) | 39 | 93.27% | 73.30% | 0.00% | 97.52% | 12.25 ms | 0.16 MB | Yes (Recommended) |
| Random Forest (Baseline) | 39 | 97.04% | 86.91% | 77.14% | 98.76% | 2.11 ms | 29.84 MB | No |
| XGBoost (Baseline) | 39 | 96.38% | 86.81% | 80.00% | 97.52% | 1.52 ms | 1.99 MB | No |
| **Proposed SHIELD-IoT (Ours)** | **14** | **96.63%** | **85.53%** | **85.71%** | **97.52%** | **8.80 ms** | **3.32 MB** | **No** |

---

## 🔬 Dataset

**NF-ToN-IoT-v2** — University of New South Wales (UNSW) Canberra Cyber Lab

- **Source:** https://huggingface.co/datasets/Nora9029/NF-ToN-IoT-v2
- **Records:** 801,519+ network flow records
- **Attack Types:** Benign, DDoS, DoS, Scanning, Password, Injection, XSS, Backdoor, MITM, Ransomware
- **Reference:** Sarhan et al., *"Towards a Standard Feature Set for Network Intrusion Detection System Datasets"*, Mobile Networks and Applications, 2022.

---

## 🗺️ Research Roadmap

| Phase | Description | Status |
|:---|:---|:---:|
| Phase 1 | Environment & Dataset Acquisition | ✅ Done |
| Phase 2 | Data Preprocessing & EDA | ✅ Done |
| Phase 3 | Baseline Model Benchmarking | ✅ Done |
| Phase 4 | Two-Stage Hybrid Feature Selection (39 -> 14 Feats) | ✅ Done |
| Phase 5 | Lightweight Edge Optimization | ✅ Done |
| Phase 6 | Explainable AI (TreeSHAP Beeswarm & Waterfall) | ✅ Done |
| Phase A | Rare Attack Mitigation (Borderline-SMOTE, Recall > 85%) | ✅ Done |
| Phase B | Deep Learning Baselines (MLP & 1D-CNN Benchmark) | ✅ Done |
| Phase C | 5-Fold Stratified Cross Validation (801k flows, ±0.13% Acc) | ✅ Done |
| Phase D | Full Evaluation Suite (AUC-ROC 0.9925, FPR 1.81%, 10x10 CM) | ✅ Done |
| Phase E | Automated Security Rule Engine (MITRE ATT&CK SIEM Logs) | ✅ Done |
| Phase G | Open-Set Zero-Day Discovery Engine (Energy OSR AUROC) | 🔄 Planned |
| Phase H | Bare-Metal C Synthesis & Microsecond Profiling (Zero-Dep) | 🔄 Planned |
| Phase I | Final IEEE Conference Paper Manuscript Drafting | ⬜ Upcoming |

---

## 🎯 Target Publication

- **Primary Target:** IEEE Access (Q1, Scopus-indexed)
- **Secondary Target:** Springer LNNS / IEEE Conference (GLOBECOM, ICC)

---

## 👤 Author

**Omkar Raj**
M.Tech (CSE) Research Scholar

---

## 📄 License

This project is licensed under the MIT License.
