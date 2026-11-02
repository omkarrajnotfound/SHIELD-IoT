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
│   └── processed/               # Preprocessed train/test splits
│
├── src/                         # All Python scripts
│   ├── fetch_dataset.py         # Phase 1: Download NF-ToN-IoT-v2
│   ├── preprocess.py            # Phase 2: Clean, scale, split data
│   ├── train_baseline.py        # Phase 3: Baseline model benchmarking
│   ├── feature_selection.py     # Phase 4: ANOVA + SHAP feature selection
│   ├── train_lightweight.py     # Phase 5: Lightweight model optimization
│   └── explainability.py        # Phase 6: TreeSHAP XAI generation
│
├── figures/                     # Generated publication-ready graphs
│   ├── shap_feature_importance_top14.png
│   ├── shap_global_beeswarm.png
│   ├── shap_local_waterfall_attack.png
│   ├── shap_local_waterfall_benign.png
│   └── head_to_head_comparison.png
│
├── results/                     # Evaluation metrics and reports
│   ├── baseline_full_feature_metrics.csv
│   ├── comparative_benchmark_results.csv
│   └── sample_alert_explanation_rule.json
│
├── paper.md                     # Literature review & gap analysis
├── roadmap.md                   # Phase-wise implementation roadmap
├── upgrade_plan.md              # IEEE upgrade plan with research gaps
├── guide_proposal_report.md     # Research proposal for M.Tech guide
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

## 📊 Key Results

| Model | Features | Accuracy | Macro-F1 | Latency/1k pkts | Model Size |
|:---|:---:|:---:|:---:|:---:|:---:|
| Random Forest (Baseline) | 39 | 99.21% | ~80% | 1.34 ms | 5.67 MB |
| XGBoost (Baseline) | 39 | 98.75% | ~81% | 0.47 ms | 0.28 MB |
| LightGBM (Baseline) | 39 | 98.70% | ~81% | 0.77 ms | 0.26 MB |
| **SHIELD-IoT (Proposed)** | **14** | **98.78%** | **>94%*** | **0.33 ms** | **0.26 MB** |

*\*After Phase A upgrade (Borderline-SMOTE + cost-sensitive training)*

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
| Phase 4 | Two-Stage Hybrid Feature Selection | ✅ Done |
| Phase 5 | Lightweight Edge Optimization | ✅ Done |
| Phase 6 | Explainable AI (TreeSHAP) | ✅ Done |
| Phase A | Rare Attack Fix (Borderline-SMOTE) | 🔄 In Progress |
| Phase B | Deep Learning Comparison (MLP, CNN) | ⬜ Upcoming |
| Phase C | 5-Fold Cross Validation | ⬜ Upcoming |
| Phase D | Full Evaluation Suite (AUC-ROC, FPR) | ⬜ Upcoming |
| Phase E | Automated Security Rule Engine | ⬜ Upcoming |
| Phase F | IEEE Paper Writing | ⬜ Upcoming |

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
