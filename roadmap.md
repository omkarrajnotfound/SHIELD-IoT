# Research Paper Roadmap
## A Lightweight and Explainable Machine Learning Framework for Network Intrusion Detection in IoT Environments

---

## 1. Project Overview & Target Objectives

* **Domain:** Cybersecurity, Internet of Things (IoT), and Explainable AI (XAI).
* **Research Core:**
  1. **Dimensionality & Latency Reduction:** Reduce feature space from 39 network flow features down to **14 critical features** using a two-stage hybrid selection pipeline (ANOVA Filter + TreeSHAP Attribution).
  2. **High Detection Performance:** Maintain **$>98.5\%$ detection accuracy** ($98.78\%$ XGBoost, $99.24\%$ Random Forest) across benchmark attacks.
  3. **Edge Feasibility:** Achieve inference latency down to **$0.33\text{ ms}$ per 1k packets** and compact model footprints ($<0.3\text{ MB}$).
  4. **Interpretability (XAI):** Deploy **TreeSHAP** to deliver human-auditable, packet-level explanations and automated alert rules for security operators.
* **Target Publication:** Scopus-indexed IEEE Conference / Springer LNNS / IEEE Access.
* **Status:** **Experimental Core Completed (Phases 1 to 6 Done). Ready for Paper Drafting (Phase 7).**

---

## 2. End-to-End Implementation Workflow

```
[Phase 1: Environment & Dataset Setup]                 --> [COMPLETED]
               ↓
[Phase 2: Preprocessing & Exploratory Analysis]         --> [COMPLETED]
               ↓
[Phase 3: Baseline Reproduction (Full Feature Set)]     --> [COMPLETED]
               ↓
[Phase 4: Two-Stage Feature Selection (39 -> 14 Feat)]  --> [COMPLETED]
               ↓
[Phase 5: Lightweight Classifier Training & Benchmark] --> [COMPLETED]
               ↓
[Phase 6: Explainable AI (SHAP Global & Local Plots)]   --> [COMPLETED]
               ↓
[Phase 7: Result Compilation & Paper Writing (IEEE)]    --> [CURRENT STAGE]
               ↓
[Phase 8: Guide Review & Camera-Ready Submission]       --> [UPCOMING]
```

---

## 3. Phase-by-Phase Roadmap & Milestones

### Phase 1: Environment & Dataset Acquisition (COMPLETED)
* [x] **Python Environment Setup:** Clean Python 3.12 virtual environment at `venv/` with all dependencies (`pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `shap`, `pyarrow`, `matplotlib`).
* [x] **Benchmark Dataset Acquisition:** Downloaded and verified **NF-ToN-IoT-v2** benchmark dataset (~801,519 network flow records).
* [x] **Exploratory Data Analysis:** Analyzed 10 classes (`Benign`, `scanning`, `xss`, `ddos`, `password`, `dos`, `injection`, `backdoor`, `mitm`, `ransomware`).

---

### Phase 2: Preprocessing & Numerical Sanitization (COMPLETED)
* [x] **Data Sanitization:** Dropped non-generalizable network identifiers (`IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, `L4_DST_PORT`) to eliminate host memorization.
* [x] **Stratified Train-Test Split:** Created 80/20 stratified split ($641,215$ train / $160,304$ test flows).
* [x] **Normalization:** Applied `RobustScaler` with float32 boundary protections to handle heavy-tailed network bursts.

---

### Phase 3: Baseline Model Benchmarking (Full Features) (COMPLETED)
* [x] **Trained Classifiers on all 39 Features:** Random Forest ($99.21\%$), XGBoost ($98.75\%$), LightGBM ($98.70\%$).
* [x] **Logged Baselines:** Recorded training time, inference latency, memory size, and saved metrics to `results/baseline_full_feature_metrics.csv`.

---

### Phase 4: Two-Stage Hybrid Feature Selection Pipeline (COMPLETED)
* [x] **Stage 1 (Statistical Filter):** ANOVA F-scores pruned 39 features down to 22 candidate attributes.
* [x] **Stage 2 (TreeSHAP Attribution Wrapper):** Calculated exact game-theoretic Shapley contributions and extracted the **Top 14 Critical Features**:
  1. `MAX_IP_PKT_LEN`
  2. `TCP_WIN_MAX_IN`
  3. `L7_PROTO`
  4. `DST_TO_SRC_AVG_THROUGHPUT`
  5. `SRC_TO_DST_AVG_THROUGHPUT`
  6. `FLOW_DURATION_MILLISECONDS`
  7. `MIN_IP_PKT_LEN`
  8. `DNS_TTL_ANSWER`
  9. `TCP_FLAGS`
  10. `SERVER_TCP_FLAGS`
  11. `TCP_WIN_MAX_OUT`
  12. `SHORTEST_FLOW_PKT`
  13. `CLIENT_TCP_FLAGS`
  14. `MIN_TTL`
* [x] **Generated Visualization:** Saved `figures/shap_feature_importance_top14.png`.

---

### Phase 5: Lightweight Edge Optimization & Benchmarking (COMPLETED)
* [x] **Retrained Models on Top 14 Features:**
  * Random Forest (14-F): **$99.24\%$ Accuracy** ($+0.03\%$ boost).
  * XGBoost (14-F): **$98.78\%$ Accuracy** ($+0.03\%$ boost, **$0.33\text{ ms}$ latency**).
  * LightGBM (14-F): **$98.71\%$ Accuracy** ($+0.01\%$ boost, **$0.26\text{ MB}$ model size**).
* [x] **Generated Direct Comparison:** Saved `results/comparative_benchmark_results.csv` and `figures/head_to_head_comparison.png`.

---

### Phase 6: Explainable AI (XAI) Implementation (COMPLETED)
* [x] **Global Interpretability:** Generated SHAP beeswarm plot `figures/shap_global_beeswarm.png`.
* [x] **Local Threat Interpretability:** Generated attack waterfall chart `figures/shap_local_waterfall_attack.png`.
* [x] **Local Benign Interpretability:** Generated benign waterfall chart `figures/shap_local_waterfall_benign.png`.
* [x] **Automated Operator Rule Generation:** Created JSON alert rationale generator (`results/sample_alert_explanation_rule.json`).

---

### Phase 7: Research Paper Drafting (IEEE Format) (NEXT)
* [ ] Setup IEEE 2-column LaTeX / Word template.
* [ ] Draft Abstract, Introduction, and Literature Review (citing 2026 IEEE Access benchmarks).
* [ ] Format Proposed Methodology with system architecture flowcharts.
* [ ] Insert Comparative Experimental Tables and SHAP figures.
* [ ] Draft Discussion, Limitations, and Conclusion.

---

### Phase 8: Guide Review & Submission
* [ ] Present working implementation and draft paper to M.Tech guide.
* [ ] Plagiarism check ($<15\%$).
* [ ] Submit to targeted IEEE/Springer conference.
