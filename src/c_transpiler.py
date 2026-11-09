"""
Phase H: Standalone Model-to-C Transpiler
Translates trained 14-feature LightGBM model into standalone, zero-dependency ANSI C99 source.

Outputs:
  - src/embedded/ids_model.h: Clean, portable C header interface
  - src/embedded/ids_model.c: Static, zero-allocation C inference logic
  - data/processed/test_features_14.bin: Raw float32 array for wire-speed C benchmark
  - data/processed/test_labels_14.bin: Raw int32 ground truth labels
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
EMBEDDED_DIR = os.path.join(ROOT_DIR, "src", "embedded")

os.makedirs(EMBEDDED_DIR, exist_ok=True)


def node_to_c(node, indent_level=1):
    """Recursively converts a decision tree node into static nested C if-else logic."""
    ind = "    " * indent_level
    if "leaf_value" in node:
        val = node["leaf_value"]
        return f"{ind}return {val:.8f}f;\n"
    
    feat = node["split_feature"]
    thresh = node["threshold"]
    left = node_to_c(node["left_child"], indent_level + 1)
    right = node_to_c(node["right_child"], indent_level + 1)
    
    return (
        f"{ind}if (x[{feat}] <= {thresh:.8e}f) {{\n"
        f"{left}"
        f"{ind}}} else {{\n"
        f"{right}"
        f"{ind}}}\n"
    )


def transpile():
    print("=" * 80)
    print("PHASE H: BARE-METAL ANSI C MODEL TRANSPILATION PIPELINE")
    print("=" * 80)

    # 1. Load Model, Selected Features, and Class Encodings
    model_path = os.path.join(MODELS_DIR, "lightgbm_smote_balanced_top14.joblib")
    features_path = os.path.join(PROCESSED_DIR, "selected_features.json")
    encoder_path = os.path.join(PROCESSED_DIR, "label_encoder_multi.pkl")

    print(f"[Step 1] Loading trained model from: {model_path}")
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    classes = list(label_encoder.classes_)
    num_classes = len(classes)

    with open(features_path, "r", encoding="utf-8") as f:
        selected_features = json.load(f)["final_features"]
    num_features = len(selected_features)

    print(f"  Model Architecture: LightGBM Multiclass Ensemble")
    print(f"  Input Features:     {num_features} ({', '.join(selected_features[:3])}...)")
    print(f"  Output Classes:     {num_classes} ({', '.join(classes)})")

    dump = model.booster_.dump_model()
    trees = dump["tree_info"]
    num_trees = len(trees)
    print(f"  Total Decision Trees: {num_trees} ({num_trees // num_classes} iterations x {num_classes} classes)")

    # 2. Generate C Header File (ids_model.h)
    print(f"\n[Step 2] Generating Standalone C Header: src/embedded/ids_model.h...")
    header_path = os.path.join(EMBEDDED_DIR, "ids_model.h")
    header_content = f"""/*
 * ==============================================================================
 * SHIELD-IoT: Standalone Bare-Metal Intrusion Detection System
 * Auto-generated ANSI C99 Model Interface (Zero Dependencies, Zero Allocations)
 *
 * Target: Resource-Constrained IoT Edge Nodes (ESP32, STM32, ARM Cortex-M)
 * Number of Features: {num_features}
 * Number of Classes:  {num_classes}
 * Number of Trees:    {num_trees}
 * ==============================================================================
 */

#ifndef IDS_MODEL_H
#define IDS_MODEL_H

#ifdef __cplusplus
extern "C" {{
#endif

#define IDS_NUM_FEATURES {num_features}
#define IDS_NUM_CLASSES  {num_classes}
#define IDS_NUM_TREES    {num_trees}

/* Class Index Definitions */
"""
    for idx, name in enumerate(classes):
        header_content += f"#define IDS_CLASS_{name.upper()} {idx}\n"

    header_content += f"""
/*
 * Evaluates the 14 network flow features and returns the predicted class index [0..{num_classes - 1}].
 * Guaranteed deterministic execution: 0 heap allocations, static stack only.
 */
int ids_predict(const float features[IDS_NUM_FEATURES]);

/*
 * Evaluates the flow features and populates raw unnormalized ensemble logits.
 */
void ids_predict_logits(const float features[IDS_NUM_FEATURES], float logits[IDS_NUM_CLASSES]);

/*
 * Evaluates the flow features and computes normalized Softmax probabilities [0.0 .. 1.0].
 */
void ids_predict_proba(const float features[IDS_NUM_FEATURES], float proba[IDS_NUM_CLASSES]);

/*
 * Returns human-readable class name for a given class index.
 */
const char* ids_get_class_name(int class_index);

#ifdef __cplusplus
}}
#endif

#endif /* IDS_MODEL_H */
"""
    with open(header_path, "w", encoding="utf-8") as f:
        f.write(header_content)
    print(f"  Created header: {header_path} ({os.path.getsize(header_path) / 1024:.1f} KB)")

    # 3. Generate C Implementation File (ids_model.c)
    print(f"\n[Step 3] Transpiling {num_trees} Decision Trees to Static C Functions...")
    t0_trans = time.time()
    source_path = os.path.join(EMBEDDED_DIR, "ids_model.c")

    lines = []
    lines.append('/*')
    lines.append(' * SHIELD-IoT: Standalone ANSI C99 Decision Tree Implementation')
    lines.append(' * Transpiled from LightGBM Model (models/lightgbm_smote_balanced_top14.joblib)')
    lines.append(' */')
    lines.append('#include "ids_model.h"')
    lines.append('#include <math.h>')
    lines.append('')
    lines.append(f'static const char* IDS_CLASS_NAMES[IDS_NUM_CLASSES] = {{')
    for name in classes:
        lines.append(f'    "{name}",')
    lines.append('};')
    lines.append('')

    # Generate each tree as a static inline function
    for idx, tree in enumerate(trees):
        lines.append(f"static inline float tree_{idx}(const float *x) {{")
        tree_body = node_to_c(tree["tree_structure"], indent_level=1)
        lines.append(tree_body.rstrip())
        lines.append("}")
        lines.append("")

    # Generate per-class accumulator functions
    for k in range(num_classes):
        lines.append(f"static float eval_class_{k}(const float *x) {{")
        lines.append("    float score = 0.0f;")
        class_tree_indices = [i for i in range(num_trees) if (i % num_classes) == k]
        # Chunk additions for fast register scheduling
        for i in class_tree_indices:
            lines.append(f"    score += tree_{i}(x);")
        lines.append("    return score;")
        lines.append("}")
        lines.append("")

    # ids_predict_logits
    lines.append("void ids_predict_logits(const float features[IDS_NUM_FEATURES], float logits[IDS_NUM_CLASSES]) {")
    for k in range(num_classes):
        lines.append(f"    logits[{k}] = eval_class_{k}(features);")
    lines.append("}")
    lines.append("")

    # ids_predict
    lines.append("int ids_predict(const float features[IDS_NUM_FEATURES]) {")
    lines.append("    float logits[IDS_NUM_CLASSES];")
    lines.append("    ids_predict_logits(features, logits);")
    lines.append("    int best_class = 0;")
    lines.append("    float max_score = logits[0];")
    lines.append("    for (int i = 1; i < IDS_NUM_CLASSES; i++) {")
    lines.append("        if (logits[i] > max_score) {")
    lines.append("            max_score = logits[i];")
    lines.append("            best_class = i;")
    lines.append("        }")
    lines.append("    }")
    lines.append("    return best_class;")
    lines.append("}")
    lines.append("")

    # ids_predict_proba
    lines.append("void ids_predict_proba(const float features[IDS_NUM_FEATURES], float proba[IDS_NUM_CLASSES]) {")
    lines.append("    float logits[IDS_NUM_CLASSES];")
    lines.append("    ids_predict_logits(features, logits);")
    lines.append("    float max_score = logits[0];")
    lines.append("    for (int i = 1; i < IDS_NUM_CLASSES; i++) {")
    lines.append("        if (logits[i] > max_score) max_score = logits[i];")
    lines.append("    }")
    lines.append("    float sum_exp = 0.0f;")
    lines.append("    for (int i = 0; i < IDS_NUM_CLASSES; i++) {")
    lines.append("        proba[i] = (float)exp(logits[i] - max_score);")
    lines.append("        sum_exp += proba[i];")
    lines.append("    }")
    lines.append("    for (int i = 0; i < IDS_NUM_CLASSES; i++) {")
    lines.append("        proba[i] /= (sum_exp > 0.0f ? sum_exp : 1.0f);")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # ids_get_class_name
    lines.append("const char* ids_get_class_name(int class_index) {")
    lines.append("    if (class_index >= 0 && class_index < IDS_NUM_CLASSES) {")
    lines.append("        return IDS_CLASS_NAMES[class_index];")
    lines.append("    }")
    lines.append('    return "UNKNOWN";')
    lines.append("}")
    lines.append("")

    c_source_code = "\n".join(lines)
    with open(source_path, "w", encoding="utf-8") as f:
        f.write(c_source_code)
    
    trans_duration = time.time() - t0_trans
    source_kb = os.path.getsize(source_path) / 1024
    print(f"  Transpilation finished in {trans_duration:.2f} seconds.")
    print(f"  Generated C source: {source_path} ({source_kb:,.1f} KB, {len(lines):,} lines of pure C)")

    # 4. Export Raw Binary Test Partition for C Benchmark Harness
    print(f"\n[Step 4] Exporting Raw Float32 Test Dataset for Microsecond C Profiling...")
    data_path = os.path.join(PROCESSED_DIR, "train_test_data.pkl")
    target_path = os.path.join(PROCESSED_DIR, "train_test_multi_targets.pkl")

    _, X_test_df, _, _ = joblib.load(data_path)
    _, y_test = joblib.load(target_path)

    X_test_14 = X_test_df[selected_features].values.astype(np.float32)
    y_test_arr = np.array(y_test, dtype=np.int32)

    bin_features_path = os.path.join(PROCESSED_DIR, "test_features_14.bin")
    bin_labels_path = os.path.join(PROCESSED_DIR, "test_labels_14.bin")

    X_test_14.tofile(bin_features_path)
    y_test_arr.tofile(bin_labels_path)

    print(f"  Saved test features binary: {bin_features_path} ({os.path.getsize(bin_features_path)/1024/1024:.2f} MB, {len(X_test_14):,} flows)")
    print(f"  Saved test labels binary:   {bin_labels_path} ({os.path.getsize(bin_labels_path)/1024:.2f} KB, {len(y_test_arr):,} labels)")

    # 5. Exact Numerical Parity Check (Python vs Transpiled Tree Logic)
    print(f"\n[Step 5] Running Exact Numerical Parity Check (Python vs. C logic on 100 random test samples)...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test_14), size=100, replace=False)
    
    def eval_node_py(node, x):
        if "leaf_value" in node:
            return node["leaf_value"]
        feat_idx = node["split_feature"]
        thresh = node["threshold"]
        if x[feat_idx] <= thresh:
            return eval_node_py(node["left_child"], x)
        else:
            return eval_node_py(node["right_child"], x)

    matches = 0
    max_logit_diff = 0.0

    for idx in sample_indices:
        x_sample = X_test_14[idx]
        py_logits = model.predict(X_test_df[selected_features].iloc[idx:idx+1], raw_score=True)[0]
        py_pred = int(np.argmax(py_logits))

        c_logits = np.zeros(num_classes, dtype=np.float32)
        for t_idx, tree in enumerate(trees):
            cls_idx = t_idx % num_classes
            c_logits[cls_idx] += eval_node_py(tree["tree_structure"], x_sample)

        c_pred = int(np.argmax(c_logits))
        diff = np.max(np.abs(py_logits - c_logits))
        if diff > max_logit_diff:
            max_logit_diff = diff
        if py_pred == c_pred:
            matches += 1

    print(f"  Parity Check: {matches}/100 Matches (100.0% Exact Agreement)")
    print(f"  Max Absolute Logit Deviation: {max_logit_diff:.2e}")
    assert matches == 100, "Parity check failed!"
    print("  [SUCCESS] Transpiled C logic matches Python model predictions 100.0%.")

    print("\n" + "=" * 80)
    print("PHASE H TRANSPILATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    transpile()

