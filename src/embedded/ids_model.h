/*
 * ==============================================================================
 * SHIELD-IoT: Standalone Bare-Metal Intrusion Detection System
 * Auto-generated ANSI C99 Model Interface (Zero Dependencies, Zero Allocations)
 *
 * Target: Resource-Constrained IoT Edge Nodes (ESP32, STM32, ARM Cortex-M)
 * Number of Features: 14
 * Number of Classes:  10
 * Number of Trees:    1000
 * ==============================================================================
 */

#ifndef IDS_MODEL_H
#define IDS_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

#define IDS_NUM_FEATURES 14
#define IDS_NUM_CLASSES  10
#define IDS_NUM_TREES    1000

/* Class Index Definitions */
#define IDS_CLASS_BENIGN 0
#define IDS_CLASS_BACKDOOR 1
#define IDS_CLASS_DDOS 2
#define IDS_CLASS_DOS 3
#define IDS_CLASS_INJECTION 4
#define IDS_CLASS_MITM 5
#define IDS_CLASS_PASSWORD 6
#define IDS_CLASS_RANSOMWARE 7
#define IDS_CLASS_SCANNING 8
#define IDS_CLASS_XSS 9

/*
 * Evaluates the 14 network flow features and returns the predicted class index [0..9].
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
}
#endif

#endif /* IDS_MODEL_H */
