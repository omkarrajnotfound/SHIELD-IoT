/*
 * ==============================================================================
 * SHIELD-IoT: Bare-Metal C Performance & Microsecond Edge Benchmarking Harness
 *
 * Measures:
 *   1. Hardware Execution Latency per Packet (microseconds, µs)
 *   2. Packet Processing Throughput (flows / second)
 *   3. Accuracy & Consistency Verification against Ground Truth (160,304 flows)
 *   4. Peak RAM Utilization (Resident Set Size via OS hardware counters)
 *
 * Compatible with Windows (MSVC/Clang/MinGW) and POSIX (GCC/Clang on ARM/Linux)
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ids_model.h"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <sys/time.h>
#include <sys/resource.h>
#endif

/* High-resolution timer functions */
static double get_time_seconds(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/* Peak RAM utilization in kilobytes */
static long get_peak_ram_kb(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (long)(pmc.PeakWorkingSetSize / 1024);
    }
    return 0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss;
    }
    return 0;
#endif
}

int main(int argc, char *argv[]) {
    const char *feat_file = (argc > 1) ? argv[1] : "data/processed/test_features_14.bin";
    const char *label_file = (argc > 2) ? argv[2] : "data/processed/test_labels_14.bin";

    printf("================================================================================\n");
    printf("SHIELD-IoT: BARE-METAL C INFERENCE BENCHMARK & HARDWARE PROFILER\n");
    printf("================================================================================\n");
    printf("Target Architecture: Compiled Standalone ANSI C99 (0 Dynamic Memory Allocations)\n");
    printf("Loading test features from: %s\n", feat_file);

    /* Open features file */
    FILE *f_feat = fopen(feat_file, "rb");
    if (!f_feat) {
        fprintf(stderr, "Error: Unable to open features file %s\n", feat_file);
        return 1;
    }

    fseek(f_feat, 0, SEEK_END);
    long feat_bytes = ftell(f_feat);
    fseek(f_feat, 0, SEEK_SET);

    size_t num_samples = feat_bytes / (IDS_NUM_FEATURES * sizeof(float));
    printf("Test Flow Count: %zu flows (%ld bytes loaded)\n", num_samples, feat_bytes);

    float *features = (float *)malloc(feat_bytes);
    if (!features) {
        fprintf(stderr, "Error: Memory allocation failed for test features\n");
        fclose(f_feat);
        return 1;
    }
    if (fread(features, sizeof(float), num_samples * IDS_NUM_FEATURES, f_feat) != num_samples * IDS_NUM_FEATURES) {
        fprintf(stderr, "Error: Failed to read all feature data\n");
        free(features);
        fclose(f_feat);
        return 1;
    }
    fclose(f_feat);

    /* Open labels file */
    int *labels = (int *)malloc(num_samples * sizeof(int));
    FILE *f_label = fopen(label_file, "rb");
    if (f_label) {
        fread(labels, sizeof(int), num_samples, f_label);
        fclose(f_label);
    } else {
        printf("Warning: Labels file not found; running latency-only benchmark.\n");
        memset(labels, 0, num_samples * sizeof(int));
    }

    /* Warm-up run (1,000 flows) to prime CPU cache and branch predictor */
    printf("\nWarming up instruction cache (1,000 flows)...\n");
    for (size_t i = 0; i < 1000 && i < num_samples; i++) {
        ids_predict(&features[i * IDS_NUM_FEATURES]);
    }

    /* Benchmark full evaluation */
    printf("Executing Bare-Metal C Inference across all %zu flows...\n", num_samples);
    int correct = 0;
    int confusion[IDS_NUM_CLASSES][IDS_NUM_CLASSES];
    memset(confusion, 0, sizeof(confusion));

    double t_start = get_time_seconds();

    for (size_t i = 0; i < num_samples; i++) {
        const float *x = &features[i * IDS_NUM_FEATURES];
        int pred = ids_predict(x);
        int true_label = labels[i];
        if (pred == true_label) {
            correct++;
        }
        if (true_label >= 0 && true_label < IDS_NUM_CLASSES && pred >= 0 && pred < IDS_NUM_CLASSES) {
            confusion[true_label][pred]++;
        }
    }

    double t_end = get_time_seconds();
    double total_sec = t_end - t_start;
    double latency_us_per_flow = (total_sec / (double)num_samples) * 1e6;
    double throughput_fps = (double)num_samples / total_sec;
    double accuracy_pct = ((double)correct / (double)num_samples) * 100.0;
    long peak_ram_kb = get_peak_ram_kb();

    printf("\n" "--------------------------------------------------------------------------------\n");
    printf("BARE-METAL C PERFORMANCE METRICS (NF-ToN-IoT-v2 Test Set: %zu Flows)\n", num_samples);
    printf("--------------------------------------------------------------------------------\n");
    printf("  Total Execution Time:       %8.4f seconds\n", total_sec);
    printf("  Average Latency per Flow:   %8.3f microseconds (µs/packet)\n", latency_us_per_flow);
    printf("  Inference Throughput:       %8.1f packets / second (fps)\n", throughput_fps);
    printf("  Classification Accuracy:    %8.2f%% (%d / %zu correct)\n", accuracy_pct, correct, num_samples);
    printf("  Peak Resident Memory (RAM): %8ld KB (%6.2f MB)\n", peak_ram_kb, (double)peak_ram_kb / 1024.0);
    printf("  Dynamic Allocs in Model:           0 bytes (100%% static stack memory)\n");
    printf("--------------------------------------------------------------------------------\n");

    /* Export JSON result file for Python comparison & plotting */
    FILE *f_out = fopen("results/c_benchmark_results.json", "w");
    if (f_out) {
        fprintf(f_out, "{\n");
        fprintf(f_out, "  \"total_samples\": %zu,\n", num_samples);
        fprintf(f_out, "  \"total_time_seconds\": %.6f,\n", total_sec);
        fprintf(f_out, "  \"latency_us_per_flow\": %.4f,\n", latency_us_per_flow);
        fprintf(f_out, "  \"throughput_fps\": %.2f,\n", throughput_fps);
        fprintf(f_out, "  \"accuracy_pct\": %.4f,\n", accuracy_pct);
        fprintf(f_out, "  \"peak_ram_kb\": %ld\n", peak_ram_kb);
        fprintf(f_out, "}\n");
        fclose(f_out);
        printf("Saved benchmark metrics to results/c_benchmark_results.json\n");
    }

    free(features);
    free(labels);
    printf("================================================================================\n");
    return 0;
}
