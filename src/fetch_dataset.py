"""
Dataset Acquisition & Exploration for NF-ToN-IoT-v2
"""
import os
import sys
import requests
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

CSV_FILE = "NF-ToN-IoT-v2-train.csv"
SAMPLE_CSV_PATH = os.path.join(DATA_DIR, "nf_ton_iot_sample.csv")
DOWNLOAD_URL = f"https://huggingface.co/datasets/Nora9029/NF-ToN-IoT-v2/resolve/main/{CSV_FILE}"

def download_sample_dataset(target_mb=120):
    if os.path.exists(SAMPLE_CSV_PATH) and os.path.getsize(SAMPLE_CSV_PATH) > 10 * 1024 * 1024:
        print(f"Sample dataset already exists: {SAMPLE_CSV_PATH} ({os.path.getsize(SAMPLE_CSV_PATH)/(1024*1024):.2f} MB)")
        return SAMPLE_CSV_PATH

    print(f"Connecting to: {DOWNLOAD_URL}")
    print(f"Streaming target size: ~{target_mb} MB to {SAMPLE_CSV_PATH}...")
    
    response = requests.get(DOWNLOAD_URL, stream=True, timeout=60)
    response.raise_for_status()
    
    target_bytes = target_mb * 1024 * 1024
    downloaded = 0
    block_size = 1024 * 1024 # 1MB

    with open(SAMPLE_CSV_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                pct = min(100.0, (downloaded / target_bytes) * 100)
                sys.stdout.write(f"\rDownloaded: {downloaded/(1024*1024):.1f} MB / {target_mb:.1f} MB ({pct:.1f}%)")
                sys.stdout.flush()
                if downloaded >= target_bytes:
                    print(f"\nTarget sample size (~{target_mb} MB) successfully reached.")
                    break

    # Clean the last incomplete line if any
    clean_last_line(SAMPLE_CSV_PATH)
    print(f"Sample dataset ready at: {SAMPLE_CSV_PATH}")
    return SAMPLE_CSV_PATH

def clean_last_line(filepath):
    """Ensure the CSV does not end on a truncated line"""
    with open(filepath, "rb+") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        # Read last 4KB to find the last newline
        seek_back = min(size, 4096)
        f.seek(-seek_back, os.SEEK_END)
        tail = f.read(seek_back)
        last_nl = tail.rfind(b"\n")
        if last_nl != -1 and last_nl < len(tail) - 1:
            cut_pos = size - (len(tail) - last_nl - 1)
            f.seek(cut_pos)
            f.truncate()
            print("Truncated last partial line for clean CSV parsing.")

def inspect_dataset(csv_path):
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA) - NF-ToN-IoT-v2")
    print("=" * 60)
    
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Total Rows Loaded:    {len(df):,}")
    print(f"Total Columns:        {df.shape[1]}")
    print(f"Memory Footprint:     {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    
    print("\nFeature Columns Preview (First 15):")
    for i, col in enumerate(df.columns[:15]):
        print(f"  {i+1}. {col} ({df[col].dtype})")
    if len(df.columns) > 15:
        print(f"  ... and {len(df.columns) - 15} more features.")

    # Target label inspection
    print("\nTarget Label Distribution:")
    for target_col in ["Attack", "Label", "label", "attack"]:
        if target_col in df.columns:
            print(f"\nColumn '{target_col}' Value Counts:")
            counts = df[target_col].value_counts()
            for k, v in counts.items():
                print(f"  - {k}: {v:,} ({v / len(df) * 100:.2f}%)")

    # Missing value check
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) == 0:
        print("\nMissing Values: None found! (Dataset is clean)")
    else:
        print(f"\nColumns with missing values: {len(null_cols)}")
        for col, cnt in null_cols.items():
            print(f"  - {col}: {cnt} missing")

    # Save a quick summary stats file
    summary_path = os.path.join(os.path.dirname(csv_path), "dataset_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"NF-ToN-IoT Dataset Summary\n")
        f.write(f"Total Records: {len(df):,}\n")
        f.write(f"Total Features: {df.shape[1]}\n\n")
        f.write("Columns:\n")
        for c in df.columns:
            f.write(f"  - {c} ({df[c].dtype})\n")
    print(f"\nDetailed schema saved to: {summary_path}")

if __name__ == "__main__":
    sample_path = download_sample_dataset(target_mb=120)
    inspect_dataset(sample_path)
