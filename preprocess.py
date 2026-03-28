"""
preprocess.py
Loads all CICIoT2023 CSV files, maps fine-grained labels to 8 macro categories,
cleans + scales features, and saves stratified 70/15/15 train/val/test splits.

Usage:
    python preprocess.py --data-dir <path-to-MERGED_CSV> [--max-rows-per-file 10000]
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# ── Label → macro-category mapping ──────────────────────────────────────────
LABEL_MAP = {
    # DDoS
    "DDOS-ACK_FRAGMENTATION":   "DDoS",
    "DDOS-HTTP_FLOOD":          "DDoS",
    "DDOS-ICMP_FLOOD":          "DDoS",
    "DDOS-ICMP_FRAGMENTATION":  "DDoS",
    "DDOS-PSHACK_FLOOD":        "DDoS",
    "DDOS-RSTFINFLOOD":         "DDoS",
    "DDOS-SLOWLORIS":           "DDoS",
    "DDOS-SYNONYMOUSIP_FLOOD":  "DDoS",
    "DDOS-SYN_FLOOD":           "DDoS",
    "DDOS-TCP_FLOOD":           "DDoS",
    "DDOS-UDP_FLOOD":           "DDoS",
    "DDOS-UDP_FRAGMENTATION":   "DDoS",
    # DoS
    "DOS-HTTP_FLOOD":           "DoS",
    "DOS-SYN_FLOOD":            "DoS",
    "DOS-TCP_FLOOD":            "DoS",
    "DOS-UDP_FLOOD":            "DoS",
    # Recon
    "RECON-HOSTDISCOVERY":      "Recon",
    "RECON-OSSCAN":             "Recon",
    "RECON-PINGSWEEP":          "Recon",
    "RECON-PORTSCAN":           "Recon",
    "VULNERABILITYSCAN":        "Recon",
    # Web-Based
    "XSS":                      "Web-Based",
    "COMMANDINJECTION":         "Web-Based",
    "SQLINJECTION":             "Web-Based",
    "BACKDOOR_MALWARE":         "Web-Based",
    "BROWSERHIJACKING":         "Web-Based",
    "UPLOADING_ATTACK":         "Web-Based",
    # Brute Force
    "DICTIONARYBRUTEFORCE":     "Brute Force",
    # Spoofing
    "DNS_SPOOFING":             "Spoofing",
    "MITM-ARPSPOOFING":         "Spoofing",
    # Mirai
    "MIRAI-GREETH_FLOOD":       "Mirai",
    "MIRAI-GREIP_FLOOD":        "Mirai",
    "MIRAI-UDPPLAIN":           "Mirai",
    # Benign
    "BENIGN":                   "Benign",
}

FEATURE_COLS = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number",
    "cwr_flag_number", "ack_count", "syn_count", "fin_count",
    "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH",
    "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv",
    "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size",
    "IAT", "Number", "Variance",
]


def load_data(data_dir: str, max_rows_per_file: int | None = None) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    print(f"Found {len(files)} CSV files.")

    chunks = []
    for i, f in enumerate(files, 1):
        try:
            df = pd.read_csv(f, nrows=max_rows_per_file)
            chunks.append(df)
            print(f"  [{i:02d}/{len(files)}] {os.path.basename(f):25s}  {len(df):>7,} rows")
        except Exception as e:
            print(f"  WARNING: skipping {f}: {e}")

    data = pd.concat(chunks, ignore_index=True)
    print(f"\nTotal rows loaded: {len(data):,}")
    return data


def preprocess(data_dir: str, out_dir: str, max_rows_per_file: int | None = None):
    os.makedirs(out_dir, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    df = load_data(data_dir, max_rows_per_file)

    # ── Map labels ───────────────────────────────────────────────────────────
    unknown = set(df["Label"].unique()) - set(LABEL_MAP)
    if unknown:
        print(f"WARNING: unknown labels dropped: {unknown}")
    df["Category"] = df["Label"].map(LABEL_MAP)
    df = df.dropna(subset=["Category"])
    print("\nCategory distribution:")
    print(df["Category"].value_counts().to_string())

    # ── Features ─────────────────────────────────────────────────────────────
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLS].copy()
    y_raw = df["Category"].values

    # Clean: replace inf → nan → median
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    medians = X.median()
    X.fillna(medians, inplace=True)

    # ── Encode labels ────────────────────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"\nClasses ({len(le.classes_)}): {list(le.classes_)}")

    # ── Split 70 / 15 / 15 ───────────────────────────────────────────────────
    X_arr = X.values.astype(np.float32)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_arr, y, test_size=0.15, random_state=42, stratify=y
    )
    val_ratio = 0.15 / 0.85  # 15% of original from the remaining 85%
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_ratio, random_state=42, stratify=y_tmp
    )
    print(f"\nSplit sizes -- train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}")

    # ── Scale (fit on train only) ────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # ── Save ─────────────────────────────────────────────────────────────────
    np.savez_compressed(
        os.path.join(out_dir, "splits.npz"),
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
    )
    with open(os.path.join(out_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSaved splits + encoder + scaler to '{out_dir}/'")
    return len(le.classes_), X_train.shape[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=r"C:\Users\danny\Downloads\MERGED_CSV\MERGED_CSV")
    parser.add_argument("--out-dir",  default="processed")
    parser.add_argument("--max-rows-per-file", type=int, default=None,
                        help="Cap rows read per CSV (None = all). Use e.g. 15000 for a quick run.")
    args = parser.parse_args()
    preprocess(args.data_dir, args.out_dir, args.max_rows_per_file)
