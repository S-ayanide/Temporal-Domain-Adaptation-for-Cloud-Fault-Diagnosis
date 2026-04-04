"""
Step 0: Data Preparation — Temporal Windowed Sequences
=======================================================
Builds on the replicate/ preprocessing but adds TEMPORAL WINDOWING:
instead of treating each machine reading as an independent point,
readings are grouped by machine and ordered by timestamp into
overlapping windows of W timesteps.

Each window  →  shape (W, 6)   — the raw input to TA-DATL
Each window  →  shape (6,)     — mean-pooled, for flat baselines (DANN etc.)

This captures *how* resource metrics evolve before a fault, not just
their instantaneous values — the key insight of this research.

Outputs
-------
data/processed/source_temporal.npz   X:(N,W,F)  y:(N,)  labeled:(N,) machine:(N,)
data/processed/target_temporal.npz   same
data/processed/source_flat.parquet   X:(N,F)    y:(N,)  labeled:(N,)  ← for baselines
data/processed/target_flat.parquet   same
data/processed/meta.json
"""

import json
import tarfile
import urllib.request
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
RAW_DIR  = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
WINDOW_SIZE  = 20        # timesteps per sequence  (W)
WINDOW_STEP  = 5         # stride between windows  (overlap = W - step)
MIN_SEQ_LEN  = WINDOW_SIZE   # machines with fewer readings are dropped

FEATURE_COLS = [
    "cpu_util_percent",   # CPU usage
    "mem_util_percent",   # Memory
    "mem_gps",            # Memory bandwidth
    "net_in",             # Network in
    "net_out",            # Network out
    "disk_io_percent",    # Disk I/O
]

FAULT_NAMES = {
    0: "Normal",
    1: "CPU_Overload",
    2: "Memory_Leak",
    3: "Disk_IO_Fault",
    4: "Network_Fault",
    5: "Mixed_Fault",
}
N_CLASSES = len(FAULT_NAMES)

ALIBABA_BASE = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces"
COL_NAMES_USAGE = ["machine_id", "time_stamp", "cpu_util_percent",
                   "mem_util_percent", "mem_gps", "mkpi",
                   "net_in", "net_out", "disk_io_percent"]
COL_NAMES_META  = ["machine_id", "time_stamp", "failure_domain_1",
                   "failure_domain_2", "cpu_num", "mem_size", "status"]


# ── Download helpers ──────────────────────────────────────────────────────────

def _try_download(filename: str, dest: Path, timeout: int = 20) -> bool:
    url = f"{ALIBABA_BASE}/{filename}"
    print(f"  Trying {url} ...")
    try:
        urllib.request.urlopen(url, timeout=timeout).read(512)
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def load_machine_meta(raw_dir: Path) -> Optional[pd.DataFrame]:
    csv = raw_dir / "machine_meta.csv"
    if csv.exists():
        return pd.read_csv(csv, header=None, names=COL_NAMES_META)
    dest = raw_dir / "machine_meta.tar.gz"
    if not dest.exists():
        _try_download("machine_meta.tar.gz", dest)
    if dest.exists():
        try:
            with tarfile.open(dest, "r:gz") as tf:
                tf.extractall(raw_dir)
            f = list(raw_dir.glob("machine_meta.csv"))
            if f:
                return pd.read_csv(f[0], header=None, names=COL_NAMES_META)
        except Exception as e:
            print(f"  Cannot extract machine_meta: {e}")
    return None


def load_machine_usage(raw_dir: Path,
                       target_rows: int = 500_000,
                       seed: int = 42) -> Optional[pd.DataFrame]:
    """Random sample spread across the full CSV (not sequential first N rows)."""
    def _sample(path: Path) -> pd.DataFrame:
        file_bytes  = path.stat().st_size
        est_total   = max(file_bytes // 70, target_rows * 2)
        prob        = min(1.0, (target_rows * 1.5) / est_total)
        rng_s       = np.random.default_rng(seed)
        print(f"  Sampling ~{target_rows:,} rows (p≈{prob:.4f}) from "
              f"{file_bytes/1e9:.2f} GB ...")
        df = pd.read_csv(path, header=None,
                         skiprows=lambda i: i > 0 and rng_s.random() > prob,
                         names=COL_NAMES_USAGE, low_memory=False)
        print(f"  Loaded {len(df):,} rows across {df['machine_id'].nunique():,} machines")
        return df

    csv = raw_dir / "machine_usage.csv"
    if csv.exists():
        return _sample(csv)
    dest = raw_dir / "machine_usage.tar.gz"
    if dest.exists():
        print("  Extracting machine_usage.tar.gz ...")
        with tarfile.open(dest, "r:gz") as tf:
            tf.extractall(raw_dir)
        f = list(raw_dir.glob("machine_usage.csv"))
        if f:
            return _sample(f[0])
    print("  machine_usage.csv not found — will use synthetic data.")
    return None


# ── Fault labelling (percentile-based, computed on full data) ─────────────────

def compute_thresholds(df: pd.DataFrame) -> dict:
    """85th-percentile thresholds so every fault class covers ~15% of samples."""
    return {
        "cpu":  float(np.percentile(df["cpu_util_percent"].clip(0,100), 85)),
        "mem":  float(np.percentile(df["mem_util_percent"].clip(0,100), 85)),
        "disk": float(np.percentile(df["disk_io_percent"].clip(0,100),  85)),
        "net":  float(np.percentile((df["net_in"]+df["net_out"]).clip(0,200), 80)),
    }


def assign_labels(df: pd.DataFrame, thr: dict) -> np.ndarray:
    cpu  = df["cpu_util_percent"] > thr["cpu"]
    mem  = df["mem_util_percent"] > thr["mem"]
    disk = df["disk_io_percent"]  > thr["disk"]
    net  = (df["net_in"] + df["net_out"]) > thr["net"]
    multi = cpu.astype(int)+mem.astype(int)+disk.astype(int)+net.astype(int) >= 2
    labels = np.zeros(len(df), dtype=np.int64)
    labels[multi]          = 5
    labels[cpu  & ~multi]  = 1
    labels[mem  & ~multi]  = 2
    labels[disk & ~multi]  = 3
    labels[net  & ~multi]  = 4
    return labels


# ── Temporal windowing ────────────────────────────────────────────────────────

def make_windows(df: pd.DataFrame,
                 thr: dict,
                 window_size: int = WINDOW_SIZE,
                 step: int = WINDOW_STEP) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each machine, sort readings by time and slide a window of
    `window_size` steps with stride `step`.

    Returns
    -------
    X       : (N, window_size, n_features)  float32
    y       : (N,)                          int64   label of last timestep
    machine : (N,)                          object  machine_id
    """
    sequences, labels, machines = [], [], []
    feat_arr = df[FEATURE_COLS].values.astype(np.float32)
    lab_arr  = assign_labels(df, thr)
    mid_arr  = df["machine_id"].values

    for mid in df["machine_id"].unique():
        mask  = mid_arr == mid
        idx   = np.where(mask)[0]
        if len(idx) < window_size:
            continue
        # Sort by position in the (already time-sorted) dataframe
        f_m = feat_arr[idx]
        l_m = lab_arr[idx]
        for start in range(0, len(idx) - window_size + 1, step):
            end = start + window_size
            sequences.append(f_m[start:end])        # (W, F)
            labels.append(l_m[end - 1])             # label at last step
            machines.append(mid)

    if not sequences:
        return np.empty((0, window_size, len(FEATURE_COLS)), dtype=np.float32), \
               np.empty((0,), dtype=np.int64), np.array([])
    return (np.stack(sequences),
            np.array(labels, dtype=np.int64),
            np.array(machines))


# ── Synthetic data (fallback) ─────────────────────────────────────────────────

def _synthetic_machine(n_machines: int, n_rows_each: int,
                        node_type: str, domain_id: int,
                        rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for m in range(n_machines):
        mid = f"m_{domain_id}_{m:04d}"
        for t in range(n_rows_each):
            if node_type == "high_load":
                cpu, mem  = rng.uniform(30,95), rng.uniform(40,90)
                disk, net = rng.uniform(5,70),  rng.uniform(5,80)
            elif node_type == "cpu_heavy":
                cpu, mem  = rng.uniform(60,100), rng.uniform(20,60)
                disk, net = rng.uniform(5,40),   rng.uniform(5,50)
            elif node_type == "mem_heavy":
                cpu, mem  = rng.uniform(20,60), rng.uniform(60,100)
                disk, net = rng.uniform(5,40),  rng.uniform(5,50)
            elif node_type == "io_heavy":
                cpu, mem  = rng.uniform(20,70), rng.uniform(20,70)
                disk, net = rng.uniform(50,100), rng.uniform(20,80)
            else:
                cpu, mem  = rng.uniform(20,90), rng.uniform(20,90)
                disk, net = rng.uniform(10,80), rng.uniform(10,80)
            rows.append({
                "machine_id":       mid,
                "time_stamp":       float(t * 30),
                "cpu_util_percent": float(np.clip(cpu,0,100)),
                "mem_util_percent": float(np.clip(mem,0,100)),
                "mem_gps":          float(np.clip(mem*0.6+rng.uniform(-5,5),0,100)),
                "mkpi":             int(rng.uniform(100,3000)),
                "net_in":           float(np.clip(net,0,100)),
                "net_out":          float(np.clip(net*0.8+rng.uniform(-5,5),0,100)),
                "disk_io_percent":  float(np.clip(disk,0,100)),
                "failure_domain_1": domain_id,
            })
    return pd.DataFrame(rows)


def generate_synthetic(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    print("  Generating synthetic data (Alibaba 2018 schema) ...")
    src = _synthetic_machine(200, 60, "high_load", 0, rng)

    parts = []
    for i, t in enumerate(["cpu_heavy","mem_heavy","io_heavy","mixed"], 1):
        parts.append(_synthetic_machine(50, 60, t, i, rng))
    tgt = pd.concat(parts, ignore_index=True)

    src["failure_domain_1"] = 0
    tgt["failure_domain_1"] = tgt.get("failure_domain_1",
                                       pd.Series(range(1,5)).repeat(50*60).values[:len(tgt)])
    return src, tgt


# ── Build domains ─────────────────────────────────────────────────────────────

def build_domains(meta: Optional[pd.DataFrame],
                  usage: pd.DataFrame,
                  seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into source/target by failure_domain_1, clean, sort by machine+time."""
    if meta is not None:
        meta_last = (meta.sort_values("time_stamp")
                        .groupby("machine_id").last().reset_index()
                     [["machine_id","failure_domain_1","cpu_num","mem_size"]])
        df = usage.merge(meta_last, on="machine_id", how="left")
        df["failure_domain_1"] = df["failure_domain_1"].fillna(0).astype(int)
    else:
        df = usage.copy()
        df["failure_domain_1"] = 0

    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace(-1, np.nan).replace(101, np.nan).clip(0, 100)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # Sort by machine then time (essential for windowing)
    df = df.sort_values(["machine_id","time_stamp"]).reset_index(drop=True)

    unique_fd  = sorted(df["failure_domain_1"].unique())
    split      = max(1, len(unique_fd)//2)
    src_fd, tgt_fd = unique_fd[:split], unique_fd[split:]
    if not tgt_fd:
        tgt_fd = src_fd  # fallback: random machine split
        src_m  = df["machine_id"].unique()
        np.random.default_rng(seed).shuffle(src_m)
        split_m = int(len(src_m)*0.6)
        src = df[df["machine_id"].isin(src_m[:split_m])].copy()
        tgt = df[df["machine_id"].isin(src_m[split_m:])].copy()
    else:
        src = df[df["failure_domain_1"].isin(src_fd)].copy()
        tgt = df[df["failure_domain_1"].isin(tgt_fd)].copy()

    return src.reset_index(drop=True), tgt.reset_index(drop=True)


# ── Node type labels ──────────────────────────────────────────────────────────

def node_type_col(df: pd.DataFrame) -> pd.Series:
    cpu_p70  = df["cpu_util_percent"].quantile(0.70)
    mem_p70  = df["mem_util_percent"].quantile(0.70)
    disk_p60 = df["disk_io_percent"].quantile(0.60)
    def _nt(row):
        if row["cpu_util_percent"]  > cpu_p70:  return "cpu_heavy"
        if row["mem_util_percent"]  > mem_p70:  return "mem_heavy"
        if row["disk_io_percent"]   > disk_p60: return "io_heavy"
        return "mixed"
    return df.apply(_nt, axis=1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Step 0: Preparing Temporal Data (TA-DATL pipeline)")
    print("=" * 65)

    # Try real data first
    print("\n[1/4] Loading Alibaba 2018 data ...")
    meta  = load_machine_meta(RAW_DIR)
    usage = load_machine_usage(RAW_DIR, target_rows=500_000)

    if usage is not None:
        print("\n[2/4] Building source/target domains ...")
        src_df, tgt_df = build_domains(meta, usage)
        data_source = "real"
    else:
        print("\n[2/4] Generating synthetic data ...")
        src_df, tgt_df = generate_synthetic()
        data_source = "synthetic"

    print(f"  Source machines: {src_df['machine_id'].nunique():,}  "
          f"rows: {len(src_df):,}")
    print(f"  Target machines: {tgt_df['machine_id'].nunique():,}  "
          f"rows: {len(tgt_df):,}")

    # Compute thresholds on full data (shared between domains)
    full_df = pd.concat([src_df, tgt_df], ignore_index=True)
    thr = compute_thresholds(full_df)
    print(f"\n  Fault thresholds (percentile-based):\n    {thr}")

    # ── Temporal windows ─────────────────────────────────────────────────
    print("\n[3/4] Creating temporal windows ...")
    X_src_t, y_src_t, m_src = make_windows(src_df, thr)
    X_tgt_t, y_tgt_t, m_tgt = make_windows(tgt_df, thr)
    print(f"  Source windows: {X_src_t.shape}")
    print(f"  Target windows: {X_tgt_t.shape}")

    # 30% of target labeled (simulates limited annotation)
    rng = np.random.default_rng(42)
    tgt_labeled = rng.random(len(X_tgt_t)) < 0.30

    # ── Flat representation (mean-pool over time) for baselines ───────────
    X_src_flat = X_src_t.mean(axis=1)   # (N, F)
    X_tgt_flat = X_tgt_t.mean(axis=1)

    # ── Node types for heterogeneous experiment ───────────────────────────
    # Assign node type from the mean feature values of each window
    def _nt_from_window(X_win):
        means = X_win.mean(axis=1)   # (N, F)
        df_tmp = pd.DataFrame(means, columns=FEATURE_COLS)
        cpu_p70  = df_tmp["cpu_util_percent"].quantile(0.70)
        mem_p70  = df_tmp["mem_util_percent"].quantile(0.70)
        disk_p60 = df_tmp["disk_io_percent"].quantile(0.60)
        nt = []
        for _, r in df_tmp.iterrows():
            if r["cpu_util_percent"] > cpu_p70:  nt.append("cpu_heavy")
            elif r["mem_util_percent"] > mem_p70: nt.append("mem_heavy")
            elif r["disk_io_percent"]  > disk_p60: nt.append("io_heavy")
            else: nt.append("mixed")
        return np.array(nt)
    tgt_node_types = _nt_from_window(X_tgt_t)

    # ── Save ──────────────────────────────────────────────────────────────
    print("\n[4/4] Saving ...")

    np.savez_compressed(PROC_DIR / "source_temporal.npz",
                        X=X_src_t, y=y_src_t, machine=m_src)
    np.savez_compressed(PROC_DIR / "target_temporal.npz",
                        X=X_tgt_t, y=y_tgt_t, machine=m_tgt,
                        labeled=tgt_labeled, node_type=tgt_node_types)

    # Flat parquet files for baselines
    def _to_parquet(X_flat, y, labeled, path):
        df = pd.DataFrame(X_flat, columns=FEATURE_COLS)
        df["label"]   = y
        df["labeled"] = labeled
        df.to_parquet(path, index=False)

    _to_parquet(X_src_flat, y_src_t,
                np.ones(len(X_src_t), dtype=bool),
                PROC_DIR / "source_flat.parquet")
    _to_parquet(X_tgt_flat, y_tgt_t, tgt_labeled,
                PROC_DIR / "target_flat.parquet")

    # Source/target label distributions
    def _dist(y):
        vals, cnts = np.unique(y, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, cnts)}

    meta_info = {
        "data_source":    data_source,
        "window_size":    WINDOW_SIZE,
        "window_step":    WINDOW_STEP,
        "n_features":     len(FEATURE_COLS),
        "feature_cols":   FEATURE_COLS,
        "n_classes":      N_CLASSES,
        "fault_names":    FAULT_NAMES,
        "n_src_windows":  int(len(X_src_t)),
        "n_tgt_windows":  int(len(X_tgt_t)),
        "tgt_labeled_pct": float(tgt_labeled.mean() * 100),
        "thresholds":     thr,
        "src_label_dist": _dist(y_src_t),
        "tgt_label_dist": _dist(y_tgt_t),
    }
    with open(PROC_DIR / "meta.json", "w") as f:
        json.dump(meta_info, f, indent=2)

    # Print distribution
    print("\n  Source fault distribution:")
    for k, v in _dist(y_src_t).items():
        pct = v / len(y_src_t) * 100
        print(f"    {FAULT_NAMES[k]:>16s} (class {k}): {v:>6,}  ({pct:.1f}%)")
    print("\n  Target fault distribution:")
    for k, v in _dist(y_tgt_t).items():
        pct = v / len(y_tgt_t) * 100
        print(f"    {FAULT_NAMES[k]:>16s} (class {k}): {v:>6,}  ({pct:.1f}%)")

    print(f"\n  Data source: {data_source.upper()}")
    print("  Done. Run 01_train_all_models.py next.")
    print("=" * 65)


if __name__ == "__main__":
    main()
