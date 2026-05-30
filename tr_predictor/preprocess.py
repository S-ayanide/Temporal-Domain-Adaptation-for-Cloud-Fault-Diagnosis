"""
preprocess.py — Data preparation for Tr-Predictor (Liu et al. 2022).

Tr-Predictor is designed for *small-sample* prediction:
  - Target domain: short task/machine traces  (< 100 time points)
  - Source domain: many longer traces from related workloads

We extract:
  1. GC19 per-cell traces from the Rossi-replication preprocessed .npy files.
     Each cell (GC19_a … GC19_h) is treated as ONE source/target domain
     (the full preprocessed trace is used; real small-sample comes from
     splitting into a tiny target window of 72 points = 6h).

  2. AC18 per-machine traces from raw CSV (alibaba/machine_usage.csv),
     keeping machines with at least MIN_LEN 5-min bins.

The preprocess_for_tr_predictor() function returns a dict:
    {
        "name": str,
        "series": np.ndarray,      # shape (T, C) — cpu, mem (normalised 0-1)
        "type": "gc19" | "ac18",
    }

Rolling-window helpers for small-sample experiments are also here.
"""

import os
import gzip
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEQ_LEN       = 24    # 2h look-back (24 × 5min) — paper uses short windows
HORIZON       = 1     # 5 min ahead
TARGET_LEN    = 72    # 6h of 5-min bins for target training split
MIN_LEN       = 200   # minimum series length to be kept
AC18_TOP_N    = 10    # number of AC18 machines to extract


# ---------------------------------------------------------------------------
# 1. Load GC19 from preprocessed .npy files (Rossi replication output)
# ---------------------------------------------------------------------------

def load_gc19_npy(data_dir: str) -> List[Dict]:
    """
    Load preprocessed GC19 cell traces from .npy files.

    Expects files named GC19_a.npy … GC19_h.npy in data_dir.
    Each file is (T, 2) float32: [cpu_fraction, mem_fraction].

    Returns list of dicts with keys: name, series, type.
    """
    datasets = []
    for cell in "abcdefgh":
        path = os.path.join(data_dir, f"GC19_{cell}.npy")
        if not os.path.exists(path):
            continue
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if len(arr) < MIN_LEN:
            continue
        datasets.append({
            "name":   f"GC19_{cell}",
            "series": arr,
            "type":   "gc19",
        })
    return datasets


# ---------------------------------------------------------------------------
# 2. Load AC18 per-machine from raw CSV
# ---------------------------------------------------------------------------

def load_ac18_raw(raw_dir: str, top_n: int = AC18_TOP_N,
                  min_len: int = MIN_LEN) -> List[Dict]:
    """
    Extract per-machine 5-min traces from the Alibaba AC18 CSV.

    CSV columns: machine_id, time_stamp, cpu_util_percent, mem_util_percent, ...
    time_stamp is in seconds; bins by 5-min intervals.

    Returns list of dicts: name, series (T,2) normalised [0,1], type='ac18'.
    """
    csv_path = os.path.join(raw_dir, "alibaba", "machine_usage.csv")
    if not os.path.exists(csv_path):
        print(f"[preprocess] AC18 CSV not found at {csv_path}")
        return []

    print("[preprocess] Loading AC18 CSV …", flush=True)
    cols = ["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"]
    try:
        df = pd.read_csv(csv_path, header=None,
                         names=["machine_id","time_stamp","cpu_util_percent",
                                "mem_util_percent","disk_io_percent"],
                         usecols=[0,1,2,3], dtype={"machine_id": str})
    except Exception:
        df = pd.read_csv(csv_path, usecols=cols, dtype={"machine_id": str})

    # 5-min bin
    BIN_S = 300
    df["bin"] = (df["time_stamp"] // BIN_S).astype(int)

    # Count bins per machine; keep machines with longest traces
    counts = df.groupby("machine_id")["bin"].nunique().sort_values(ascending=False)
    top_machines = counts[counts >= min_len].head(top_n).index.tolist()

    if not top_machines:
        print("[preprocess] No AC18 machines with enough data.")
        return []

    datasets = []
    for mid in top_machines:
        sub = df[df["machine_id"] == mid].copy()
        grp = sub.groupby("bin")[["cpu_util_percent","mem_util_percent"]].mean()
        grp = grp.sort_index()
        # Fill gaps with forward-fill then zero
        full_idx = pd.RangeIndex(grp.index.min(), grp.index.max() + 1)
        grp = grp.reindex(full_idx).ffill().fillna(0.0)
        arr = grp.values.astype(np.float32)
        # Normalise percent → [0,1]
        arr = np.clip(arr / 100.0, 0.0, 1.0)
        if len(arr) < min_len:
            continue
        datasets.append({
            "name":   f"AC18_{mid[:8]}",
            "series": arr,
            "type":   "ac18",
        })
    print(f"[preprocess] Loaded {len(datasets)} AC18 machines.")
    return datasets


# ---------------------------------------------------------------------------
# 3. Rolling-window maker for small-sample experiments
# ---------------------------------------------------------------------------

def make_windows(
    series: np.ndarray,
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create (X, Y) pairs from a time series by sliding window.

    Parameters
    ----------
    series  : (T, C) or (T,) float array
    seq_len : look-back window length
    horizon : forecast steps ahead

    Returns
    -------
    X : (N, seq_len, C)
    Y : (N, C)
    """
    if series.ndim == 1:
        series = series[:, None]
    T, C = series.shape
    xs, ys = [], []
    for i in range(T - seq_len - horizon + 1):
        xs.append(series[i: i + seq_len])
        ys.append(series[i + seq_len + horizon - 1])
    if not xs:
        return np.empty((0, seq_len, C)), np.empty((0, C))
    return np.stack(xs), np.stack(ys)


# ---------------------------------------------------------------------------
# 4. Small-sample split: target = first TARGET_LEN points (after seq_len burn-in)
# ---------------------------------------------------------------------------

def small_sample_split(
    dataset: Dict,
    target_len: int = TARGET_LEN,
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
) -> Dict:
    """
    Split a single dataset into source context + small target train/val/test.

    The 'source' portion is the tail of the series (excluded from target training)
    used only when this dataset is a *source* in the TrAdaBoost experiment.

    Returns
    -------
    dict with keys:
        X_src, Y_src       — source windows (long part → used only as source domain)
        X_tgt_tr, Y_tgt_tr — tiny target train
        X_tgt_val, Y_tgt_val
        X_tgt_te, Y_tgt_te — test windows (unseen)
        scaler_min, scaler_max — MinMax params fitted on tgt_tr
    """
    s = dataset["series"]   # (T, C)
    T = len(s)

    # Target = first window of target_len points (small-sample regime)
    # Test   = last test_frac of the series (held out)
    n_test = max(seq_len + horizon, int(T * test_frac))
    n_tgt  = min(target_len, T - n_test - seq_len)

    if n_tgt <= seq_len + horizon:
        n_tgt = seq_len + horizon + 1

    # Indices
    tgt_end  = n_tgt
    src_end  = T - n_test
    test_end = T

    s_tgt = s[:tgt_end]
    s_src = s[tgt_end: src_end]
    s_te  = s[src_end - seq_len: test_end]   # overlap by seq_len for context

    # MinMax scale on target train data
    scaler_min = s_tgt.min(axis=0, keepdims=True)
    scaler_max = s_tgt.max(axis=0, keepdims=True)
    rng = scaler_max - scaler_min
    rng[rng < 1e-8] = 1.0

    def _scale(x):
        return (x - scaler_min) / rng

    s_tgt_sc = _scale(s_tgt)
    s_src_sc  = _scale(s_src)
    s_te_sc   = _scale(s_te)

    # Val split from target
    n_val = max(1, int(len(s_tgt_sc) * val_frac))
    s_tgt_tr  = s_tgt_sc[:-n_val]
    s_tgt_val = s_tgt_sc[-n_val:]

    X_tgt_tr,  Y_tgt_tr  = make_windows(s_tgt_tr,  seq_len, horizon)
    X_tgt_val, Y_tgt_val = make_windows(s_tgt_val, seq_len, horizon)
    X_src,     Y_src     = make_windows(s_src_sc,  seq_len, horizon)
    X_te,      Y_te      = make_windows(s_te_sc,   seq_len, horizon)

    return {
        "name":        dataset["name"],
        "X_src":       X_src,
        "Y_src":       Y_src,
        "X_tgt_tr":    X_tgt_tr,
        "Y_tgt_tr":    Y_tgt_tr,
        "X_tgt_val":   X_tgt_val,
        "Y_tgt_val":   Y_tgt_val,
        "X_tgt_te":    X_te,
        "Y_tgt_te":    Y_te,
        "scaler_min":  scaler_min,
        "scaler_max":  scaler_max,
        "tgt_series":  s_tgt,   # raw, for similarity computation
        "src_series":  s_src,
    }


# ---------------------------------------------------------------------------
# 5. Master loader
# ---------------------------------------------------------------------------

def load_all(
    data_dir: str,          # Rossi-replication output (*.npy files)
    raw_dir: str = None,    # raw data dir (for AC18 per-machine)
    target_len: int = TARGET_LEN,
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON,
) -> List[Dict]:
    """
    Load all available datasets and apply small-sample split.

    Returns list of split-dicts (one per dataset).
    """
    all_raw = load_gc19_npy(data_dir)

    if raw_dir:
        all_raw += load_ac18_raw(raw_dir)

    if not all_raw:
        raise RuntimeError(f"No datasets found in {data_dir}")

    splits = []
    for ds in all_raw:
        sp = small_sample_split(
            ds, target_len=target_len, seq_len=seq_len, horizon=horizon
        )
        splits.append(sp)
        print(f"  {ds['name']}: "
              f"src={len(sp['X_src'])} tgt_tr={len(sp['X_tgt_tr'])} "
              f"te={len(sp['X_tgt_te'])}")
    return splits
