"""
preprocess.py — Data preparation for Tr-Predictor (Liu et al. 2022).

Loads DIRECTLY from raw data. Supports a --cache_dir so processed .npy
files are saved after each cell — progress is never lost on timeout.

Raw data layout expected at --raw_dir:
  google/cell_a/instance_usage-*.json.gz   → GC19_a
  google/cell_b/instance_usage-*.json.gz   → GC19_b
  ...
  google/cell_h/instance_usage-*.json.gz   → GC19_h
  alibaba/machine_usage.csv                → AC18 (per-machine)

Cache layout (--cache_dir, default: <raw_dir>/../tr_cache):
  GC19_a.npy, GC19_b.npy, …, AC18_<id>.npy
"""

import os
import gzip
import json
import glob
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEQ_LEN    = 24    # 2h look-back (24 × 5-min bins)
HORIZON    = 1     # 5 min ahead
TARGET_LEN = 72    # 6h of 5-min bins — "small sample" target train window
MIN_LEN    = 200   # minimum series length to keep a domain
AC18_TOP_N = 5     # number of AC18 machines to extract
BIN_S      = 300   # 5 minutes in seconds


# ---------------------------------------------------------------------------
# 1. GC19 — fast vectorised loading from JSON.gz shards
# ---------------------------------------------------------------------------

def _parse_gc19_shard(gz_path: str) -> pd.DataFrame:
    """Parse one JSON.gz shard → DataFrame[start_s, end_s, cpu, mem]."""
    rows = []
    with gzip.open(gz_path, "rt", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            try:
                start_s = float(rec.get("start_time", 0)) / 1e6
                end_s   = float(rec.get("end_time",   0)) / 1e6
            except (TypeError, ValueError):
                continue

            usage = rec.get("average_usage", {})
            if isinstance(usage, str):
                try:
                    usage = json.loads(usage)
                except Exception:
                    usage = {}

            cpu = usage.get("cpus",   None)
            mem = usage.get("memory", None)
            if cpu is None or mem is None:
                continue
            if end_s <= start_s:
                continue

            rows.append((start_s, end_s, float(cpu), float(mem)))

    if not rows:
        return pd.DataFrame(columns=["start_s", "end_s", "cpu", "mem"])
    return pd.DataFrame(rows, columns=["start_s", "end_s", "cpu", "mem"])


def _aggregate_5min(df: pd.DataFrame) -> np.ndarray:
    """
    Vectorised 5-min bin aggregation (no Python row loops).

    Each record is assigned to its midpoint bin; then we do a
    groupby-mean. This is ~100× faster than the iterrows approach
    and accurate when records are ≤ one bin long (true for GC19).
    """
    if df.empty:
        return np.empty((0, 2), dtype=np.float32)

    # Midpoint bin assignment (vectorised)
    mid_s = (df["start_s"].values + df["end_s"].values) / 2.0
    df = df.copy()
    df["bin"] = (mid_s // BIN_S).astype(np.int64)

    grp = df.groupby("bin")[["cpu", "mem"]].mean().sort_index()

    # Fill gaps with forward-fill
    full_idx = pd.RangeIndex(grp.index.min(), grp.index.max() + 1)
    grp = grp.reindex(full_idx).ffill().fillna(0.0)

    arr = np.clip(grp.values.astype(np.float32), 0.0, 1.0)
    return arr


def load_gc19_raw(raw_dir: str, cache_dir: str = None,
                  max_shards: int = None) -> List[Dict]:
    """
    Load GC19 per-cell traces.  Saves/loads .npy cache so progress
    survives a timeout.

    Parameters
    ----------
    raw_dir    : root containing google/cell_{a-h}/
    cache_dir  : directory to save/load .npy files (created if needed)
    max_shards : cap JSON.gz shards per cell (None = all)
    """
    datasets = []

    for cell in "abcdefgh":
        name      = f"GC19_{cell}"
        cache_npy = os.path.join(cache_dir, f"{name}.npy") if cache_dir else None

        # ---- Try cache first ----
        if cache_npy and os.path.exists(cache_npy):
            arr = np.load(cache_npy)
            print(f"[preprocess] {name}: loaded from cache  shape={arr.shape}")
            datasets.append({"name": name, "series": arr, "type": "gc19"})
            continue

        cell_dir = os.path.join(raw_dir, "google", f"cell_{cell}")
        if not os.path.isdir(cell_dir):
            continue

        shards = sorted(glob.glob(
            os.path.join(cell_dir, "instance_usage-*.json.gz")))
        if not shards:
            print(f"[preprocess] {name}: no shards in {cell_dir}")
            continue
        if max_shards is not None:
            shards = shards[:max_shards]

        n_workers = min(cpu_count(), len(shards), 16)
        print(f"[preprocess] {name}: reading {len(shards)} shards "
              f"({n_workers} workers) …", flush=True)

        with Pool(processes=n_workers) as pool:
            results = pool.map(_parse_gc19_shard, shards)
        frames = [df for df in results if not df.empty]

        if not frames:
            print(f"[preprocess] {name}: no usable records — skipping.")
            continue

        all_df = pd.concat(frames, ignore_index=True)
        arr    = _aggregate_5min(all_df)

        if len(arr) < MIN_LEN:
            print(f"[preprocess] {name}: only {len(arr)} bins — skipping.")
            continue

        print(f"[preprocess] {name}: {len(arr)} 5-min bins  shape={arr.shape}")

        # ---- Save to cache immediately ----
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_npy, arr)
            print(f"[preprocess] {name}: saved to cache → {cache_npy}")

        datasets.append({"name": name, "series": arr, "type": "gc19"})

    return datasets


# ---------------------------------------------------------------------------
# 2. AC18 — per-machine from CSV
# ---------------------------------------------------------------------------

def load_ac18_raw(raw_dir: str, cache_dir: str = None,
                  top_n: int = AC18_TOP_N,
                  min_len: int = MIN_LEN) -> List[Dict]:
    """Load per-machine AC18 traces. Saves/loads .npy cache."""

    # Check cache for any AC18 files
    if cache_dir:
        cached = sorted(glob.glob(os.path.join(cache_dir, "AC18_*.npy")))
        if cached:
            datasets = []
            for p in cached:
                arr  = np.load(p)
                name = os.path.splitext(os.path.basename(p))[0]
                datasets.append({"name": name, "series": arr, "type": "ac18"})
                print(f"[preprocess] {name}: loaded from cache  shape={arr.shape}")
            return datasets

    csv_path = os.path.join(raw_dir, "alibaba", "machine_usage.csv")
    if not os.path.exists(csv_path):
        print(f"[preprocess] AC18 CSV not found at {csv_path} — skipping.")
        return []

    print("[preprocess] Loading AC18 CSV …", flush=True)
    # Sniff the actual column count from the first line
    with open(csv_path) as _f:
        first = _f.readline()
    n_cols = len(first.split(","))
    # Always read by position 0,1,2,3 regardless of total column count
    col_names = [f"c{i}" for i in range(n_cols)]
    col_names[0] = "machine_id"
    col_names[1] = "time_stamp"
    col_names[2] = "cpu_util_percent"
    col_names[3] = "mem_util_percent"
    df = pd.read_csv(csv_path, header=None, names=col_names,
                     usecols=[0, 1, 2, 3],
                     dtype={"machine_id": str})

    df["bin"] = (df["time_stamp"] // BIN_S).astype(int)

    counts = (df.groupby("machine_id")["bin"]
                .nunique()
                .sort_values(ascending=False))
    top_machines = counts[counts >= min_len].head(top_n).index.tolist()

    if not top_machines:
        print("[preprocess] No AC18 machines with enough data.")
        return []

    datasets = []
    for mid in top_machines:
        sub  = df[df["machine_id"] == mid]
        grp  = (sub.groupby("bin")[["cpu_util_percent", "mem_util_percent"]]
                   .mean().sort_index())
        full = pd.RangeIndex(grp.index.min(), grp.index.max() + 1)
        grp  = grp.reindex(full).ffill().fillna(0.0)
        arr  = np.clip(grp.values.astype(np.float32) / 100.0, 0.0, 1.0)
        if len(arr) < min_len:
            continue
        name = f"AC18_{str(mid)[:8]}"
        print(f"[preprocess] {name}: {len(arr)} 5-min bins  shape={arr.shape}")

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(os.path.join(cache_dir, f"{name}.npy"), arr)

        datasets.append({"name": name, "series": arr, "type": "ac18"})

    return datasets


# ---------------------------------------------------------------------------
# 3. Rolling-window helper
# ---------------------------------------------------------------------------

def make_windows(series: np.ndarray,
                 seq_len: int = SEQ_LEN,
                 horizon: int = HORIZON) -> Tuple[np.ndarray, np.ndarray]:
    if series.ndim == 1:
        series = series[:, None]
    T, C = series.shape
    N = T - seq_len - horizon + 1
    if N <= 0:
        return (np.empty((0, seq_len, C), dtype=np.float32),
                np.empty((0, C), dtype=np.float32))
    # Vectorised strided construction
    idx = np.arange(N)[:, None] + np.arange(seq_len)[None, :]   # (N, seq_len)
    X   = series[idx]                                             # (N, seq_len, C)
    Y   = series[idx[:, -1] + horizon]                           # (N, C)
    return X.astype(np.float32), Y.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Small-sample split
# ---------------------------------------------------------------------------

def small_sample_split(dataset: Dict,
                       target_len: int = TARGET_LEN,
                       seq_len: int = SEQ_LEN,
                       horizon: int = HORIZON,
                       val_frac: float = 0.2,
                       test_frac: float = 0.2) -> Dict:
    s = dataset["series"]
    T = len(s)

    n_test = max(seq_len + horizon, int(T * test_frac))
    n_tgt  = min(target_len, T - n_test - seq_len)
    if n_tgt <= seq_len + horizon:
        n_tgt = seq_len + horizon + 1

    s_tgt = s[:n_tgt]
    s_src = s[n_tgt: T - n_test]
    s_te  = s[T - n_test - seq_len:]

    mn  = s_tgt.min(axis=0, keepdims=True)
    mx  = s_tgt.max(axis=0, keepdims=True)
    rng = np.where(mx - mn < 1e-8, 1.0, mx - mn)

    def _sc(x):
        return (x - mn) / rng

    tgt_sc = _sc(s_tgt)
    src_sc = _sc(s_src)
    te_sc  = _sc(s_te)

    # Val needs at least seq_len+horizon steps to form any windows
    min_val_steps = seq_len + horizon
    n_val = max(min_val_steps, int(len(tgt_sc) * val_frac))
    # Don't let val consume so much that train has nothing left
    if n_val >= len(tgt_sc) - (seq_len + horizon):
        n_val = 0
    if n_val > 0:
        tgt_tr_sc  = tgt_sc[:-n_val]
        tgt_val_sc = tgt_sc[-n_val:]
    else:
        tgt_tr_sc  = tgt_sc
        tgt_val_sc = np.empty((0, tgt_sc.shape[1]), dtype=np.float32)

    X_tr,  Y_tr  = make_windows(tgt_tr_sc,  seq_len, horizon)
    X_val, Y_val = make_windows(tgt_val_sc, seq_len, horizon)
    X_src, Y_src = make_windows(src_sc,     seq_len, horizon)
    X_te,  Y_te  = make_windows(te_sc,      seq_len, horizon)

    return {
        "name":       dataset["name"],
        "X_src":      X_src,  "Y_src":      Y_src,
        "X_tgt_tr":   X_tr,   "Y_tgt_tr":   Y_tr,
        "X_tgt_val":  X_val,  "Y_tgt_val":  Y_val,
        "X_tgt_te":   X_te,   "Y_tgt_te":   Y_te,
        "scaler_min": mn,     "scaler_max":  mx,
        "tgt_series": s_tgt,  "src_series":  s_src,
    }


# ---------------------------------------------------------------------------
# 5. Master loader
# ---------------------------------------------------------------------------

def load_all(raw_dir: str,
             cache_dir: str = None,
             target_len: int = TARGET_LEN,
             seq_len: int = SEQ_LEN,
             horizon: int = HORIZON,
             max_shards: int = None) -> List[Dict]:
    """
    Load all domains and apply small-sample split.

    Processed arrays are cached as .npy files in cache_dir so that a
    timeout on the first run doesn't lose everything — subsequent runs
    load instantly from cache.

    Parameters
    ----------
    raw_dir    : root with google/ and alibaba/ subdirs
    cache_dir  : where to save/load .npy files
                 (default: <raw_dir>/../tr_cache)
    max_shards : cap JSON.gz shards per GC19 cell (None = all)
    """
    raw_dir = os.path.realpath(os.path.expanduser(raw_dir))
    if not os.path.isdir(raw_dir):
        raise RuntimeError(f"raw_dir does not exist: {raw_dir}")

    if cache_dir is None:
        cache_dir = os.path.join(raw_dir, "..", "tr_cache")
    cache_dir = os.path.realpath(os.path.expanduser(cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[preprocess] raw_dir  : {raw_dir}")
    print(f"[preprocess] cache_dir: {cache_dir}")

    all_raw  = load_gc19_raw(raw_dir, cache_dir=cache_dir,
                              max_shards=max_shards)
    all_raw += load_ac18_raw(raw_dir, cache_dir=cache_dir)

    if not all_raw:
        raise RuntimeError(
            f"No datasets found under {raw_dir}.\n"
            f"Expected google/cell_*/instance_usage-*.json.gz "
            f"and/or alibaba/machine_usage.csv"
        )

    print(f"\n[preprocess] {len(all_raw)} domains. Splitting …\n")
    splits = []
    for ds in all_raw:
        sp = small_sample_split(ds, target_len=target_len,
                                seq_len=seq_len, horizon=horizon)
        splits.append(sp)
        print(f"  {sp['name']:<22}  "
              f"tgt_tr={len(sp['X_tgt_tr']):>4}  "
              f"src={len(sp['X_src']):>6}  "
              f"test={len(sp['X_tgt_te']):>4}")
    return splits
