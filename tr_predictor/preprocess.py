"""
preprocess.py — Data preparation for Tr-Predictor (Liu et al. 2022).

Loads DIRECTLY from raw data — no prior preprocessing required.

Raw data layout expected at --raw_dir:
  google/cell_a/instance_usage-*.json.gz   → GC19_a
  google/cell_b/instance_usage-*.json.gz   → GC19_b
  ...
  google/cell_h/instance_usage-*.json.gz   → GC19_h
  alibaba/machine_usage.csv                → AC18 (per-machine)

Each GC19 cell and each AC18 machine is one domain.
Small-sample split: first TARGET_LEN timesteps = target train,
remainder = source context for other domains, last 20% = test.
"""

import os
import gzip
import json
import glob
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEQ_LEN       = 24    # 2h look-back (24 × 5-min bins)
HORIZON       = 1     # 5 min ahead
TARGET_LEN    = 72    # 6h of 5-min bins — "small sample" target train window
MIN_LEN       = 200   # minimum series length to keep a domain
AC18_TOP_N    = 5     # number of AC18 machines to extract
BIN_S         = 300   # 5 minutes in seconds


# ---------------------------------------------------------------------------
# 1. GC19 — load from JSON.gz shards
# ---------------------------------------------------------------------------

def _parse_gc19_shard(gz_path: str) -> pd.DataFrame:
    """
    Parse one instance_usage-*.json.gz shard.
    Returns DataFrame with columns: start_s, end_s, cpu, mem.
    """
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

            # start/end in microseconds → seconds
            start_s = rec.get("start_time", 0) / 1e6
            end_s   = rec.get("end_time",   0) / 1e6

            usage = rec.get("average_usage", {})
            if isinstance(usage, str):
                try:
                    usage = json.loads(usage)
                except Exception:
                    usage = {}

            cpu = usage.get("cpus", None)
            mem = usage.get("memory", None)

            if cpu is None or mem is None:
                continue
            if end_s <= start_s:
                continue

            rows.append({"start_s": start_s, "end_s": end_s,
                         "cpu": float(cpu), "mem": float(mem)})

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["start_s", "end_s", "cpu", "mem"]
    )


def _aggregate_5min(df: pd.DataFrame) -> np.ndarray:
    """
    Aggregate fractional-overlap records into 5-min bins.
    Returns (T, 2) float32 array [cpu_fraction, mem_fraction].
    """
    if df.empty:
        return np.empty((0, 2), dtype=np.float32)

    t_min = int(df["start_s"].min() // BIN_S)
    t_max = int(df["end_s"].max()   // BIN_S) + 1

    cpu_acc = np.zeros(t_max - t_min + 1, dtype=np.float64)
    mem_acc = np.zeros_like(cpu_acc)
    wgt_acc = np.zeros_like(cpu_acc)

    for _, row in df.iterrows():
        b0 = int(row["start_s"] // BIN_S)
        b1 = int(row["end_s"]   // BIN_S)
        for b in range(b0, b1 + 1):
            overlap = min(row["end_s"],   (b + 1) * BIN_S) \
                    - max(row["start_s"],  b       * BIN_S)
            if overlap <= 0:
                continue
            idx = b - t_min
            if 0 <= idx < len(cpu_acc):
                cpu_acc[idx] += row["cpu"] * overlap
                mem_acc[idx] += row["mem"] * overlap
                wgt_acc[idx] += overlap

    valid = wgt_acc > 0
    if not valid.any():
        return np.empty((0, 2), dtype=np.float32)

    cpu_avg = np.where(valid, cpu_acc / wgt_acc, np.nan)
    mem_avg = np.where(valid, mem_acc / wgt_acc, np.nan)

    arr = np.stack([cpu_avg, mem_avg], axis=1).astype(np.float32)
    # Drop leading/trailing NaN rows
    first = int(np.argmax(valid))
    last  = len(valid) - 1 - int(np.argmax(valid[::-1]))
    arr   = arr[first: last + 1]

    # Forward-fill interior NaNs
    df_tmp = pd.DataFrame(arr, columns=["cpu", "mem"])
    df_tmp = df_tmp.ffill().fillna(0.0)
    return df_tmp.values.astype(np.float32)


def load_gc19_raw(raw_dir: str, max_shards: int = None) -> List[Dict]:
    """
    Load GC19 per-cell traces from JSON.gz shards.

    Parameters
    ----------
    raw_dir    : root raw data dir (contains google/cell_a/ … cell_h/)
    max_shards : limit shards per cell (None = all). Use a small number
                 (e.g. 20) for a quick smoke test.

    Returns list of dicts: name, series (T,2) in [0,1], type='gc19'.
    """
    datasets = []
    for cell in "abcdefgh":
        cell_dir = os.path.join(raw_dir, "google", f"cell_{cell}")
        if not os.path.isdir(cell_dir):
            continue

        shards = sorted(glob.glob(os.path.join(cell_dir, "instance_usage-*.json.gz")))
        if not shards:
            print(f"[preprocess] GC19_{cell}: no JSON.gz shards found in {cell_dir}")
            continue

        if max_shards is not None:
            shards = shards[:max_shards]

        print(f"[preprocess] GC19_{cell}: reading {len(shards)} shards …", flush=True)

        frames = []
        for sh in shards:
            df = _parse_gc19_shard(sh)
            if not df.empty:
                frames.append(df)

        if not frames:
            print(f"[preprocess] GC19_{cell}: no usable records.")
            continue

        all_df = pd.concat(frames, ignore_index=True)
        arr = _aggregate_5min(all_df)

        if len(arr) < MIN_LEN:
            print(f"[preprocess] GC19_{cell}: only {len(arr)} bins after aggregation — skipping.")
            continue

        # Clip to [0,1] (cpu/mem are already fractions in GC19)
        arr = np.clip(arr, 0.0, 1.0)
        datasets.append({"name": f"GC19_{cell}", "series": arr, "type": "gc19"})
        print(f"[preprocess] GC19_{cell}: {len(arr)} 5-min bins  shape={arr.shape}")

    return datasets


# ---------------------------------------------------------------------------
# 2. AC18 — per-machine from CSV
# ---------------------------------------------------------------------------

def load_ac18_raw(raw_dir: str, top_n: int = AC18_TOP_N,
                  min_len: int = MIN_LEN) -> List[Dict]:
    """
    Extract per-machine 5-min traces from alibaba/machine_usage.csv.

    Returns list of dicts: name, series (T,2) in [0,1], type='ac18'.
    """
    csv_path = os.path.join(raw_dir, "alibaba", "machine_usage.csv")
    if not os.path.exists(csv_path):
        print(f"[preprocess] AC18 CSV not found at {csv_path} — skipping.")
        return []

    print("[preprocess] Loading AC18 CSV …", flush=True)
    try:
        df = pd.read_csv(csv_path, header=None,
                         names=["machine_id", "time_stamp",
                                "cpu_util_percent", "mem_util_percent",
                                "disk_io_percent"],
                         usecols=[0, 1, 2, 3],
                         dtype={"machine_id": str})
    except Exception:
        cols = ["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"]
        df = pd.read_csv(csv_path, usecols=cols, dtype={"machine_id": str})

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
        sub = df[df["machine_id"] == mid]
        grp = (sub.groupby("bin")[["cpu_util_percent", "mem_util_percent"]]
                  .mean()
                  .sort_index())
        full_idx = pd.RangeIndex(grp.index.min(), grp.index.max() + 1)
        grp = grp.reindex(full_idx).ffill().fillna(0.0)
        arr = np.clip(grp.values.astype(np.float32) / 100.0, 0.0, 1.0)
        if len(arr) < min_len:
            continue
        name = f"AC18_{str(mid)[:8]}"
        datasets.append({"name": name, "series": arr, "type": "ac18"})
        print(f"[preprocess] {name}: {len(arr)} 5-min bins  shape={arr.shape}")

    return datasets


# ---------------------------------------------------------------------------
# 3. Rolling-window helper
# ---------------------------------------------------------------------------

def make_windows(
    series: np.ndarray,
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window segmentation.

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
        return np.empty((0, seq_len, C), dtype=np.float32), \
               np.empty((0, C), dtype=np.float32)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Small-sample split
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
    Split one domain into:
      - tiny target train (first target_len steps)
      - source context (middle portion — used when this domain is a source)
      - held-out test (last test_frac of the series)

    MinMax scaling is fitted on target train only.
    """
    s = dataset["series"]   # (T, C)
    T = len(s)

    n_test = max(seq_len + horizon, int(T * test_frac))
    n_tgt  = min(target_len, T - n_test - seq_len)
    if n_tgt <= seq_len + horizon:
        n_tgt = seq_len + horizon + 1

    tgt_end = n_tgt
    src_end = T - n_test

    s_tgt = s[:tgt_end]
    s_src = s[tgt_end: src_end]
    s_te  = s[src_end - seq_len:]   # include look-back context

    # MinMax on target train
    scaler_min = s_tgt.min(axis=0, keepdims=True)
    scaler_max = s_tgt.max(axis=0, keepdims=True)
    rng = scaler_max - scaler_min
    rng[rng < 1e-8] = 1.0

    def _scale(x):
        return (x - scaler_min) / rng

    s_tgt_sc = _scale(s_tgt)
    s_src_sc  = _scale(s_src)
    s_te_sc   = _scale(s_te)

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
        "tgt_series":  s_tgt,
        "src_series":  s_src,
    }


# ---------------------------------------------------------------------------
# 5. Master loader
# ---------------------------------------------------------------------------

def load_all(
    raw_dir: str,
    target_len: int = TARGET_LEN,
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON,
    max_shards: int = None,
) -> List[Dict]:
    """
    Load all available datasets from raw_dir and apply small-sample split.

    Parameters
    ----------
    raw_dir    : root of raw data (contains google/ and/or alibaba/ subdirs)
    target_len : target training window size (timesteps)
    seq_len    : look-back window
    horizon    : forecast horizon
    max_shards : limit GC19 JSON.gz shards per cell (None = all; use ~20 for smoke test)

    Returns
    -------
    List of split dicts, one per domain.
    """
    raw_dir = os.path.realpath(os.path.expanduser(raw_dir))
    print(f"[preprocess] raw_dir: {raw_dir}")
    if not os.path.isdir(raw_dir):
        raise RuntimeError(f"raw_dir does not exist: {raw_dir}")

    all_raw = []
    all_raw += load_gc19_raw(raw_dir, max_shards=max_shards)
    all_raw += load_ac18_raw(raw_dir)

    if not all_raw:
        raise RuntimeError(
            f"No datasets found under {raw_dir}.\n"
            "Expected:\n"
            "  {raw_dir}/google/cell_a/instance_usage-*.json.gz\n"
            "  {raw_dir}/alibaba/machine_usage.csv"
        )

    print(f"\n[preprocess] {len(all_raw)} domains loaded. Applying small-sample split …\n")
    splits = []
    for ds in all_raw:
        sp = small_sample_split(
            ds, target_len=target_len, seq_len=seq_len, horizon=horizon
        )
        splits.append(sp)
        print(f"  {sp['name']:<20}  "
              f"tgt_tr={len(sp['X_tgt_tr']):>4}  "
              f"src={len(sp['X_src']):>5}  "
              f"test={len(sp['X_tgt_te']):>4}")
    return splits
