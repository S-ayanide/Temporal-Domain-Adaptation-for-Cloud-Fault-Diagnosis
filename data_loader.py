"""
data_loader.py
==============
Loads raw data for the Google → Alibaba workload prediction replication.

Google:  instance_usage-*.parquet  (your 23 converted shards)
         Expected columns (Google 2019 schema):
           collection_id, instance_index, start_time,
           average_usage.cpus  OR  average_usage_cpus
           average_usage.memory OR average_usage_memory

Alibaba: data/raw/machine_usage.csv  (Alibaba 2017)
         Expected columns:
           machine_id, time_stamp, cpu_util_percent, ...

Both return a plain list of 1-D float32 numpy arrays (one per job/machine),
values in [0, 100]. Normalisation happens in preprocess.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_nested(val, key: str) -> float:
    """
    Google parquet sometimes stores average_usage as a dict-like object
    or as a JSON string rather than flattened columns.
    """
    if val is None:
        return np.nan
    if isinstance(val, float) and np.isnan(val):
        return np.nan
    if isinstance(val, dict):
        v = val.get(key)
        try:
            return float(v) if v is not None else np.nan
        except (TypeError, ValueError):
            return np.nan
    if isinstance(val, str) and val.strip():
        try:
            d = json.loads(val)
            if isinstance(d, dict):
                v = d.get(key)
                return float(v) if v is not None else np.nan
        except Exception:
            pass
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def _get_cpu_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Try every known column name / nesting pattern for CPU usage.
    Returns a Series of raw CPU values (may be fractions 0-1 or percent 0-100).
    """
    for col in [
        "average_usage.cpus",
        "average_usage_cpus",
        "random_sample_usage.cpus",
        "random_sample_usage_cpus",
        "mean_cpu_usage_rate",
        "cpu_rate",
    ]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() > 0:
                return s

    # Nested dict column
    for col in ["average_usage", "random_sample_usage"]:
        if col in df.columns:
            s = df[col].apply(lambda x: _extract_nested(x, "cpus"))
            s = pd.to_numeric(s, errors="coerce")
            if s.notna().sum() > 0:
                return s

    return None


def _get_job_id(df: pd.DataFrame) -> pd.Series:
    """Build a job identifier series from whatever columns are available."""
    if "collection_id" in df.columns and "instance_index" in df.columns:
        return (
            df["collection_id"].astype(str) + "_"
            + df["instance_index"].astype(str)
        )
    if "machine_id" in df.columns:
        return df["machine_id"].astype(str)
    if "collection_id" in df.columns:
        return df["collection_id"].astype(str)
    return pd.Series(["job_0"] * len(df), index=df.index)


def _get_timestamp(df: pd.DataFrame) -> pd.Series:
    for col in ["start_time", "end_time", "time_stamp", "timestamp", "ts"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.arange(len(df), dtype=float), index=df.index)


def _shard_to_series(df: pd.DataFrame) -> List[np.ndarray]:
    """Convert one parquet shard DataFrame → list of per-job CPU series."""
    cpu = _get_cpu_series(df)
    if cpu is None:
        return []

    job_id = _get_job_id(df)
    ts     = _get_timestamp(df)

    tmp = pd.DataFrame({"job": job_id, "ts": ts, "cpu": cpu})
    tmp = tmp.dropna(subset=["cpu"])
    if len(tmp) == 0:
        return []

    # Auto-detect fraction vs percent (Google reports fractions 0-1)
    p99 = float(tmp["cpu"].quantile(0.99))
    if p99 <= 2.0:
        tmp["cpu"] = (tmp["cpu"] * 100.0).clip(0, 100)
    else:
        tmp["cpu"] = tmp["cpu"].clip(0, 100)

    out: List[np.ndarray] = []
    for _, grp in tmp.groupby("job", sort=False):
        grp = grp.sort_values("ts")
        arr = grp["cpu"].dropna().values.astype(np.float32)
        if len(arr) >= 10:
            out.append(arr)
    return out


# ─── Google loader ────────────────────────────────────────────────────────────

def load_google(
    google_root: str | Path,
    max_series: int = 5000,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Load all instance_usage-*.parquet shards from google_root (recursive).
    Returns a list of CPU time series, values in [0, 100].
    """
    root = Path(google_root)
    if not root.exists():
        raise FileNotFoundError(f"Google root not found: {root.resolve()}")

    files = sorted(root.rglob("instance_usage*.parquet"))
    if not files:
        found = [p.name for p in root.rglob("*") if p.is_file()][:15]
        raise FileNotFoundError(
            f"No instance_usage*.parquet found under {root.resolve()}\n"
            f"Files found: {found}"
        )

    print(f"  Google: {len(files)} parquet shard(s) found")

    # Print columns of first shard so you can verify
    try:
        sample = pd.read_parquet(files[0])
        print(f"  First shard columns : {list(sample.columns)}")
        print(f"  First shard shape   : {sample.shape}")
    except Exception as e:
        print(f"  Could not peek first shard: {e}")

    all_series: List[np.ndarray] = []
    for path in files:
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"  [skip] {path.name}: {e}")
            continue

        shard_series = _shard_to_series(df)
        all_series.extend(shard_series)
        print(f"  {path.name}: {len(shard_series)} series  "
              f"(total: {len(all_series)})")

        if len(all_series) >= max_series * 3:
            break

    if not all_series:
        raise RuntimeError(
            "No Google series extracted.\n"
            "The column names in your parquet don't match expected patterns.\n"
            "Run:  python data_loader.py --inspect\n"
            "to see what columns are actually in your first shard."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(all_series)
    all_series = all_series[:max_series]

    lengths = [len(s) for s in all_series]
    print(f"\n  Google summary : {len(all_series)} series | "
          f"median length {int(np.median(lengths))} | "
          f"CPU range [{all_series[0].min():.1f}, {all_series[0].max():.1f}]")
    return all_series


# ─── Alibaba 2017 loader ──────────────────────────────────────────────────────

def load_alibaba(
    alibaba_root: str | Path,
    max_series: int = 5000,
    nrows: int = 5_000_000,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Load Alibaba 2017 machine_usage.csv from alibaba_root.

    Handles two common variants:
      - WITH header row  (some releases)
      - WITHOUT header   (raw Alibaba 2017 release — 8 columns, no names)

    Returns a list of CPU time series, values in [0, 100].
    """
    root = Path(alibaba_root)
    if not root.exists():
        raise FileNotFoundError(f"Alibaba root not found: {root.resolve()}")

    candidates = (
        list(root.glob("machine_usage.csv"))
        + list(root.glob("machine_usage.csv.gz"))
        + list(root.rglob("machine_usage.csv"))
        + list(root.rglob("machine_usage.csv.gz"))
    )
    # deduplicate while preserving order
    seen = set()
    candidates = [p for p in candidates if not (p in seen or seen.add(p))]

    if not candidates:
        found = [p.name for p in root.rglob("*.csv")][:15]
        raise FileNotFoundError(
            f"machine_usage.csv not found under {root.resolve()}\n"
            f"CSV files found: {found}"
        )

    path = candidates[0]
    print(f"\n  Alibaba: loading {path}")

    df = _read_machine_usage(path, nrows=nrows)
    print(f"  Alibaba columns : {list(df.columns)}")
    print(f"  Alibaba shape   : {df.shape}")

    cpu_col = _pick_col(df, ["cpu_util_percent", "cpu_util", "cpu"])
    id_col  = _pick_col(df, ["machine_id", "machineID", "machine"])
    ts_col  = _pick_col(df, ["time_stamp", "timestamp", "ts", "time"])

    if cpu_col is None:
        raise RuntimeError(
            f"Cannot find CPU column in machine_usage.csv.\n"
            f"Columns present: {list(df.columns)}"
        )
    if id_col is None:
        raise RuntimeError(
            f"Cannot find machine_id column.\n"
            f"Columns present: {list(df.columns)}"
        )

    df[cpu_col] = pd.to_numeric(df[cpu_col], errors="coerce")
    df = df[df[cpu_col] >= 0].copy()           # drop -1 sentinels
    df[cpu_col] = df[cpu_col].clip(0, 100).astype(np.float32)

    if ts_col:
        df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce")
        df = df.sort_values([id_col, ts_col])
    else:
        df = df.sort_values(id_col)

    all_series: List[np.ndarray] = []
    for _, grp in df.groupby(id_col, sort=False):
        arr = grp[cpu_col].dropna().values.astype(np.float32)
        if len(arr) >= 10:
            all_series.append(arr)

    if not all_series:
        raise RuntimeError("No Alibaba series extracted. Check the CSV content.")

    rng = np.random.default_rng(seed)
    rng.shuffle(all_series)
    all_series = all_series[:max_series]

    lengths = [len(s) for s in all_series]
    print(f"\n  Alibaba summary : {len(all_series)} series | "
          f"median length {int(np.median(lengths))} | "
          f"CPU range [{all_series[0].min():.1f}, {all_series[0].max():.1f}]")
    return all_series


def _read_machine_usage(path: Path, nrows: int) -> pd.DataFrame:
    """
    Alibaba 2017 machine_usage.csv sometimes ships without a header row.
    Detect by checking if the first value is numeric.
    """
    peek = pd.read_csv(path, nrows=1, header=None)
    first_val = str(peek.iloc[0, 0])
    has_header = not first_val.replace(".", "").replace("-", "").lstrip("-").isdigit()

    if has_header:
        return pd.read_csv(path, nrows=nrows, low_memory=False)

    # No-header variant: 8-column schema from Alibaba 2017 public release
    cols_8  = ["machine_id", "time_stamp", "cpu_util_percent",
               "mem_util_percent", "mem_gps", "net_in", "net_out", "disk_io_percent"]
    ncols   = len(peek.columns)
    names   = cols_8[:ncols]
    return pd.read_csv(path, nrows=nrows, header=None,
                       names=names, low_memory=False)


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lc = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lc:
            return lc[c.lower()]
    return None


# ─── Sanity check / inspect ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # --inspect  just shows column names of the first Google shard
    if "--inspect" in sys.argv:
        google_root = next(
            (sys.argv[i + 1] for i, a in enumerate(sys.argv)
             if a == "--google"), "data/raw/google"
        )
        files = sorted(Path(google_root).rglob("instance_usage*.parquet"))
        if not files:
            print(f"No parquet files found under {google_root}")
            sys.exit(1)
        df = pd.read_parquet(files[0])
        print(f"\nFile   : {files[0]}")
        print(f"Shape  : {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst row:\n{df.iloc[0].to_dict()}")
        sys.exit(0)

    google_root  = sys.argv[1] if len(sys.argv) > 1 else "data/raw/google"
    alibaba_root = sys.argv[2] if len(sys.argv) > 2 else "data/raw"

    print("=" * 55)
    print(" Data loading sanity check")
    print("=" * 55)

    print("\n[Google]")
    g = load_google(google_root, max_series=200)

    print("\n[Alibaba]")
    a = load_alibaba(alibaba_root, max_series=200)

    print("\n" + "=" * 55)
    print(f" Google  : {len(g)} series loaded OK")
    print(f" Alibaba : {len(a)} series loaded OK")
    print("=" * 55)
