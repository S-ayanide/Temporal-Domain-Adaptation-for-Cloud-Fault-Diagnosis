"""
data_loader.py
==============
Loads Google Cluster Trace 2019 and Alibaba 2017 for workload prediction.

Google shards are in:
    data/raw/google/cell_a/instance_usage-000000000000.json.gz
    data/raw/google/cell_b/instance_usage-000000000000.json.gz
    ...  (23 shards across multiple cell_* subdirs)

Accepts both:
  - instance_usage-*.json.gz   (raw Google Cluster Trace download) ← YOUR FORMAT
  - instance_usage-*.parquet   (converted format)

Alibaba files are in:
    data/raw/machine_usage.csv
    data/raw/machine_meta.csv   (not used — metadata only)
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Alibaba cluster trace `machine_usage` — no header in public dumps.
# v2018 has 9 columns (includes mkpi); older releases often have 8.
_ALIBABA_USAGE_COLS_9 = [
    "machine_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "mem_gps",
    "mkpi",
    "net_in",
    "net_out",
    "disk_io_percent",
]
_ALIBABA_USAGE_COLS_8 = [
    "machine_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "mem_gps",
    "net_in",
    "net_out",
    "disk_io_percent",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_nested(val, key: str) -> float:
    """Google sometimes stores average_usage as a nested dict or JSON string."""
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


def _get_cpu(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Try every known column name / nesting for CPU usage.
    Google 2019 json.gz uses nested dicts: average_usage.cpus
    """
    # Flat columns (parquet export or already flattened)
    for col in ["average_usage.cpus", "average_usage_cpus",
                "random_sample_usage.cpus", "random_sample_usage_cpus",
                "mean_cpu_usage_rate", "cpu_rate"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() > 0:
                return s

    # Nested dict column (raw json.gz format)
    for col in ["average_usage", "random_sample_usage"]:
        if col in df.columns:
            s = df[col].apply(lambda x: _extract_nested(x, "cpus"))
            s = pd.to_numeric(s, errors="coerce")
            if s.notna().sum() > 0:
                return s

    return None


def _get_job_id(df: pd.DataFrame) -> pd.Series:
    if "collection_id" in df.columns and "instance_index" in df.columns:
        return (df["collection_id"].astype(str) + "_"
                + df["instance_index"].astype(str))
    if "machine_id" in df.columns:
        return df["machine_id"].astype(str)
    if "collection_id" in df.columns:
        return df["collection_id"].astype(str)
    return pd.Series(["job_0"] * len(df), index=df.index)


def _get_ts(df: pd.DataFrame) -> pd.Series:
    for col in ["start_time", "end_time", "time_stamp", "timestamp", "ts"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.arange(len(df), dtype=float), index=df.index)


def _df_to_series(df: pd.DataFrame) -> List[np.ndarray]:
    """Convert one DataFrame shard → list of per-job CPU time series."""
    cpu = _get_cpu(df)
    if cpu is None:
        return []

    job = _get_job_id(df)
    ts  = _get_ts(df)

    tmp = pd.DataFrame({"job": job, "ts": ts, "cpu": cpu})
    tmp = tmp.dropna(subset=["cpu"])
    if len(tmp) == 0:
        return []

    # Google reports CPU as fraction 0-1; convert to percent 0-100
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


# ─── Shard reader: parquet OR json.gz ─────────────────────────────────────────

def _read_shard(path: Path, kind: str,
                max_json_lines: int = 400_000) -> Optional[pd.DataFrame]:
    """Read one shard file regardless of format."""
    if kind == "parquet":
        return pd.read_parquet(path)

    # Raw Google download: newline-delimited JSON inside gzip
    rows = []
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_json_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        print(f"  [skip] {path.name}: {e}")
        return None
    return pd.DataFrame(rows) if rows else None


# ─── Google loader ────────────────────────────────────────────────────────────

def load_google(
    google_root: str | Path,
    max_series: int = 5000,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Load all instance_usage shards from google_root (scanned recursively).

    Handles both:
      data/raw/google/cell_a/instance_usage-*.json.gz   ← your format
      data/raw/google/cell_a/instance_usage-*.parquet

    Returns a list of CPU time series (float32, values 0–100).
    """
    root = Path(google_root)
    if not root.exists():
        raise FileNotFoundError(f"Google root not found: {root.resolve()}")

    # Try parquet first, then json.gz
    files = sorted(root.rglob("instance_usage*.parquet"))
    kind  = "parquet"
    if not files:
        files = sorted(root.rglob("instance_usage*.json.gz"))
        kind  = "json.gz"
    if not files:
        found = [p.name for p in root.rglob("*") if p.is_file()][:15]
        raise FileNotFoundError(
            f"No instance_usage*.parquet or instance_usage*.json.gz found "
            f"under {root.resolve()}\nFiles found: {found}"
        )

    print(f"  Google: {len(files)} {kind} shard(s)  "
          f"(e.g. .../{files[0].parent.name}/{files[0].name})")

    # Peek first shard to show columns — crucial for debugging
    try:
        sample = _read_shard(files[0], kind)
        if sample is not None:
            print(f"  First shard columns : {list(sample.columns)}")
            print(f"  First shard shape   : {sample.shape}")
            # Show a couple of raw values so we can verify CPU extraction
            cpu = _get_cpu(sample)
            if cpu is not None:
                p99 = float(cpu.dropna().quantile(0.99))
                print(f"  CPU col p99 (raw)   : {p99:.4f}  "
                      f"({'fraction — will ×100' if p99 <= 2.0 else 'already percent'})")
            else:
                print("  WARNING: could not find CPU column — check column names above")
    except Exception as e:
        print(f"  Could not peek first shard: {e}")

    all_series: List[np.ndarray] = []
    for path in files:
        try:
            df = _read_shard(path, kind)
            if df is None:
                continue
        except Exception as e:
            print(f"  [skip] {path.parent.name}/{path.name}: {e}")
            continue

        shard_series = _df_to_series(df)
        all_series.extend(shard_series)
        print(f"  {path.parent.name}/{path.name}: "
              f"{len(shard_series)} series  (total: {len(all_series)})")

        if len(all_series) >= max_series * 3:
            break

    if not all_series:
        raise RuntimeError(
            "No Google series extracted. Column names in your shards don't match "
            "the known patterns.\n"
            "Run:  python data_loader.py --inspect --google <path>\n"
            "to see raw column names and a sample row."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(all_series)
    all_series = all_series[:max_series]

    lengths = [len(s) for s in all_series]
    print(f"\n  Google summary : {len(all_series)} series | "
          f"median length {int(np.median(lengths))} | "
          f"CPU range [{all_series[0].min():.1f}, {all_series[0].max():.1f}]")
    return all_series


# Alibaba 2018 container_usage.csv actual layout (11 columns, leading empty col):
# col0: empty | col1: container_id | col2: time_stamp | col3: cpu_util_percent
# col4: mem_util_percent | col5: cpi | col6: mpki | col7: net_in
# col8: net_out | col9: block_io_percent | col10: unknown
# We name col0 "_drop" and drop it after loading.
_ALIBABA_CONTAINER_COLS_11 = [
    "_drop",
    "container_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "cpi",
    "mpki",
    "net_in",
    "net_out",
    "block_io_percent",
    "_extra",
]
# Fallback for files without leading empty col (true 9-col layout)
_ALIBABA_CONTAINER_COLS_9 = [
    "container_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "cpi",
    "mpki",
    "net_in",
    "net_out",
    "block_io_percent",
]
_ALIBABA_CONTAINER_COLS_8 = [
    "container_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "cpi",
    "net_in",
    "net_out",
    "block_io_percent",
]


# ─── Alibaba 2017 loader ──────────────────────────────────────────────────────

def load_alibaba(
    alibaba_root: str | Path,
    max_series: int = 5000,
    nrows: int = 5_000_000,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Load Alibaba 2017 container_usage.csv (preferred) or machine_usage.csv.

    The CWPDDA paper (Wang et al., Euro-Par 2025) uses container-level Alibaba
    data — containers are short-lived jobs with few data points ("small sample
    condition"), making transfer from Google essential.  Machine-level data has
    median 6844 points per machine — too much for a plain LSTM to need transfer.

    Priority:
      1. container_usage.csv  (Alibaba 2017 container trace — correct for paper)
      2. container_usage.csv.gz
      3. machine_usage.csv    (fallback — wrong granularity for CWPDDA)

    Returns a list of CPU time series (float32, values 0–100).
    """
    root = Path(alibaba_root)
    if not root.exists():
        raise FileNotFoundError(f"Alibaba root not found: {root.resolve()}")

    # Try container_usage first (correct for CWPDDA), fall back to machine_usage
    container_candidates = (
        list(root.glob("container_usage.csv"))
        + list(root.glob("container_usage.csv.gz"))
        + list(root.rglob("container_usage.csv"))
        + list(root.rglob("container_usage.csv.gz"))
    )
    machine_candidates = (
        list(root.glob("machine_usage.csv"))
        + list(root.glob("machine_usage.csv.gz"))
        + list(root.rglob("machine_usage.csv"))
        + list(root.rglob("machine_usage.csv.gz"))
    )

    def _dedup(paths: List[Path]) -> List[Path]:
        seen: set[str] = set()
        out: List[Path] = []
        for p in paths:
            k = str(p.resolve())
            if k not in seen:
                seen.add(k)
                out.append(p)
        return out

    container_candidates = _dedup(container_candidates)
    machine_candidates   = _dedup(machine_candidates)

    use_container = bool(container_candidates)
    path = container_candidates[0] if use_container else (
           machine_candidates[0]   if machine_candidates else None)

    if path is None:
        found = [p.name for p in root.rglob("*.csv")][:15]
        raise FileNotFoundError(
            f"Neither container_usage.csv nor machine_usage.csv found under "
            f"{root.resolve()}\nCSV files found: {found}"
        )

    if use_container:
        print(f"\n  Alibaba: loading container_usage from "
              f"{path.relative_to(root.parent)}")
        df = _read_alibaba_csv(path, nrows, is_container=True)
        id_col = _pick_col(df, ["container_id", "container", "cid"])
    else:
        print(f"\n  Alibaba: loading machine_usage from "
              f"{path.relative_to(root.parent)}")
        print(f"  [WARN] machine_usage.csv has median ~6844 pts/machine — this is "
              f"the wrong granularity for CWPDDA (paper uses container_usage.csv).")
        df = _read_alibaba_csv(path, nrows, is_container=False)
        id_col = _pick_col(df, ["machine_id", "machineID", "machine"])

    print(f"  Alibaba columns : {list(df.columns)}")
    print(f"  Alibaba shape   : {df.shape}")

    cpu_col = _pick_col(df, ["cpu_util_percent", "cpu_util", "cpu"])
    ts_col  = _pick_col(df, ["time_stamp", "timestamp", "ts", "time"])

    if cpu_col is None:
        raise RuntimeError(
            f"Cannot find CPU column.\nColumns present: {list(df.columns)}")
    if id_col is None:
        raise RuntimeError(
            f"Cannot find ID column.\nColumns present: {list(df.columns)}")

    df[cpu_col] = pd.to_numeric(df[cpu_col], errors="coerce")
    df = df[df[cpu_col] >= 0].copy()         # drop -1 sentinels
    # Container CPU is stored as % of container quota — values commonly exceed 100.
    # Do NOT clip to 100; per-series min-max normalisation in preprocess.py handles scaling.
    # Machine-level data is already in [0,100] so this is safe for both.
    df[cpu_col] = df[cpu_col].clip(lower=0).astype(np.float32)

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

    # The CWPDDA paper uses "containers meeting small sample condition" — short-lived
    # containers with few data points.  The 2018 container trace has median ~1168 pts;
    # the 2017 trace had ~144 pts (12h × 5-min intervals).
    #
    # For both machine_usage (fallback) and container_usage (long series): split into
    # 144-point chunks so each chunk resembles one 12-hour container lifetime.
    # This matches the paper's regime where transfer from Google is actually needed.
    median_len = float(np.median([len(s) for s in all_series]))
    if median_len > 300:
        chunk_len = 96   # 8h at 5-min sampling; fits MCTL's <=100-pt few-shot filter
        chunked: List[np.ndarray] = []
        for s in all_series:
            for start in range(0, len(s) - chunk_len + 1, chunk_len):
                chunked.append(s[start : start + chunk_len])
        if chunked:
            src = "container" if use_container else "machine"
            all_series = chunked
            print(f"  [short-series] Split into {len(all_series)} {src} chunks of "
                  f"{chunk_len} pts (12h container lifetime regime).")

    rng.shuffle(all_series)
    all_series = all_series[:max_series]

    lengths = [len(s) for s in all_series]
    print(f"\n  Alibaba summary : {len(all_series)} series | "
          f"median length {int(np.median(lengths))} | "
          f"CPU range [{all_series[0].min():.1f}, {all_series[0].max():.1f}]")
    return all_series


def _read_alibaba_csv(path: Path, nrows: int,
                       is_container: bool = False) -> pd.DataFrame:
    """
    Alibaba CSVs typically have no header — assign column names by position.
    Only treat as headered CSV if the first cell is literally the id column name.
    """
    peek = pd.read_csv(path, nrows=1, header=None)
    ncols = len(peek.columns)
    first_cell = str(peek.iloc[0, 0]).strip().lower()

    if is_container:
        header_names = ["container_id", "cid", "container"]
        col11 = _ALIBABA_CONTAINER_COLS_11
        col9  = _ALIBABA_CONTAINER_COLS_9
        col8  = _ALIBABA_CONTAINER_COLS_8
    else:
        header_names = ["machine_id"]
        col11 = None
        col9  = _ALIBABA_USAGE_COLS_9
        col8  = _ALIBABA_USAGE_COLS_8

    if first_cell in header_names:
        df_h = pd.read_csv(path, nrows=nrows, low_memory=False)
        if _pick_col(df_h, ["cpu_util_percent", "cpu_util", "cpu"]) is not None:
            return df_h

    if ncols == 11 and col11 is not None:
        names = col11
    elif ncols == 9:
        names = col9
    elif ncols == 8:
        names = col8
    else:
        base = col9
        names = [base[i] if i < len(base) else f"col_{i}" for i in range(ncols)]

    df = pd.read_csv(path, nrows=nrows, header=None, names=names, low_memory=False)

    # Drop the leading empty column and any extra trailing columns
    for drop_col in ["_drop", "_extra"]:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    return df


# Keep old name as alias for any external callers
_read_machine_usage = _read_alibaba_csv


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lc = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lc:
            return lc[c.lower()]
    return None


# ─── CLI sanity check / inspect ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    # --inspect: show raw columns + first row of first Google shard
    if "--inspect" in args:
        i = args.index("--inspect")
        if "--google" in args:
            g = args[args.index("--google") + 1]
        elif len(args) > i + 1 and not args[i + 1].startswith("--"):
            g = args[i + 1]
        else:
            g = "data/raw/google"

        root = Path(g)
        files = sorted(root.rglob("instance_usage*.parquet"))
        kind  = "parquet"
        if not files:
            files = sorted(root.rglob("instance_usage*.json.gz"))
            kind  = "json.gz"

        if not files:
            print(f"No instance_usage shards found under {root}")
            sys.exit(1)

        print(f"\nFile  : {files[0]}")
        print(f"Format: {kind}")
        df = _read_shard(files[0], kind)
        print(f"Shape : {df.shape}")
        print(f"Cols  : {list(df.columns)}")
        print(f"\nFirst row:")
        print(df.iloc[0].to_dict())
        sys.exit(0)

    google_root  = args[0] if len(args) > 0 else "data/raw/google"
    alibaba_root = args[1] if len(args) > 1 else "data/raw"

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