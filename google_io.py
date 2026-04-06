"""
Load Google Cluster Trace 2019 `instance_usage` shards (parquet from JSON.gz conversion).

Expected layout (same as pivot/01_download_datasets.py):
  data/raw/google/cell_*/instance_usage-000000000000.parquet

Maps to the 6 Alibaba-aligned feature columns for cross-domain transfer.
Disk / network are not in instance_usage; we use deterministic proxies so
temporal structure and multi-class rules remain defined.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def _cpu_to_percent(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    mx = float(s.quantile(0.99)) if len(s) else 0.0
    if mx <= 1.5:
        return (s * 100.0).clip(0, 100)
    denom = max(mx, 1e-6)
    return (s / denom * 100.0).clip(0, 100)


def _mem_to_percent(series: pd.Series) -> pd.Series:
    return _cpu_to_percent(series)


def google_instance_usage_to_canonical(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One row per usage interval → canonical machine_usage-like schema."""
    if "collection_id" not in df.columns or "instance_index" not in df.columns:
        raise ValueError(
            "Google instance_usage needs collection_id and instance_index columns. "
            f"Got: {list(df.columns)[:25]}"
        )

    mid = df["collection_id"].astype(str) + "_" + df["instance_index"].astype(str)

    if "start_time" in df.columns:
        ts = pd.to_numeric(df["start_time"], errors="coerce")
    else:
        ts = pd.Series(np.arange(len(df), dtype=float))

    cpu_col = None
    for c in ("average_usage_cpus", "maximum_usage_cpus", "random_sample_usage_cpus"):
        if c in df.columns:
            cpu_col = c
            break
    if cpu_col is None:
        raise ValueError("No CPU usage column found in Google instance_usage.")

    mem_col = None
    for c in (
        "average_usage_memory",
        "maximum_usage_memory",
        "random_sample_usage_memory",
    ):
        if c in df.columns:
            mem_col = c
            break
    if mem_col is None:
        raise ValueError("No memory usage column found in Google instance_usage.")

    cpu_pct = _cpu_to_percent(df[cpu_col])
    mem_pct = _mem_to_percent(df[mem_col])
    mem_gps = (mem_pct * 0.65 + rng.normal(0, 2.0, len(df))).clip(0, 100)

    if (
        "maximum_usage_cpus" in df.columns
        and "average_usage_cpus" in df.columns
    ):
        d = (
            pd.to_numeric(df["maximum_usage_cpus"], errors="coerce").fillna(0)
            - pd.to_numeric(df["average_usage_cpus"], errors="coerce").fillna(0)
        ).abs()
        disk_pct = (d / (d.quantile(0.99) + 1e-6) * 100.0).clip(0, 100)
    else:
        disk_pct = (cpu_pct * 0.3 + rng.uniform(0, 12, len(df))).clip(0, 100)

    net_in = (cpu_pct * 0.22 + rng.uniform(0, 18, len(df))).clip(0, 100)
    net_out = (cpu_pct * 0.18 + rng.uniform(0, 14, len(df))).clip(0, 100)

    out = pd.DataFrame(
        {
            "machine_id": mid.values,
            "time_stamp": ts.values,
            "cpu_util_percent": cpu_pct.values.astype(np.float64),
            "mem_util_percent": mem_pct.values.astype(np.float64),
            "mem_gps": mem_gps.astype(np.float64),
            "net_in": net_in.astype(np.float64),
            "net_out": net_out.astype(np.float64),
            "disk_io_percent": disk_pct.values.astype(np.float64),
        }
    )
    out = out.dropna(subset=["time_stamp"])
    out = out.sort_values(["machine_id", "time_stamp"]).reset_index(drop=True)
    return out


def load_google_instance_usage(
    google_root: Path,
    max_rows: int = 400_000,
    seed: int = 43,
    shard_glob: str = "**/instance_usage*.parquet",
) -> Optional[pd.DataFrame]:
    """
    Load and concatenate parquet shards under google_root, subsample to max_rows.
    """
    paths: List[Path] = sorted(google_root.glob(shard_glob))
    if not paths:
        print(f"  No parquet files matching {shard_glob!r} under {google_root}")
        return None

    rng = np.random.default_rng(seed)
    chunks: List[pd.DataFrame] = []
    total = 0
    for p in paths:
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"  Skip {p.name}: {e}")
            continue
        if len(df) == 0:
            continue
        try:
            canon = google_instance_usage_to_canonical(df, rng)
        except ValueError as e:
            print(f"  Skip {p.name}: {e}")
            continue
        chunks.append(canon)
        total += len(canon)
        if total >= max_rows * 2:
            break

    if not chunks:
        return None

    out = pd.concat(chunks, ignore_index=True)
    if len(out) > max_rows:
        idx = rng.choice(len(out), size=max_rows, replace=False)
        out = out.iloc[np.sort(idx)].reset_index(drop=True)

    out = out.sort_values(["machine_id", "time_stamp"]).reset_index(drop=True)
    print(
        f"  Google canonical: {len(out):,} rows, "
        f"{out['machine_id'].nunique():,} pseudo-machines (collection_index)"
    )
    return out


def synthetic_google_source(
    n_machines: int = 180,
    n_rows_each: int = 55,
    seed: int = 43,
) -> pd.DataFrame:
    """Fallback when no Google parquet is present — shifted stats vs Alibaba synthetic."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_machines):
        mid = f"g_{m:05d}"
        for t in range(n_rows_each):
            cpu = float(rng.uniform(45, 100))
            mem = float(rng.uniform(25, 85))
            disk = float(rng.uniform(10, 75))
            net = float(rng.uniform(15, 70))
            rows.append(
                {
                    "machine_id": mid,
                    "time_stamp": float(t * 60),
                    "cpu_util_percent": cpu,
                    "mem_util_percent": mem,
                    "mem_gps": float(np.clip(mem * 0.55 + rng.normal(0, 4), 0, 100)),
                    "net_in": float(np.clip(net * 0.9, 0, 100)),
                    "net_out": float(np.clip(net * 0.75, 0, 100)),
                    "disk_io_percent": disk,
                }
            )
    return pd.DataFrame(rows).sort_values(["machine_id", "time_stamp"]).reset_index(drop=True)
