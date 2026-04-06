"""
Load Google Cluster Trace 2019 `instance_usage` shards.

Supported on disk (same as pivot/01_download_datasets.py):

  • Parquet (after conversion):
      data/raw/google/cell_*/instance_usage-000000000000.parquet
  • Raw download (JSON Lines, gzip) — **no conversion step needed**:
      data/raw/google/cell_*/instance_usage-000000000000.json.gz

Maps to the 6 Alibaba-aligned feature columns for cross-domain transfer.
Disk / network are not in instance_usage; we use deterministic proxies so
temporal structure and multi-class rules remain defined.
"""

from __future__ import annotations

import gzip
import json
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


def _usage_dict_scalar(val, key: str) -> float:
    """Extract cpus or memory from Google nested dict / JSON string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    if isinstance(val, dict):
        v = val.get(key)
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0
    if isinstance(val, str) and val.strip():
        try:
            d = json.loads(val)
            if isinstance(d, dict):
                v = d.get(key)
                return float(v) if v is not None else 0.0
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.0
    return 0.0


def _series_from_usage_column(df: pd.DataFrame, col: str, key: str) -> pd.Series:
    """Column may hold dicts (JSON load) or flat numeric (parquet flatten)."""
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=np.float64)
    s = df[col]
    first = None
    for v in s.head(20):
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            first = v
            break
    if first is None:
        return pd.Series(0.0, index=df.index, dtype=np.float64)
    if isinstance(first, (int, float)) or (
        isinstance(first, str) and first.replace(".", "").replace("-", "").isdigit()
    ):
        return pd.to_numeric(s, errors="coerce").fillna(0.0)
    return s.apply(lambda x: _usage_dict_scalar(x, key))


def google_instance_usage_to_canonical(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    One row per usage interval → canonical machine_usage-like schema.

    Google 2019 JSON has nested ``average_usage`` / ``maximum_usage`` dicts with
    ``cpus`` and ``memory`` keys (see pivot/data/raw/google/cell_a/*.json.gz).
    Some parquet exports use flattened ``average_usage_cpus`` style names.
    """
    # Series id: prefer physical machine_id when present (matches pivot layout)
    if "machine_id" in df.columns:
        def _fmt_mid(x):
            if pd.isna(x):
                return np.nan
            try:
                return str(int(float(x)))
            except (ValueError, TypeError):
                s = str(x).strip()
                return s if s else np.nan

        physical = df["machine_id"].map(_fmt_mid)
        if "collection_id" in df.columns and "instance_index" in df.columns:
            fb = df["collection_id"].astype(str) + "_" + df["instance_index"].astype(str)
            mid_str = physical.where(physical.notna(), fb)
        else:
            mid_str = physical.fillna("unknown")
    elif "collection_id" in df.columns and "instance_index" in df.columns:
        mid_str = df["collection_id"].astype(str) + "_" + df["instance_index"].astype(str)
    else:
        raise ValueError(
            "Google instance_usage needs machine_id or (collection_id + instance_index). "
            f"Got: {list(df.columns)[:30]}"
        )

    if "start_time" in df.columns:
        ts = pd.to_numeric(df["start_time"], errors="coerce")
    else:
        ts = pd.Series(np.arange(len(df), dtype=float))

    # CPU / memory: flat columns (parquet) OR nested dicts (raw JSON)
    if "average_usage_cpus" in df.columns:
        cpu_raw = pd.to_numeric(df["average_usage_cpus"], errors="coerce").fillna(0.0)
    elif "average_usage" in df.columns:
        cpu_raw = _series_from_usage_column(df, "average_usage", "cpus")
    elif "random_sample_usage" in df.columns:
        cpu_raw = _series_from_usage_column(df, "random_sample_usage", "cpus")
    else:
        raise ValueError(
            "No CPU usage (expected average_usage_cpus or average_usage.cpus). "
            f"Columns: {list(df.columns)[:25]}"
        )

    if "average_usage_memory" in df.columns:
        mem_raw = pd.to_numeric(df["average_usage_memory"], errors="coerce").fillna(0.0)
    elif "average_usage" in df.columns:
        mem_raw = _series_from_usage_column(df, "average_usage", "memory")
    elif "random_sample_usage" in df.columns:
        mem_raw = _series_from_usage_column(df, "random_sample_usage", "memory")
    else:
        raise ValueError(
            "No memory usage (expected average_usage_memory or average_usage.memory)."
        )

    cpu_pct = _cpu_to_percent(cpu_raw)
    mem_pct = _mem_to_percent(mem_raw)
    mem_gps = (mem_pct * 0.65 + rng.normal(0, 2.0, len(df))).clip(0, 100)

    if "maximum_usage_cpus" in df.columns and "average_usage_cpus" in df.columns:
        d = (
            pd.to_numeric(df["maximum_usage_cpus"], errors="coerce").fillna(0)
            - pd.to_numeric(df["average_usage_cpus"], errors="coerce").fillna(0)
        ).abs()
        disk_pct = (d / (d.quantile(0.99) + 1e-6) * 100.0).clip(0, 100)
    elif "maximum_usage" in df.columns and "average_usage" in df.columns:
        max_c = _series_from_usage_column(df, "maximum_usage", "cpus")
        avg_c = _series_from_usage_column(df, "average_usage", "cpus")
        d = (max_c - avg_c).abs()
        disk_pct = (d / (d.quantile(0.99) + 1e-6) * 100.0).clip(0, 100)
    else:
        disk_pct = (cpu_pct * 0.3 + rng.uniform(0, 12, len(df))).clip(0, 100)

    net_in = (cpu_pct * 0.22 + rng.uniform(0, 18, len(df))).clip(0, 100)
    net_out = (cpu_pct * 0.18 + rng.uniform(0, 14, len(df))).clip(0, 100)

    out = pd.DataFrame(
        {
            "machine_id": mid_str.values,
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


def _read_json_gz_shard(path: Path, max_lines_per_shard: int = 400_000) -> Optional[pd.DataFrame]:
    """Read newline-delimited JSON from a .json.gz shard (Google trace format)."""
    rows: List[dict] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines_per_shard:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        print(f"  Skip {path.name}: {e}")
        return None
    if not rows:
        return None
    return pd.DataFrame(rows)


def _discover_google_shards(google_root: Path) -> tuple[List[Path], str]:
    """
    Return (paths, kind) where kind is 'parquet' or 'json_gz'.
    """
    if not google_root.is_dir():
        return [], ""

    parq = sorted(set(google_root.glob("**/instance_usage*.parquet")))
    if parq:
        return parq, "parquet"

    gz = sorted(set(google_root.glob("**/instance_usage*.json.gz")))
    if gz:
        return gz, "json_gz"

    # Helpful debug: show what is actually under google_root
    any_files = list(google_root.rglob("*"))[:30]
    names = [p.relative_to(google_root) for p in any_files if p.is_file()]
    if names:
        print(f"  (Found files under google/, first few: {[str(x) for x in names[:8]]})")
    return [], ""


def load_google_instance_usage(
    google_root: Path,
    max_rows: int = 400_000,
    seed: int = 43,
    max_lines_per_json_shard: int = 350_000,
) -> Optional[pd.DataFrame]:
    """
    Load instance_usage shards: tries .parquet first, then .json.gz (raw download).
    """
    paths, kind = _discover_google_shards(google_root)
    if not paths:
        print(f"  No instance_usage*.parquet or instance_usage*.json.gz under {google_root}")
        return None

    print(f"  Found {len(paths)} Google shard(s) ({kind}): e.g. {paths[0].name}")

    rng = np.random.default_rng(seed)
    chunks: List[pd.DataFrame] = []
    total = 0
    for p in paths:
        try:
            if kind == "parquet":
                df = pd.read_parquet(p)
            else:
                df = _read_json_gz_shard(p, max_lines_per_shard=max_lines_per_json_shard)
                if df is None or len(df) == 0:
                    continue
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
        f"{out['machine_id'].nunique():,} machines (physical id or collection_index)"
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
