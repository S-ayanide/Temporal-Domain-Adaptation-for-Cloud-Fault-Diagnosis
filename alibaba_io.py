"""Alibaba Cluster Trace 2018 download + load helpers (shared)."""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ALIBABA_BASE = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces"
COL_NAMES_USAGE = [
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
COL_NAMES_META = [
    "machine_id",
    "time_stamp",
    "failure_domain_1",
    "failure_domain_2",
    "cpu_num",
    "mem_size",
    "status",
]


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


def load_machine_usage(
    raw_dir: Path, target_rows: int = 500_000, seed: int = 42
) -> Optional[pd.DataFrame]:
    def _sample(path: Path) -> pd.DataFrame:
        file_bytes = path.stat().st_size
        est_total = max(file_bytes // 70, target_rows * 2)
        prob = min(1.0, (target_rows * 1.5) / est_total)
        rng_s = np.random.default_rng(seed)
        print(
            f"  Sampling ~{target_rows:,} rows (p≈{prob:.4f}) from "
            f"{file_bytes/1e9:.2f} GB ..."
        )
        df = pd.read_csv(
            path,
            header=None,
            skiprows=lambda i: i > 0 and rng_s.random() > prob,
            names=COL_NAMES_USAGE,
            low_memory=False,
        )
        print(
            f"  Loaded {len(df):,} rows across {df['machine_id'].nunique():,} machines"
        )
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
    print("  machine_usage.csv not found.")
    return None
