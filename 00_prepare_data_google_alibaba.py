"""
Step 0 (cross-domain): Google → Alibaba transfer
================================================
Source domain: Google Cluster Trace 2019 `instance_usage` (or synthetic Google).
Target domain: Alibaba Cluster Trace 2018 (same target construction as 00_prepare_data).

Writes to data/processed_google_alibaba/ by default so the within-domain
pipeline in data/processed/ is untouched.

Run training with:
  python 01_train_all_models.py --processed-dir data/processed_google_alibaba
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from alibaba_io import load_machine_meta, load_machine_usage
from google_io import load_google_instance_usage, synthetic_google_source
from prepare_common import (
    FEATURE_COLS,
    FAULT_NAMES,
    N_CLASSES,
    WINDOW_SIZE,
    WINDOW_STEP,
    compute_thresholds,
    make_windows,
    node_types_from_windows,
)

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
DEFAULT_PROC_CROSS = BASE_DIR / "data" / "processed_google_alibaba"


def build_alibaba_target(
    meta: Optional[pd.DataFrame], usage: pd.DataFrame, seed: int = 42
) -> pd.DataFrame:
    """Same target-side split logic as 00_prepare_data.build_domains (target half only)."""
    if meta is not None:
        meta_last = (
            meta.sort_values("time_stamp")
            .groupby("machine_id")
            .last()
            .reset_index()[["machine_id", "failure_domain_1", "cpu_num", "mem_size"]]
        )
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
    df = df.sort_values(["machine_id", "time_stamp"]).reset_index(drop=True)

    unique_fd = sorted(df["failure_domain_1"].unique())
    split = max(1, len(unique_fd) // 2)
    src_fd, tgt_fd = unique_fd[:split], unique_fd[split:]
    if not tgt_fd:
        tgt_fd = src_fd
        src_m = df["machine_id"].unique()
        np.random.default_rng(seed).shuffle(src_m)
        split_m = int(len(src_m) * 0.6)
        tgt = df[df["machine_id"].isin(src_m[split_m:])].copy()
    else:
        tgt = df[df["failure_domain_1"].isin(tgt_fd)].copy()

    return tgt.reset_index(drop=True)


def save_processed(
    proc_dir: Path,
    src_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    thr: dict,
    data_tag: str,
) -> None:
    proc_dir.mkdir(parents=True, exist_ok=True)

    X_src_t, y_src_t, m_src = make_windows(src_df, thr)
    X_tgt_t, y_tgt_t, m_tgt = make_windows(tgt_df, thr)
    print(f"  Source (Google) windows: {X_src_t.shape}")
    print(f"  Target (Alibaba) windows: {X_tgt_t.shape}")

    if len(X_src_t) == 0 or len(X_tgt_t) == 0:
        raise RuntimeError(
            "Empty windows — increase sampling rows or check time series length per ID."
        )

    rng = np.random.default_rng(42)
    tgt_labeled = rng.random(len(X_tgt_t)) < 0.30

    X_src_flat = X_src_t.mean(axis=1)
    X_tgt_flat = X_tgt_t.mean(axis=1)
    tgt_node_types = node_types_from_windows(X_tgt_t)

    np.savez_compressed(
        proc_dir / "source_temporal.npz", X=X_src_t, y=y_src_t, machine=m_src
    )
    np.savez_compressed(
        proc_dir / "target_temporal.npz",
        X=X_tgt_t,
        y=y_tgt_t,
        machine=m_tgt,
        labeled=tgt_labeled,
        node_type=tgt_node_types,
    )

    def _to_parquet(X_flat, y, labeled, path):
        dfp = pd.DataFrame(X_flat, columns=FEATURE_COLS)
        dfp["label"] = y
        dfp["labeled"] = labeled
        dfp.to_parquet(path, index=False)

    _to_parquet(
        X_src_flat, y_src_t, np.ones(len(X_src_t), dtype=bool), proc_dir / "source_flat.parquet"
    )
    _to_parquet(X_tgt_flat, y_tgt_t, tgt_labeled, proc_dir / "target_flat.parquet")

    def _dist(y):
        vals, cnts = np.unique(y, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, cnts)}

    meta_info = {
        "transfer_setup": "google_to_alibaba",
        "data_source_tag": data_tag,
        "window_size": WINDOW_SIZE,
        "window_step": WINDOW_STEP,
        "n_features": len(FEATURE_COLS),
        "feature_cols": FEATURE_COLS,
        "n_classes": N_CLASSES,
        "fault_names": FAULT_NAMES,
        "n_src_windows": int(len(X_src_t)),
        "n_tgt_windows": int(len(X_tgt_t)),
        "tgt_labeled_pct": float(tgt_labeled.mean() * 100),
        "thresholds": thr,
        "src_label_dist": _dist(y_src_t),
        "tgt_label_dist": _dist(y_tgt_t),
    }
    with open(proc_dir / "meta.json", "w") as f:
        json.dump(meta_info, f, indent=2)

    print("\n  Source (Google) fault distribution:")
    for k, v in _dist(y_src_t).items():
        print(f"    {FAULT_NAMES[k]:>16s} (class {k}): {v:>6,}  ({v/len(y_src_t)*100:.1f}%)")
    print("\n  Target (Alibaba) fault distribution:")
    for k, v in _dist(y_tgt_t).items():
        print(f"    {FAULT_NAMES[k]:>16s} (class {k}): {v:>6,}  ({v/len(y_tgt_t)*100:.1f}%)")


def main():
    ap = argparse.ArgumentParser(description="Google → Alibaba cross-domain prep")
    ap.add_argument(
        "--processed-dir",
        type=str,
        default=str(DEFAULT_PROC_CROSS),
        help="Output directory for npz/parquet/meta.json",
    )
    ap.add_argument(
        "--google-root",
        type=str,
        default=str(RAW_DIR / "google"),
        help="Root folder containing cell_*/instance_usage*.parquet",
    )
    ap.add_argument(
        "--google-max-rows",
        type=int,
        default=400_000,
        help="Max rows after concatenating Google shards",
    )
    ap.add_argument(
        "--alibaba-rows",
        type=int,
        default=500_000,
        help="Sample size for Alibaba machine_usage.csv",
    )
    args = ap.parse_args()
    proc_dir = Path(args.processed_dir)
    google_root = Path(args.google_root)

    print("=" * 65)
    print("  Step 0b: Google (source) → Alibaba (target)")
    print("=" * 65)

    print("\n[1/4] Loading Google as SOURCE ...")
    src_df = load_google_instance_usage(
        google_root, max_rows=args.google_max_rows, seed=43
    )
    if src_df is None:
        print("  Falling back to synthetic Google-like source.")
        src_df = synthetic_google_source()
        g_tag = "synthetic_google"
    else:
        g_tag = "real_google_parquet"

    print("\n[2/4] Loading Alibaba as TARGET ...")
    meta = load_machine_meta(RAW_DIR)
    usage = load_machine_usage(RAW_DIR, target_rows=args.alibaba_rows)
    if usage is None:
        print("  No Alibaba usage — generating synthetic Alibaba TARGET (Alibaba-like schema).")
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "prep00", BASE_DIR / "00_prepare_data.py"
        )
        prep00 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prep00)
        _, tgt_df = prep00.generate_synthetic()
        a_tag = "synthetic_alibaba_target"
    else:
        tgt_df = build_alibaba_target(meta, usage)
        a_tag = "real_alibaba"

    print(f"  Google rows: {len(src_df):,}  machines: {src_df['machine_id'].nunique():,}")
    print(f"  Alibaba rows: {len(tgt_df):,}  machines: {tgt_df['machine_id'].nunique():,}")

    print("\n[3/4] Joint thresholds (source + target) ...")
    full_df = pd.concat([src_df, tgt_df], ignore_index=True)
    thr = compute_thresholds(full_df)
    print(f"  {thr}")

    print("\n[4/4] Windows + save ...")
    data_tag = f"{g_tag}+{a_tag}"
    save_processed(proc_dir, src_df, tgt_df, thr, data_tag)

    print(f"\n  Saved → {proc_dir}")
    print("  Train with:")
    print(f"    python 01_train_all_models.py --processed-dir {proc_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
