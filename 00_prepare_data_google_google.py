"""
Step 0c: Within-Google domain adaptation (Google → Google)
===========================================================
Same *methodology* as Alibaba→Alibaba (`00_prepare_data.py`): two disjoint
sets of machines from the **same** trace, shared percentile thresholds,
temporal windows, ~30% labeled target.

Use this to evaluate TA-DATL vs DATL on **Google Cluster Trace 2019** without
the extreme cross-provider shift of Google→Alibaba.

Default output: data/processed_google_google/

Train:
  python 01_train_all_models.py --processed-dir data/processed_google_google
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

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
DEFAULT_PROC = BASE_DIR / "data" / "processed_google_google"


def split_google_by_machines(
    df: pd.DataFrame,
    frac_source: float = 0.6,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Disjoint source/target by machine_id (same idea as random machine split in Alibaba prep)."""
    mids = df["machine_id"].unique()
    if len(mids) < 2:
        raise RuntimeError(
            f"Need ≥2 distinct machine_id values for source/target split; got {len(mids)}."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(mids)
    n_src = max(1, int(len(mids) * frac_source))
    if n_src >= len(mids):
        n_src = len(mids) - 1
    src_m = set(mids[:n_src])
    tgt_m = set(mids[n_src:])
    src_df = df[df["machine_id"].isin(src_m)].copy()
    tgt_df = df[df["machine_id"].isin(tgt_m)].copy()
    return (
        src_df.sort_values(["machine_id", "time_stamp"]).reset_index(drop=True),
        tgt_df.sort_values(["machine_id", "time_stamp"]).reset_index(drop=True),
    )


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
    print(f"  Source windows (Google): {X_src_t.shape}")
    print(f"  Target windows (Google): {X_tgt_t.shape}")

    if len(X_src_t) == 0 or len(X_tgt_t) == 0:
        raise RuntimeError(
            "Empty windows — load more Google rows (see --google-max-rows) or "
            "check that many machines have ≥20 readings."
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
        X_src_flat,
        y_src_t,
        np.ones(len(X_src_t), dtype=bool),
        proc_dir / "source_flat.parquet",
    )
    _to_parquet(X_tgt_flat, y_tgt_t, tgt_labeled, proc_dir / "target_flat.parquet")

    def _dist(y):
        vals, cnts = np.unique(y, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, cnts)}

    meta_info = {
        "transfer_setup": "google_to_google",
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

    # Warn if eval set likely degenerate
    y_lb = y_tgt_t[tgt_labeled]
    if len(np.unique(y_lb)) < 2:
        print(
            "\n  WARNING: Labeled target evaluation has <2 classes → AUC will be nan. "
            "Increase data or adjust frac_source / seed."
        )

    print("\n  Source fault distribution (Google):")
    for k, v in _dist(y_src_t).items():
        print(f"    {FAULT_NAMES[k]:>16s} (class {k}): {v:>6,}  ({v/len(y_src_t)*100:.1f}%)")
    print("\n  Target fault distribution (Google):")
    for k, v in _dist(y_tgt_t).items():
        print(f"    {FAULT_NAMES[k]:>16s} (class {k}): {v:>6,}  ({v/len(y_tgt_t)*100:.1f}%)")


def main():
    ap = argparse.ArgumentParser(description="Within-Google domain adaptation prep")
    ap.add_argument(
        "--processed-dir",
        type=str,
        default=str(DEFAULT_PROC),
        help="Output directory",
    )
    ap.add_argument(
        "--google-root",
        type=str,
        default=str(RAW_DIR / "google"),
        help="Root with cell_*/instance_usage*.json.gz",
    )
    ap.add_argument(
        "--google-max-rows",
        type=int,
        default=600_000,
        help="Max rows loaded from Google (higher → more machines & windows)",
    )
    ap.add_argument(
        "--frac-source",
        type=float,
        default=0.6,
        help="Fraction of unique machine_ids assigned to source domain",
    )
    ap.add_argument("--seed", type=int, default=42, help="Machine split RNG seed")
    args = ap.parse_args()
    proc_dir = Path(args.processed_dir)
    google_root = Path(args.google_root)

    print("=" * 65)
    print("  Step 0c: Google → Google (within-trace domain split)")
    print("=" * 65)

    print("\n[1/3] Loading Google instance_usage ...")
    full = load_google_instance_usage(
        google_root, max_rows=args.google_max_rows, seed=43
    )
    if full is None:
        print("  No Google shards — synthetic Google (split across machines).")
        full = synthetic_google_source(n_machines=320, n_rows_each=60, seed=44)
        tag = "synthetic_google_split"
    else:
        tag = "real_google_split"

    print("\n[2/3] Splitting machines into source / target ...")
    src_df, tgt_df = split_google_by_machines(
        full, frac_source=args.frac_source, seed=args.seed
    )
    print(
        f"  Rows: source {len(src_df):,} ({src_df['machine_id'].nunique():,} machines) | "
        f"target {len(tgt_df):,} ({tgt_df['machine_id'].nunique():,} machines)"
    )

    print("\n[3/3] Joint thresholds + windows + save ...")
    full_df = pd.concat([src_df, tgt_df], ignore_index=True)
    thr = compute_thresholds(full_df)
    print(f"  Thresholds: {thr}")
    save_processed(proc_dir, src_df, tgt_df, thr, tag)

    print(f"\n  Saved → {proc_dir}")
    print(f"    python 01_train_all_models.py --processed-dir {proc_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
