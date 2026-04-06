"""
run.py
======
Single entry point. Runs the full Google → Alibaba transfer pipeline.

Usage:
    python run.py \
        --google  data/raw/google \
        --alibaba data/raw/alibaba \
        --device  cuda            \
        --out     results/

Output:
    results/metrics.json   — all model metrics
    results/table.txt      — human-readable table matching paper layout
    checkpoints/mctl.pt    — saved model weights
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--google",   default="data/raw/google",
                   help="Root dir of Google Cluster Trace shards")
    p.add_argument("--alibaba",  default="data/raw",
                   help="Root dir containing machine_usage.csv (Alibaba 2017)")
    p.add_argument("--out",      default="results",
                   help="Output directory for metrics and tables")
    p.add_argument("--ckpt",     default="checkpoints",
                   help="Directory to save model weights")
    p.add_argument("--device",   default="cpu",
                   help="'cpu', 'cuda', or 'cuda:0' etc.")
    p.add_argument("--max-google-series",  type=int, default=3000)
    p.add_argument("--max-alibaba-series", type=int, default=3000)
    p.add_argument("--window-size",  type=int, default=24)
    p.add_argument("--horizon",      type=int, default=1)
    p.add_argument("--hidden-dim",   type=int, default=128)
    p.add_argument("--n-layers",     type=int, default=3)
    p.add_argument("--kernel-size",  type=int, default=3)
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--stage1-epochs", type=int, default=50)
    p.add_argument("--stage2a-epochs",type=int, default=50)
    p.add_argument("--stage2b-epochs",type=int, default=50)
    p.add_argument("--batch-size",   type=int, default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--no-dtw",       action="store_true",
                   help="Skip DTW source selection (faster, less accurate)")
    p.add_argument("--skip-arima",   action="store_true",
                   help="Skip ARIMA baselines (slow on large test sets)")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    out_dir  = Path(args.out)
    ckpt_dir = Path(args.ckpt)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Step 1/4 — Load data")
    print("=" * 60)
    from data_loader import load_google, load_alibaba

    google_series  = load_google(args.google,  max_series=args.max_google_series)
    alibaba_series = load_alibaba(args.alibaba, max_series=args.max_alibaba_series)

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Step 2/4 — Preprocess")
    print("=" * 60)
    from preprocess import build_source_target

    data = build_source_target(
        google_series,
        alibaba_series,
        window_size=args.window_size,
        horizon=args.horizon,
        use_dtw=not args.no_dtw,
        seed=args.seed,
    )

    # Save meta
    with open(out_dir / "meta.json", "w") as f:
        json.dump(data["meta"], f, indent=2)

    # ── 3. Train MCTL ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Step 3/4 — Train MCTL")
    print("=" * 60)
    from train import train_mctl

    model = train_mctl(
        data,
        device=args.device,
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        horizon=args.horizon,
        stage1_epochs=args.stage1_epochs,
        stage2a_epochs=args.stage2a_epochs,
        stage2b_epochs=args.stage2b_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=str(ckpt_dir),
        verbose=True,
    )

    # ── 4. Evaluate all models ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Step 4/4 — Evaluate (all baselines + MCTL)")
    print("=" * 60)
    from evaluate import run_full_evaluation, print_results_table

    results = run_full_evaluation(
        model, data,
        device=args.device,
        skip_arima=args.skip_arima,
    )

    # ── Save + print ──────────────────────────────────────────────────────────
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print_results_table(results, title="Google → Alibaba  (MCTL paper replication)")

    # Write table to file too
    table_lines = []
    header = f"{'Method':<14}  {'MAE':>12}  {'MSE':>12}  {'MAPE':>12}  {'sMAPE':>12}  {'Variance':>12}"
    table_lines.append(header)
    table_lines.append("-" * 75)
    for name, m in results.items():
        row = (
            f"{name:<14}  "
            f"{m['MAE']:12.4E}  "
            f"{m['MSE']:12.4E}  "
            f"{m['MAPE']:12.4E}  "
            f"{m['sMAPE']:12.4E}  "
            f"{m['Variance']:12.4E}"
        )
        table_lines.append(row)
    with open(out_dir / "table.txt", "w") as f:
        f.write("\n".join(table_lines))

    print(f"\nTotal time: {time.time() - t_total:.1f}s")
    print(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
