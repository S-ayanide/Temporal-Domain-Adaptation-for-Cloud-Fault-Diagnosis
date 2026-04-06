"""
run.py
======
Replicate either CWPDDA or MCTL (or both) from a single command.

Usage:
    # Replicate CWPDDA (Wang et al., Euro-Par 2025) — YOUR MAIN TARGET
    python run.py --paper cwpdda \
        --google  data/raw/google \
        --alibaba data/raw \
        --device  cuda

    # Replicate MCTL (Zuo et al., Computing 2025)
    python run.py --paper mctl \
        --google  data/raw/google \
        --alibaba data/raw \
        --device  cuda

    # Both
    python run.py --paper both ...

    # Quick smoke test (no GPU, few epochs, skip slow baselines)
    python run.py --paper cwpdda --quick
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paper",   default="cwpdda",
                   choices=["cwpdda", "mctl", "both"],
                   help="Which paper to replicate")
    p.add_argument("--google",  default="data/raw/google")
    p.add_argument("--alibaba", default="data/raw")
    p.add_argument("--out",     default="results")
    p.add_argument("--ckpt",    default="checkpoints")
    p.add_argument("--device",  default="cpu")
    p.add_argument("--seed",    type=int, default=42)

    # Data
    p.add_argument("--max-google",  type=int, default=3000)
    p.add_argument("--max-alibaba", type=int, default=3000)
    p.add_argument("--window-size", type=int, default=24,
                   help="Input window length (24 steps = 2h at 5min sampling)")
    p.add_argument("--horizon",     type=int, default=1,
                   help="Steps ahead to predict")
    p.add_argument("--no-dtw",      action="store_true",
                   help="Skip DTW source selection (faster, slightly worse)")

    # CWPDDA hyperparams (Table 2 of paper)
    p.add_argument("--d-model",     type=int,   default=64)
    p.add_argument("--lstm-hidden", type=int,   default=40)
    p.add_argument("--lstm-layers", type=int,   default=2)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch-size",  type=int,   default=32)

    # MCTL hyperparams
    p.add_argument("--stage1-epochs",  type=int, default=50)
    p.add_argument("--stage2a-epochs", type=int, default=50)
    p.add_argument("--stage2b-epochs", type=int, default=50)

    # Convenience flags
    p.add_argument("--quick",        action="store_true",
                   help="Few epochs, skip slow baselines — for smoke testing")
    p.add_argument("--skip-gluonts", action="store_true",
                   help="Skip DeepAR/DRP/MQF2 baselines (require gluonts)")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.quick:
        args.epochs = 10
        args.stage1_epochs = 5
        args.stage2a_epochs = 5
        args.stage2b_epochs = 5
        args.skip_gluonts = True
        args.max_google   = 500
        args.max_alibaba  = 500
        print("[quick mode] Reduced epochs and data size for smoke test")


    # ── GPU check ─────────────────────────────────────────────────────────────
    import torch
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"\n[WARNING] --device {args.device} requested but CUDA is not available.")
            print("          Falling back to CPU.\n")
            args.device = "cpu"
        else:
            print(f"\n[GPU] {args.device}  —  {torch.cuda.get_device_name(0)}  "
                  f"({torch.cuda.device_count()} GPU(s))\n")
    else:
        print(f"\n[Device] {args.device}  — pass --device cuda to use GPU\n")

    out_dir  = Path(args.out);  out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt); ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" Step 1/4 — Load data")
    print("="*60)
    from data_loader import load_google, load_alibaba

    google_series  = load_google(args.google,  max_series=args.max_google)
    alibaba_series = load_alibaba(args.alibaba, max_series=args.max_alibaba)

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" Step 2/4 — Preprocess")
    print("="*60)
    from preprocess import build_source_target

    data = build_source_target(
        google_series, alibaba_series,
        window_size=args.window_size,
        horizon=args.horizon,
        use_dtw=not args.no_dtw,
        seed=args.seed,
    )
    with open(out_dir / "meta.json", "w") as f:
        json.dump(data["meta"], f, indent=2)

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" Step 3/4 — Train")
    print("="*60)
    from train import train_cwpdda, train_mctl

    results_all = {}

    if args.paper in ("cwpdda", "both"):
        from cwpdda import CWPDDA
        cwpdda = CWPDDA(
            window_size=args.window_size,
            d_model=args.d_model,
            lstm_hidden=args.lstm_hidden,
            lstm_layers=args.lstm_layers,
            dropout=args.dropout,
            horizon=args.horizon,
        )
        train_cwpdda(
            cwpdda, data,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_dir=str(ckpt_dir),
            verbose=True,
        )

    if args.paper in ("mctl", "both"):
        from mctl import MCTL
        mctl = MCTL(
            window_size=args.window_size,
            hidden_dim=128,
            horizon=args.horizon,
        )
        train_mctl(
            mctl, data,
            device=args.device,
            stage1_epochs=args.stage1_epochs,
            stage2a_epochs=args.stage2a_epochs,
            stage2b_epochs=args.stage2b_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_dir=str(ckpt_dir),
            verbose=True,
        )

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" Step 4/4 — Evaluate")
    print("="*60)
    from evaluate import (
        run_cwpdda_comparison, run_mctl_comparison,
        print_cwpdda_table, print_mctl_table,
    )

    if args.paper in ("cwpdda", "both"):
        print("\n[CWPDDA baselines]")
        results_cwpdda = run_cwpdda_comparison(
            cwpdda, data, device=args.device,
            skip_gluonts=args.skip_gluonts,
        )
        print_cwpdda_table(results_cwpdda)
        results_all["cwpdda"] = results_cwpdda
        with open(out_dir / "cwpdda_results.json", "w") as f:
            json.dump(results_cwpdda, f, indent=2)

        # Write human-readable table
        lines = ["CWPDDA Results — Google → Alibaba 2017",
                 "Target: MAE=2.4183  MAPE=8.66%  RMSE=2.5859", "",
                 f"{'Method':<12}  {'MAE':>8}  {'MAPE %':>8}  {'RMSE':>8}"]
        lines.append("-" * 45)
        for name, m in results_cwpdda.items():
            lines.append(f"{name:<12}  {m['MAE']:8.4f}  {m['MAPE_%']:8.2f}  {m['RMSE']:8.4f}")
        (out_dir / "cwpdda_table.txt").write_text("\n".join(lines))

    if args.paper in ("mctl", "both"):
        print("\n[MCTL baselines]")
        results_mctl = run_mctl_comparison(mctl, data, device=args.device)
        print_mctl_table(results_mctl)
        results_all["mctl"] = results_mctl
        with open(out_dir / "mctl_results.json", "w") as f:
            json.dump(results_mctl, f, indent=2)

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    main()