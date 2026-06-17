"""
deepjdot/run.py
===============
Entry point for DeepJDOT workload prediction experiments.
Adapts DeepJDOT (Damodaran et al., ECCV 2018) from image classification to
cloud workload prediction: Google Cluster Trace → Alibaba 2017.

Usage:
    # First run — preprocess and save cache
    python deepjdot/run.py \
        --google  data/raw/google \
        --alibaba data/raw/alibaba \
        --device  cuda \
        --save-cache results/preprocessed.npz

    # Subsequent runs — load from cache (skip slow data loading)
    python deepjdot/run.py \
        --load-cache results/preprocessed.npz \
        --device cuda

    # Quick smoke test
    python deepjdot/run.py --quick

    # Resume after server timeout
    python deepjdot/run.py \
        --load-cache results/preprocessed.npz \
        --resume checkpoints/deepjdot_resume.pt \
        --device cuda

Run from the updated_research/ directory:
    cd updated_research && python deepjdot/run.py [args]
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Allow importing from parent directory (preprocess, data_loader, baselines)
_HERE   = Path(__file__).resolve().parent         # updated_research/deepjdot/
_PARENT = _HERE.parent                             # updated_research/
# deepjdot/ must come first so "from train import" finds deepjdot/train.py
# not the parent updated_research/train.py
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_HERE))


def parse_args():
    p = argparse.ArgumentParser(description="DeepJDOT workload prediction")

    # Data paths
    p.add_argument("--google",  default="data/raw/google")
    p.add_argument("--alibaba", default="data/raw/alibaba")
    p.add_argument("--out",     default="results/deepjdot")
    p.add_argument("--ckpt",    default="checkpoints")
    p.add_argument("--device",  default="cpu")
    p.add_argument("--seed",    type=int, default=42)

    # Data options (match parent run.py for cache compatibility)
    p.add_argument("--max-google",  type=int, default=3000)
    p.add_argument("--max-alibaba", type=int, default=3000)
    p.add_argument("--window-size", type=int, default=24)
    p.add_argument("--horizon",     type=int, default=1)
    p.add_argument("--max-target-len",   type=int, default=0)
    p.add_argument("--max-target-train", type=int, default=0)

    # Cache
    p.add_argument("--save-cache", default=None, metavar="PATH.npz")
    p.add_argument("--load-cache", default=None, metavar="PATH.npz")

    # DeepJDOT model hyperparameters
    p.add_argument("--hidden-dim",  type=int,   default=64,
                   help="LSTM hidden units in encoder")
    p.add_argument("--n-layers",    type=int,   default=2,
                   help="LSTM layers in encoder")
    p.add_argument("--d-embed",     type=int,   default=64,
                   help="Embedding dimension (OT feature space)")
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--alpha",       type=float, default=0.01,
                   help="OT cost weight on feature alignment term")
    p.add_argument("--lambda-t",    type=float, default=0.1,
                   help="OT cost weight on label consistency term")

    # Training hyperparameters (paper: lr=2e-4, batch=500, Adam)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--pretrain-epochs", type=int,   default=30,
                   help="Source-only MSE pre-training epochs before joint OT training. "
                        "Gives embedding space structure so OT coupling is meaningful at epoch 1.")
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch-size",     type=int,   default=256,
                   help="Minibatch size m (OT matrix is m×m; keep ≤512)")
    p.add_argument("--patience",       type=int,   default=20)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--resume",         default=None, metavar="PATH")

    # Eval
    p.add_argument("--eval-max-test",  type=int, default=None)
    p.add_argument("--quick",          action="store_true",
                   help="Smoke test: few epochs, small data")

    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.quick:
        args.pretrain_epochs = 3
        args.epochs          = 5
        args.max_google      = 300
        args.max_alibaba     = 300
        args.batch_size      = 64
        print("[quick mode] Reduced epochs and data size")

    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] CUDA not available — falling back to CPU")
        args.device = "cpu"

    out_dir  = Path(args.out);  out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt); ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from preprocess import (
        build_source_target,
        load_preprocess_cache,
        save_preprocess_cache,
    )

    # ── 1-2. Load / preprocess ────────────────────────────────────────────────
    if args.load_cache:
        print("\n" + "=" * 60)
        print(" Steps 1–2 skipped — loading preprocess cache")
        print("=" * 60)
        data = load_preprocess_cache(args.load_cache)
        if args.max_target_train:
            cap = args.max_target_train
            data["tgt_train_X"] = data["tgt_train_X"][:cap]
            data["tgt_train_y"] = data["tgt_train_y"][:cap]
    else:
        print("\n" + "=" * 60)
        print(" Step 1/4 — Load data")
        print("=" * 60)
        from data_loader import load_google, load_alibaba
        google_series  = load_google(args.google,  max_series=args.max_google)
        alibaba_series = load_alibaba(args.alibaba, max_series=args.max_alibaba)

        print("\n" + "=" * 60)
        print(" Step 2/4 — Preprocess")
        print("=" * 60)
        # DeepJDOT: no DTW — alignment done via OT during training
        data = build_source_target(
            google_series, alibaba_series,
            window_size=args.window_size,
            horizon=args.horizon,
            use_dtw=False,
            seed=args.seed,
            max_target_len=args.max_target_len,
            max_target_train=args.max_target_train,
        )
        if args.save_cache:
            save_preprocess_cache(args.save_cache, data)
            print(f"\n[cache] Saved to {args.save_cache}")

    with open(out_dir / "meta.json", "w") as f:
        json.dump(data["meta"], f, indent=2)

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Step 3/4 — Train")
    print("=" * 60)

    from model import DeepJDOT
    from train import train_deepjdot

    model = DeepJDOT(
        window_size=args.window_size,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        d_embed=args.d_embed,
        dropout=args.dropout,
        alpha=args.alpha,
        lambda_t=args.lambda_t,
    )

    train_deepjdot(
        model, data,
        device=args.device,
        pretrain_epochs=args.pretrain_epochs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        save_dir=str(ckpt_dir),
        verbose=True,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume,
    )

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Step 4/4 — Evaluate")
    print("=" * 60)

    from evaluate import run_deepjdot_comparison, print_deepjdot_table

    print("\n[DeepJDOT baselines]")
    results = run_deepjdot_comparison(
        model, data,
        device=args.device,
        max_test_windows=args.eval_max_test,
        subsample_seed=args.seed,
        partial_save_path=str(out_dir / "deepjdot_results_partial.json"),
    )
    print_deepjdot_table(results)

    results_path = out_dir / "deepjdot_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved metrics: {results_path.resolve()}", flush=True)

    lines = [
        "DeepJDOT Results — Google → Alibaba",
        "DeepJDOT: unsupervised DA (uses Alibaba unlabelled + Google labelled)",
        "Baselines: trained in-domain on Alibaba",
        "",
        f"{'Method':<12}  {'MAE':>8}  {'MAPE %':>8}  {'RMSE':>8}",
        "-" * 45,
    ]
    for name, m in results.items():
        lines.append(f"{name:<12}  {m['MAE']:8.4f}  {m['MAPE_%']:8.2f}  {m['RMSE']:8.4f}")
    table_path = out_dir / "deepjdot_table.txt"
    table_path.write_text("\n".join(lines))
    print(f"  Saved table:   {table_path.resolve()}", flush=True)

    if "DeepJDOT" in results:
        r = results["DeepJDOT"]
        print(f"\n  DeepJDOT — MAE={r['MAE']:.4f}  MAPE={r['MAPE_%']:.2f}%  "
              f"RMSE={r['RMSE']:.4f}", flush=True)

    print(f"\nTotal time: {time.time() - t0:.0f}s")
    print(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
