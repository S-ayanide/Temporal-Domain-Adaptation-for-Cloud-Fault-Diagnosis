"""
run.py
======
Replicate either CWPDDA or MCTL (or both) from a single command.

Usage:
    # Replicate CWPDDA (Wang et al., Euro-Par 2025) — YOUR MAIN TARGET
    python run.py --paper cwpdda \
        --google  raw/google \
        --alibaba raw/alibaba \
        --device  cuda

    # Replicate MCTL (Zuo et al., Computing 2025)
    python run.py --paper mctl \
        --google  raw/google \
        --alibaba raw/alibaba \
        --device  cuda

    # Both
    python run.py --paper both ...

    # Quick smoke test (no GPU, few epochs, skip slow baselines)
    python run.py --paper cwpdda --quick

    # First run: save preprocessed arrays (skip slow reload later)
    python run.py --paper cwpdda ... --save-cache results/preprocessed.npz

    # Later runs: train/eval from cache only (steps 1–2 skipped)
    python run.py --paper cwpdda ... --load-cache results/preprocessed.npz
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np


def _validate_preprocess_cache(meta: dict, args: argparse.Namespace) -> None:
    """Ensure CLI matches the run that produced the cache (if cache_spec present)."""
    spec = meta.get("cache_spec")
    if not spec:
        return
    checks = [
        ("max_google", spec.get("max_google"), args.max_google),
        ("max_alibaba", spec.get("max_alibaba"), args.max_alibaba),
        ("seed", spec.get("seed"), args.seed),
        ("use_dtw", spec.get("use_dtw"), not args.no_dtw),
        ("window_size", spec.get("window_size"), args.window_size),
        ("horizon", spec.get("horizon"), args.horizon),
    ]
    bad = [f"  {name}: cache={c!r} current={a!r}" for name, c, a in checks if c != a]
    if bad:
        raise RuntimeError(
            "Preprocess cache does not match current flags:\n"
            + "\n".join(bad)
            + "\nOmit --load-cache to rebuild, or use matching arguments."
        )


def _cuda_device_index(device: str) -> int:
    if not device.startswith("cuda"):
        return 0
    if ":" in device:
        return int(device.split(":", 1)[1])
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paper",   default="cwpdda",
                   choices=["cwpdda", "mctl", "both", "mc_cwpdda"],
                   help="Which model to run (mc_cwpdda = novel contribution)")
    p.add_argument("--google",  default="raw/google")
    p.add_argument("--alibaba", default="raw/alibaba")
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
    p.add_argument("--patience",    type=int,   default=20,
                   help="Early stopping patience (CWPDDA)")
    p.add_argument("--batch-size",  type=int,   default=32)

    # MCTL hyperparams
    p.add_argument("--stage1-epochs",  type=int, default=50)
    p.add_argument("--stage2a-epochs", type=int, default=50)
    p.add_argument("--stage2b-epochs", type=int, default=50)

    # MC-CWPDDA hyperparams (three-stage curriculum)
    p.add_argument("--mc-stage1-epochs", type=int, default=30,
                   help="MC-CWPDDA Stage 1: source pre-training epochs")
    p.add_argument("--mc-stage2-epochs", type=int, default=50,
                   help="MC-CWPDDA Stage 2: contrastive alignment epochs")
    p.add_argument("--mc-stage3-epochs", type=int, default=100,
                   help="MC-CWPDDA Stage 3: joint fine-tuning epochs")
    p.add_argument("--proj-dim", type=int, default=64,
                   help="MC-CWPDDA contrastive head projection dimension")

    # Convenience flags
    p.add_argument("--quick",        action="store_true",
                   help="Few epochs, skip slow baselines — for smoke testing")
    p.add_argument("--skip-gluonts", action="store_true",
                   help="Skip DeepAR/DRP/MQF2 baselines (require gluonts)")
    p.add_argument("--checkpoint-every", type=int, default=10, metavar="N",
                   help="Save a recovery checkpoint every N epochs (default 10)")
    p.add_argument("--resume",       default=None, metavar="PATH",
                   help="Resume CWPDDA training from a recovery checkpoint "
                        "(e.g. checkpoints/cwpdda_resume.pt)")
    p.add_argument(
        "--eval-max-test",
        type=int,
        default=None,
        metavar="N",
        help="Step 4 only: evaluate baselines on at most N random test windows "
             "(much faster; ARIMA fits once per row)",
    )

    # Cache preprocessed tensors (skip slow load + DTW on repeat runs)
    p.add_argument(
        "--save-cache",
        default=None,
        metavar="PATH.npz",
        help="After preprocessing, save arrays to PATH.npz and meta to PATH.json",
    )
    p.add_argument(
        "--load-cache",
        default=None,
        metavar="PATH.npz",
        help="Skip steps 1–2; load from PATH.npz (+ .json) written by --save-cache",
    )
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.quick:
        args.epochs = 10
        args.stage1_epochs = 5
        args.stage2a_epochs = 5
        args.stage2b_epochs = 5
        args.mc_stage1_epochs = 5
        args.mc_stage2_epochs = 5
        args.mc_stage3_epochs = 10
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
            di = _cuda_device_index(args.device)
            print(f"\n[GPU] {args.device}  —  {torch.cuda.get_device_name(di)}  "
                  f"({torch.cuda.device_count()} GPU(s))\n")
    else:
        print(f"\n[Device] {args.device}  — pass --device cuda to use GPU\n")

    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from preprocess import (
        build_source_target,
        load_preprocess_cache,
        save_preprocess_cache,
    )

    if args.load_cache:
        if args.save_cache:
            print("[info] --save-cache ignored when using --load-cache")
        print("\n" + "=" * 60)
        print(" Steps 1–2 skipped — loading preprocess cache")
        print("=" * 60)
        data = load_preprocess_cache(args.load_cache)
        _validate_preprocess_cache(data["meta"], args)
    else:
        # ── 1. Load ───────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print(" Step 1/4 — Load data")
        print("=" * 60)
        from data_loader import load_google, load_alibaba

        google_series  = load_google(args.google,  max_series=args.max_google)
        alibaba_series = load_alibaba(args.alibaba, max_series=args.max_alibaba)

        # ── 2. Preprocess ─────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print(" Step 2/4 — Preprocess")
        print("=" * 60)

        data = build_source_target(
            google_series, alibaba_series,
            window_size=args.window_size,
            horizon=args.horizon,
            use_dtw=not args.no_dtw,
            seed=args.seed,
        )
        data["meta"]["cache_spec"] = {
            "max_google": args.max_google,
            "max_alibaba": args.max_alibaba,
            "seed": args.seed,
            "use_dtw": not args.no_dtw,
            "window_size": args.window_size,
            "horizon": args.horizon,
        }
        if args.save_cache:
            save_preprocess_cache(args.save_cache, data)
            print(f"\n[cache] Saved preprocess bundle to {args.save_cache} (+ .json)")

    with open(out_dir / "meta.json", "w") as f:
        json.dump(data["meta"], f, indent=2)

    # ── 3. Train ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Step 3/4 — Train")
    print("=" * 60)
    from train import train_cwpdda, train_mctl, train_mc_cwpdda

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
            patience=args.patience,
            save_dir=str(ckpt_dir),
            verbose=True,
            checkpoint_every=args.checkpoint_every,
            resume_from=args.resume,
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

    if args.paper == "mc_cwpdda":
        from mc_cwpdda import MCCWPDDA
        mc_cwpdda = MCCWPDDA(
            window_size=args.window_size,
            d_model=args.d_model,
            lstm_hidden=args.lstm_hidden,
            lstm_layers=args.lstm_layers,
            dropout=args.dropout,
            horizon=args.horizon,
            proj_dim=args.proj_dim,
        )
        train_mc_cwpdda(
            mc_cwpdda, data,
            device=args.device,
            stage1_epochs=args.mc_stage1_epochs,
            stage2_epochs=args.mc_stage2_epochs,
            stage3_epochs=args.mc_stage3_epochs,
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
    from evaluate import (
        run_cwpdda_comparison,
        run_mctl_comparison,
        print_cwpdda_table,
        print_mctl_table,
    )

    if args.paper in ("cwpdda", "both"):
        print("\n[CWPDDA baselines]")
        results_cwpdda = run_cwpdda_comparison(
            cwpdda, data, device=args.device,
            skip_gluonts=args.skip_gluonts,
            max_test_windows=args.eval_max_test,
            subsample_seed=args.seed,
        )
        print_cwpdda_table(results_cwpdda)
        results_all["cwpdda"] = results_cwpdda
        cwpdda_json = out_dir / "cwpdda_results.json"
        with open(cwpdda_json, "w") as f:
            json.dump(results_cwpdda, f, indent=2)
        print(f"\n  Saved metrics JSON: {cwpdda_json.resolve()}", flush=True)

        lines = [
            "CWPDDA Results — Google → Alibaba 2017",
            "Target: MAE=2.4183  MAPE=8.66%  RMSE=2.5859",
            "",
            f"{'Method':<12}  {'MAE':>8}  {'MAPE %':>8}  {'RMSE':>8}",
        ]
        lines.append("-" * 45)
        for name, m in results_cwpdda.items():
            lines.append(
                f"{name:<12}  {m['MAE']:8.4f}  {m['MAPE_%']:8.2f}  {m['RMSE']:8.4f}"
            )
        table_path = out_dir / "cwpdda_table.txt"
        table_path.write_text("\n".join(lines))
        print(f"  Saved table:        {table_path.resolve()}", flush=True)
        if "CWPDDA" in results_cwpdda:
            r = results_cwpdda["CWPDDA"]
            print(
                f"\n  CWPDDA test — MAE={r['MAE']:.4f}  "
                f"MAPE={r['MAPE_%']:.2f}%  RMSE={r['RMSE']:.4f}",
                flush=True,
            )

    if args.paper in ("mctl", "both"):
        print("\n[MCTL baselines]")
        results_mctl = run_mctl_comparison(
            mctl, data, device=args.device,
            max_test_windows=args.eval_max_test,
            subsample_seed=args.seed,
        )
        print_mctl_table(results_mctl)
        results_all["mctl"] = results_mctl
        with open(out_dir / "mctl_results.json", "w") as f:
            json.dump(results_mctl, f, indent=2)

    if args.paper == "mc_cwpdda":
        # Reuse CWPDDA evaluation pipeline — MC-CWPDDA has the same predict interface
        print("\n[MC-CWPDDA vs baselines]")
        results_mc = run_cwpdda_comparison(
            mc_cwpdda, data, device=args.device,
            skip_gluonts=args.skip_gluonts,
            max_test_windows=args.eval_max_test,
            subsample_seed=args.seed,
        )
        # Override the model label in the results dict
        if "CWPDDA" in results_mc:
            results_mc["MC-CWPDDA"] = results_mc.pop("CWPDDA")
        print_cwpdda_table(results_mc)
        results_all["mc_cwpdda"] = results_mc
        mc_json = out_dir / "mc_cwpdda_results.json"
        with open(mc_json, "w") as f:
            json.dump(results_mc, f, indent=2)
        print(f"\n  Saved metrics JSON: {mc_json.resolve()}", flush=True)

        lines = [
            "MC-CWPDDA Results — Google → Alibaba 2017",
            "Baselines: CWPDDA MAE=2.4183  MAPE=8.66%  RMSE=2.5859",
            "",
            f"{'Method':<14}  {'MAE':>8}  {'MAPE %':>8}  {'RMSE':>8}",
            "-" * 47,
        ]
        for name, m in results_mc.items():
            lines.append(
                f"{name:<14}  {m['MAE']:8.4f}  {m['MAPE_%']:8.2f}  {m['RMSE']:8.4f}"
            )
        table_path = out_dir / "mc_cwpdda_table.txt"
        table_path.write_text("\n".join(lines))
        print(f"  Saved table:        {table_path.resolve()}", flush=True)
        if "MC-CWPDDA" in results_mc:
            r = results_mc["MC-CWPDDA"]
            print(
                f"\n  MC-CWPDDA test — MAE={r['MAE']:.4f}  "
                f"MAPE={r['MAPE_%']:.2f}%  RMSE={r['RMSE']:.4f}",
                flush=True,
            )

    print(f"\nTotal time: {time.time() - t0:.0f}s")
    print(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
