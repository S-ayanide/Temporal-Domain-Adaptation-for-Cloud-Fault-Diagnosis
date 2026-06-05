"""
run.py — CLI entry point for Tr-Predictor experiments.

Usage
-----
# Full run: all datasets as target, rest as sources
python run.py --data_dir ../rossi_replication/data --results_dir results/

# Single target
python run.py --target GC19_a --data_dir ../rossi_replication/data

# With raw AC18 per-machine traces
python run.py --data_dir ../rossi_replication/data \
              --raw_dir ~/research/data/raw \
              --results_dir results/

Experiment design (mirrors §4 of the paper):
  For each target domain d:
    1. Compute TWED + TE similarity of d against all other domains.
    2. Select top-k source domains (k=5 default, paper tunes this).
    3. Pool selected source data.
    4. Run TrAdaBoost.R2-LSTM with source + tiny target data.
    5. Evaluate on held-out test split of d.

Baselines:
  - No-transfer:  LSTM trained only on target data (no source)
  - All-source:   LSTM trained on all source data (no TL)
  - Tr-Predictor: Two-stage TrAdaBoost (this paper)
"""

import argparse
import json
import os
import sys
import numpy as np
import copy

# Ensure imports work regardless of CWD
sys.path.insert(0, os.path.dirname(__file__))

from preprocess import load_all, SEQ_LEN, HORIZON, TARGET_LEN
from similarity import select_sources
from tr_adaboost import TrAdaBoostLSTM
from lstm_model import build_weak_learner, train_weak_learner, predict_weak_learner
from metrics import evaluate_all, print_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _concat_source_data(splits: list, source_indices: list,
                         exclude_idx: int):
    """Pool X_src / Y_src from selected source domains."""
    Xs, Ys = [], []
    for idx in source_indices:
        if idx == exclude_idx:
            continue
        sp = splits[idx]
        if len(sp["X_src"]) > 0:
            Xs.append(sp["X_src"])
            Ys.append(sp["Y_src"])
    if not Xs:
        return None, None
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)


def _run_baseline_no_transfer(sp: dict, args) -> dict:
    """LSTM trained only on target train data (no source)."""
    model = build_weak_learner(
        seq_len=args.seq_len, n_features=sp["X_tgt_tr"].shape[-1],
        n_targets=sp["Y_tgt_tr"].shape[-1],
        lstm_units=args.lstm_units, dense_units=args.dense_units,
        learning_rate=args.lr,
    )
    w = np.ones(len(sp["X_tgt_tr"]))
    train_weak_learner(
        model, sp["X_tgt_tr"], sp["Y_tgt_tr"], weights=w,
        X_val=sp["X_tgt_val"], Y_val=sp["Y_tgt_val"],
        batch_size=args.batch_size, max_epochs=args.epochs,
        patience=args.patience, verbose=0,
    )
    pred = predict_weak_learner(model, sp["X_tgt_te"])
    return evaluate_all(pred, sp["Y_tgt_te"])


def _run_baseline_all_source(sp: dict, X_src: np.ndarray,
                               Y_src: np.ndarray, args) -> dict:
    """LSTM trained on all source data only (zero-shot transfer, no TL)."""
    if X_src is None or len(X_src) == 0:
        return {"mse": float("nan"), "mae": float("nan"),
                "mape": float("nan"), "r2": float("nan")}
    model = build_weak_learner(
        seq_len=args.seq_len, n_features=sp["X_tgt_te"].shape[-1],
        n_targets=sp["Y_tgt_te"].shape[-1],
        lstm_units=args.lstm_units, dense_units=args.dense_units,
        learning_rate=args.lr,
    )
    w = np.ones(len(X_src))
    train_weak_learner(
        model, X_src, Y_src, weights=w,
        batch_size=args.batch_size, max_epochs=args.epochs,
        patience=args.patience, verbose=0,
    )
    pred = predict_weak_learner(model, sp["X_tgt_te"])
    return evaluate_all(pred, sp["Y_tgt_te"])


def _run_tr_predictor(sp: dict, X_src: np.ndarray,
                       Y_src: np.ndarray, args) -> dict:
    """Full Tr-Predictor: TrAdaBoost.R2-LSTM."""
    if X_src is None or len(X_src) == 0:
        print("    [warn] No source data; falling back to no-transfer baseline.")
        return _run_baseline_no_transfer(sp, args)

    model = TrAdaBoostLSTM(
        n_rounds=args.n_rounds,
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
        seq_len=args.seq_len,
        n_features=sp["X_tgt_tr"].shape[-1],
        n_targets=sp["Y_tgt_tr"].shape[-1],
        lr=args.lr,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        verbose=args.verbose,
    )
    model.fit(
        X_src, Y_src,
        sp["X_tgt_tr"], sp["Y_tgt_tr"],
        X_val=sp["X_tgt_val"], Y_val=sp["Y_tgt_val"],
    )
    pred = model.predict(sp["X_tgt_te"])
    return evaluate_all(pred, sp["Y_tgt_te"])


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(splits: list, args) -> dict:
    all_results = {}

    target_names = (
        [args.target] if args.target else [sp["name"] for sp in splits]
    )

    for sp in splits:
        if sp["name"] not in target_names:
            continue

        tgt_idx = next(i for i, s in enumerate(splits) if s["name"] == sp["name"])
        print(f"\n{'='*60}")
        print(f"Target: {sp['name']}  "
              f"tgt_tr={len(sp['X_tgt_tr'])} src_avail={sum(len(s['X_src']) for i,s in enumerate(splits) if i!=tgt_idx)}")

        # ---- 1. Source selection ----
        tgt_rep = sp["tgt_series"][:, 0]   # CPU column for similarity
        candidate_sources = [
            s["src_series"][:, 0]
            for i, s in enumerate(splits)
            if i != tgt_idx and len(s["src_series"]) > 0
        ]
        candidate_names = [
            s["name"]
            for i, s in enumerate(splits)
            if i != tgt_idx and len(s["src_series"]) > 0
        ]
        candidate_indices = [
            i
            for i, s in enumerate(splits)
            if i != tgt_idx and len(s["src_series"]) > 0
        ]

        if candidate_sources:
            ranked = select_sources(
                tgt_rep, candidate_sources, candidate_names,
                top_k=args.top_k,
                lam=args.twed_lambda, nu=args.twed_nu,
            )
            selected_indices = [candidate_indices[r[0]] for r in ranked]
            print(f"  Selected sources: {[r[1] for r in ranked]}")
        else:
            selected_indices = []
            print("  No candidate sources available.")

        # ---- 2. Pool source data ----
        X_src, Y_src = _concat_source_data(splits, selected_indices, tgt_idx)

        # ---- 3. Baselines ----
        print("  [baseline] No-transfer …", end=" ", flush=True)
        res_nt = _run_baseline_no_transfer(sp, args)
        print_metrics("no-transfer", res_nt)

        print("  [baseline] All-source (ZS) …", end=" ", flush=True)
        res_as = _run_baseline_all_source(sp, X_src, Y_src, args)
        print_metrics("all-source", res_as)

        # ---- 4. Tr-Predictor ----
        print("  [Tr-Predictor] TrAdaBoost …")
        res_tr = _run_tr_predictor(sp, X_src, Y_src, args)
        print_metrics("Tr-Predictor", res_tr)

        all_results[sp["name"]] = {
            "no_transfer": res_nt,
            "all_source":  res_as,
            "tr_predictor": res_tr,
            "selected_sources": [r[1] for r in ranked] if candidate_sources else [],
        }

    return all_results


# ---------------------------------------------------------------------------
# Average summary
# ---------------------------------------------------------------------------

def summarise(results: dict):
    methods = ["no_transfer", "all_source", "tr_predictor"]
    metrics = ["mse", "mae", "mape", "r2"]
    print(f"\n{'='*60}")
    print("AVERAGE ACROSS ALL TARGETS")
    print(f"{'Method':<22} {'MSE':>8} {'MAE':>8} {'MAPE%':>8} {'R2':>8}")
    print("-" * 55)
    for method in methods:
        vals = {m: [] for m in metrics}
        for _, r in results.items():
            d = r.get(method, {})
            for m in metrics:
                v = d.get(m)
                if v is not None and not np.isnan(v):
                    vals[m].append(v)
        avgs = {m: np.mean(vals[m]) if vals[m] else float("nan") for m in metrics}
        print(f"{method:<22} "
              f"{avgs['mse']:>8.5f} "
              f"{avgs['mae']:>8.5f} "
              f"{avgs['mape']:>8.2f} "
              f"{avgs['r2']:>8.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Tr-Predictor: Two-Stage TrAdaBoost.R2-LSTM")

    # Data
    p.add_argument("--raw_dir",    required=True,
                   help="Raw data root: contains google/cell_{a-h}/*.json.gz and/or alibaba/machine_usage.csv")
    p.add_argument("--results_dir", default="results",
                   help="Where to save JSON results")
    p.add_argument("--target",     default=None,
                   help="Run for a single target dataset name (e.g. GC19_a)")
    p.add_argument("--target_len", type=int, default=TARGET_LEN,
                   help="Target training window length (time steps)")
    p.add_argument("--seq_len",    type=int, default=SEQ_LEN,
                   help="Look-back window length")
    p.add_argument("--horizon",    type=int, default=HORIZON,
                   help="Forecast horizon (steps)")
    p.add_argument("--max_shards", type=int, default=None,
                   help="Limit JSON.gz shards per GC19 cell (e.g. 20 for smoke test)")
    p.add_argument("--cache_dir",  default=None,
                   help="Dir to cache preprocessed .npy files (default: <raw_dir>/../tr_cache)")

    # Source selection
    p.add_argument("--top_k",        type=int,   default=5,
                   help="Number of source domains to select")
    p.add_argument("--twed_lambda",  type=float, default=0.5)
    p.add_argument("--twed_nu",      type=float, default=0.001)

    # Model
    p.add_argument("--lstm_units",   type=int,   default=64)
    p.add_argument("--dense_units",  type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--n_rounds",     type=int,   default=20,
                   help="TrAdaBoost rounds T")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--patience",     type=int,   default=10)

    # Misc
    p.add_argument("--gpu",          type=int,   default=None,
                   help="GPU index (e.g. 0). None = use all.")
    p.add_argument("--verbose",      type=int,   default=0,
                   help="TrAdaBoost verbosity (0=silent, 1=round info)")
    p.add_argument("--seed",         type=int,   default=42)

    return p.parse_args()


def main():
    args = parse_args()

    # GPU setup
    if args.gpu is not None:
        import os as _os
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Reproducibility
    import random, tensorflow as tf
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Load data
    print(f"Loading data from {args.raw_dir} …")
    splits = load_all(
        raw_dir=args.raw_dir,
        cache_dir=args.cache_dir,
        target_len=args.target_len,
        seq_len=args.seq_len,
        horizon=args.horizon,
        max_shards=args.max_shards,
    )
    print(f"Loaded {len(splits)} datasets.")

    if not splits:
        print("ERROR: no datasets found. Check --data_dir path.")
        sys.exit(1)

    # Run
    results = run_experiment(splits, args)

    # Summary table
    summarise(results)

    # Save
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "tr_predictor_results.json")

    def _serial(x):
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return str(x)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_serial)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
