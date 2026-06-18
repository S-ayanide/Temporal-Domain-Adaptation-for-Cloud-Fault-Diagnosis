"""
deepjdot/evaluate.py
====================
Evaluation for DeepJDOT workload prediction.

Uses the same CWPDDA metrics (MAE, MAPE%, RMSE on 0-100 CPU utilisation scale)
so results are directly comparable with CWPDDA, MCTL, and N-BEATS tables.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ─── Metric functions (same as parent evaluate.py) ────────────────────────────

def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mape(y_true, y_pred, eps=1e-8):
    scale  = max(float(np.abs(y_true).max()), 1.0)
    thresh = 0.01 * scale
    mask   = np.abs(y_true) > thresh
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask])
                                 / (np.abs(y_true[mask]) + eps))))

def cwpdda_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, MAPE (%), RMSE on 0-100 CPU utilisation scale."""
    y_true = y_true.squeeze() * 100.0
    y_pred = y_pred.squeeze() * 100.0
    return {
        "MAE":    _mae(y_true, y_pred),
        "MAPE_%": _mape(y_true, y_pred) * 100,
        "RMSE":   _rmse(y_true, y_pred),
    }


# ─── DeepJDOT evaluation ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_deepjdot(model, X_test, y_test, device="cpu",
                      infer_batch_size: int = 2048) -> dict:
    model.eval()
    pred = model.predict_numpy_batched(X_test, device, batch_size=infer_batch_size)
    return cwpdda_metrics(y_test, pred)


def _maybe_subsample(X, y, max_windows, seed):
    if max_windows is None or len(X) <= max_windows:
        return X, y, False
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_windows, replace=False)
    idx.sort()
    return X[idx], y[idx], True


def run_deepjdot_comparison(
    deepjdot_model,
    data: dict,
    device: str = "cpu",
    max_test_windows: Optional[int] = None,
    subsample_seed: int = 42,
    partial_save_path: Optional[str] = None,
) -> dict:
    """
    Evaluate DeepJDOT vs ARIMA and LSTM baselines on Alibaba test set.

    DeepJDOT is an unsupervised domain adaptation model — it uses Alibaba
    training windows (unlabelled) during training to align feature distributions.
    Baselines (ARIMA, LSTM) are trained in-domain on Alibaba.

    Saves results incrementally to partial_save_path after each baseline so a
    crash cannot lose already-computed metrics.
    """
    import sys, os
    _here   = os.path.dirname(os.path.abspath(__file__))   # deepjdot/
    _parent = os.path.dirname(_here)                        # updated_research/
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from baselines import ARIMABaseline, LSTMBaseline

    X_tr = data["tgt_train_X"]; y_tr = data["tgt_train_y"]
    X_te = data["tgt_test_X"];  y_te = data["tgt_test_y"]
    W    = X_tr.shape[1]

    X_te, y_te, sub = _maybe_subsample(X_te, y_te, max_test_windows, subsample_seed)
    if sub:
        print(f"  Test subsampled to {len(X_te):,} windows.", flush=True)

    print(f"  Dataset: {len(X_tr):,} train / {len(X_te):,} test (W={W}).", flush=True)
    if len(X_te) == 0:
        raise RuntimeError("No target test windows. Check preprocessing.")

    def _save(results):
        if partial_save_path:
            Path(partial_save_path).write_text(json.dumps(results, indent=2))

    def _baseline_metrics(model, X_te, y_te):
        pred = model.predict(X_te)
        return cwpdda_metrics(y_te, pred)

    # Cap baseline training data — LSTM on 200k windows × 150 epochs = hours.
    # 10k windows is more than enough to train a representative baseline LSTM.
    _BASELINE_TRAIN_CAP = 10_000
    if len(X_tr) > _BASELINE_TRAIN_CAP:
        rng = np.random.default_rng(subsample_seed)
        _bl_idx = rng.choice(len(X_tr), _BASELINE_TRAIN_CAP, replace=False)
        _bl_idx.sort()
        X_tr_bl, y_tr_bl = X_tr[_bl_idx], y_tr[_bl_idx]
        print(f"  Baselines training capped at {_BASELINE_TRAIN_CAP:,} windows "
              f"(from {len(X_tr):,}) for speed.", flush=True)
    else:
        X_tr_bl, y_tr_bl = X_tr, y_tr

    results = {}
    kw = dict(window_size=W, horizon=y_tr.shape[1], epochs=150, device=device)

    print("  ARIMA...", end=" ", flush=True)
    m = ARIMABaseline(); m.fit(X_tr_bl, y_tr_bl)
    arima_n = min(500, len(X_te))
    idx = np.random.default_rng(subsample_seed).choice(len(X_te), arima_n, replace=False)
    results["ARIMA"] = _baseline_metrics(m, X_te[idx], y_te[idx])
    _save(results)
    print(f"done  (sampled {arima_n})", flush=True)

    print("  LSTM...", end=" ", flush=True)
    m = LSTMBaseline(**kw); m.fit(X_tr_bl, y_tr_bl)
    results["LSTM"] = _baseline_metrics(m, X_te, y_te)
    _save(results)
    print("done", flush=True)

    print("  DeepJDOT...", end=" ", flush=True)
    results["DeepJDOT"] = evaluate_deepjdot(deepjdot_model, X_te, y_te, device)
    _save(results)
    print("done", flush=True)

    return results


def print_deepjdot_table(results: dict,
                          title="DeepJDOT — UDA Workload Prediction (Google→Alibaba)"):
    print(f"\n{'='*62}", flush=True)
    print(f"  {title}", flush=True)
    print(f"  DeepJDOT uses Alibaba (unlabelled) + Google (labelled)", flush=True)
    print(f"  Baselines trained in-domain on Alibaba", flush=True)
    print(f"{'='*62}", flush=True)
    if not results:
        print("  (no results)", flush=True)
        return
    print(f"{'Method':<12}  {'MAE':>8}  {'MAPE %':>8}  {'RMSE':>8}", flush=True)
    print("-" * 45, flush=True)
    for name, m in results.items():
        marker = " ←" if name == "DeepJDOT" else ""
        print(f"{name:<12}  {m['MAE']:8.4f}  {m['MAPE_%']:8.2f}  {m['RMSE']:8.4f}{marker}",
              flush=True)
    print("-" * 45, flush=True)
