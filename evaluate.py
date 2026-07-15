"""
evaluate.py
===========
Metrics and evaluation for both papers.

CWPDDA (Table 3/4): MAE, MAPE (%), RMSE
MCTL   (Table 3/4): MAE, MSE, MAPE, sMAPE, Variance

Paper target numbers to match:

CWPDDA CPU  (Table 3):
    ARIMA  MAE=6.0486  MAPE=20.05%  RMSE=6.5402
    LSTM   MAE=4.9632  MAPE=16.46%  RMSE=4.9173
    DeepAR MAE=3.6593  MAPE=12.69%  RMSE=3.8828
    DRP    MAE=5.1916  MAPE=17.36%  RMSE=5.3581
    MQF2   MAE=5.7957  MAPE=19.96%  RMSE=5.9283
    CWPDDA MAE=2.4183  MAPE=8.66%   RMSE=2.5859   ← target

MCTL Google→Alibaba JobA (Table 3):
    ARIMA  MAE=1.260E-3  MSE=3.036E-6  ...
    MCTL   MAE=7.220E-4  MSE=9.857E-7  ...          ← target
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import torch


# ─── Metric functions ─────────────────────────────────────────────────────────

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred, eps=1e-8):
    # Exclude near-zero true values to avoid division explosion.
    # Threshold = 1% of the data range, or 0.01 on [0,1]-normalised data.
    scale = max(float(np.abs(y_true).max()), 1.0)
    thresh = 0.01 * scale
    mask = np.abs(y_true) > thresh
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask])
                                 / (np.abs(y_true[mask]) + eps))))

def smape(y_true, y_pred, eps=1e-8):
    return float(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)))

def variance_err(y_true, y_pred):
    return float(np.var(y_true - y_pred))


def cwpdda_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAE, MAPE (as %), RMSE — matches Table 3/4 of CWPDDA paper.

    Data is normalised to [0,1] during preprocessing but the paper reports
    metrics on the 0-100% CPU utilisation scale.  Multiply by 100 before
    computing so that MAE=2.4183 means 2.4% CPU error, matching the paper.
    """
    y_true = y_true.squeeze() * 100.0
    y_pred = y_pred.squeeze() * 100.0
    return {
        "MAE":      mae(y_true, y_pred),
        "MAPE_%":   mape(y_true, y_pred) * 100,
        "RMSE":     rmse(y_true, y_pred),
    }


def mctl_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAE, MSE, MAPE, sMAPE, Variance — matches Table 3/4 of MCTL paper."""
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    return {
        "MAE":      mae(y_true, y_pred),
        "MSE":      mse(y_true, y_pred),
        "MAPE":     mape(y_true, y_pred),
        "sMAPE":    smape(y_true, y_pred),
        "Variance": variance_err(y_true, y_pred),
    }


# ─── Classification metrics ───────────────────────────────────────────────────

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Convert continuous CPU-utilisation predictions → binary (high-load / normal)
    and compute Accuracy, Precision, Recall, F1, MCC, G-Mean.

    threshold (on the 0-1 normalised scale):
      - If None: auto-set to the 70th percentile of y_true (top 30% = "high load").
      - Pass a fixed value (e.g. 0.7) to override.

    Returns a dict that also records the threshold used, the positive-class rate,
    and flags metrics that are undefined (NaN) due to degenerate predictions.
    """
    try:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, matthews_corrcoef, confusion_matrix,
        )
    except ImportError:
        raise ImportError("scikit-learn is required for classification metrics: pip install scikit-learn")

    y_true = y_true.squeeze().astype(float)
    y_pred = y_pred.squeeze().astype(float)

    if threshold is None:
        threshold = float(np.percentile(y_true, 70))

    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)

    n_pos_true = int(y_true_bin.sum())
    n_pos_pred = int(y_pred_bin.sum())

    nan = float("nan")

    # Degenerate: all predictions in one class → most metrics undefined
    if len(np.unique(y_pred_bin)) < 2:
        return {
            "Accuracy":  float(accuracy_score(y_true_bin, y_pred_bin)),
            "Precision": nan,
            "Recall":    nan,
            "F1":        nan,
            "MCC":       nan,
            "G-Mean":    nan,
            "threshold": threshold,
            "pct_positive_true": round(n_pos_true / len(y_true_bin), 4),
            "pct_positive_pred": round(n_pos_pred / len(y_pred_bin), 4),
            "note": "degenerate predictions (all one class)",
        }

    cm = confusion_matrix(y_true_bin, y_pred_bin)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        g_mean = float(np.sqrt(sensitivity * specificity))
    else:
        g_mean = nan

    return {
        "Accuracy":  float(accuracy_score(y_true_bin, y_pred_bin)),
        "Precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "Recall":    float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "F1":        float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "MCC":       float(matthews_corrcoef(y_true_bin, y_pred_bin)),
        "G-Mean":    g_mean,
        "threshold": threshold,
        "pct_positive_true": round(n_pos_true / len(y_true_bin), 4),
        "pct_positive_pred": round(n_pos_pred / len(y_pred_bin), 4),
    }


def combined_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scale: str = "cwpdda",   # "cwpdda" (×100) | "mctl" (0-1 normalised)
    threshold: Optional[float] = None,
) -> Dict[str, object]:
    """
    Return both regression metrics and classification metrics in one dict.

    y_true / y_pred should be on the 0-1 normalised scale coming in.
    scale="cwpdda"  → regression metrics are computed on ×100 scale (matching paper)
    scale="mctl"    → regression metrics stay on 0-1 scale
    classification  → always on 0-1 scale (threshold applies before ×100)
    """
    reg = cwpdda_metrics(y_true, y_pred) if scale == "cwpdda" else mctl_metrics(y_true, y_pred)
    clf = classification_metrics(y_true, y_pred, threshold=threshold)
    return {"regression": reg, "classification": clf}


# ─── Prediction helpers ───────────────────────────────────────────────────────

@torch.no_grad()
def predict_model(model, X_test: np.ndarray, device: str = "cpu",
                  batch_size: int = 2048) -> np.ndarray:
    """Run batched inference for any model that has .predict_numpy_batched()."""
    model.eval()
    return model.predict_numpy_batched(X_test, device, batch_size=batch_size)


@torch.no_grad()
def predict_mctl_model(model, X_test: np.ndarray, device: str = "cpu",
                       batch_size: int = 2048) -> np.ndarray:
    """Run batched inference for MCTL (uses .predict() returning tensor)."""
    model.eval()
    if len(X_test) == 0:
        return np.empty((0, 1), dtype=np.float32)
    parts = []
    for i in range(0, len(X_test), batch_size):
        xb = torch.from_numpy(X_test[i: i + batch_size]).float().to(device)
        parts.append(model.predict(xb).cpu().numpy())
    return np.concatenate(parts, axis=0)


# ─── Evaluate models ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_cwpdda(model, X_test, y_test, device="cpu", infer_batch_size: int = 2048):
    model.eval()
    pred = model.predict_numpy_batched(X_test, device, batch_size=infer_batch_size)
    return cwpdda_metrics(y_test, pred)


@torch.no_grad()
def evaluate_mctl(model, X_test, y_test, device="cpu", infer_batch_size: int = 2048):
    model.eval()
    n = len(X_test)
    if n == 0:
        h = int(model.regression_head.out_features)
        pred = np.empty((0, h), dtype=np.float32)
    else:
        parts = []
        for i in range(0, n, infer_batch_size):
            xb = torch.from_numpy(X_test[i : i + infer_batch_size]).float().to(device)
            parts.append(model.predict(xb).cpu().numpy())
        pred = np.concatenate(parts, axis=0)
    return mctl_metrics(y_test, pred)


def evaluate_baseline(model, X_test, y_test, metric_fn):
    pred = model.predict(X_test)
    return metric_fn(y_test, pred)


def _maybe_subsample_test(
    X_te: np.ndarray,
    y_te: np.ndarray,
    max_windows: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (X_te, y_te, did_subsample)."""
    if max_windows is None or len(X_te) <= max_windows:
        return X_te, y_te, False
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_te), size=max_windows, replace=False)
    idx.sort()
    return X_te[idx], y_te[idx], True


# ─── Full comparison tables ───────────────────────────────────────────────────

def run_cwpdda_comparison(
    cwpdda_model,
    data,
    device="cpu",
    skip_gluonts=False,
    max_test_windows: int | None = None,
    subsample_seed: int = 42,
):
    """
    Trains and evaluates all CWPDDA baselines + CWPDDA on the test set.
    Returns dict of {model_name: metrics}.
    """
    from baselines import (
        ARIMABaseline, LSTMBaseline,
        DeepARBaseline, DRPBaseline, MQF2Baseline,
    )

    X_tr  = data["tgt_train_X"]; y_tr  = data["tgt_train_y"]
    X_te  = data["tgt_test_X"];  y_te  = data["tgt_test_y"]
    W     = X_tr.shape[1]

    X_te, y_te, sub = _maybe_subsample_test(X_te, y_te, max_test_windows, subsample_seed)
    if sub:
        print(
            f"  Test subsampled to {len(X_te):,} windows (--eval-max-test).",
            flush=True,
        )

    print(
        f"  Dataset: {len(X_tr):,} train windows, {len(X_te):,} test windows "
        f"(W={W}). Baselines can take a while on large test sets.",
        flush=True,
    )
    if len(X_te) == 0:
        raise RuntimeError(
            "No target test windows (tgt_test_X is empty). "
            "Check preprocessing / Alibaba val-test splits."
        )

    results = {}
    kw = dict(window_size=W, horizon=y_tr.shape[1], epochs=150, device=device)

    print("  ARIMA...",  end=" ", flush=True)
    m = ARIMABaseline(); m.fit(X_tr, y_tr)
    arima_n = min(500, len(X_te))
    idx = np.random.default_rng(42).choice(len(X_te), arima_n, replace=False)
    results["ARIMA"] = evaluate_baseline(m, X_te[idx], y_te[idx], cwpdda_metrics)
    print(f"done  (sampled {arima_n} windows)")

    print("  LSTM...",   end=" ", flush=True)
    m = LSTMBaseline(**kw); m.fit(X_tr, y_tr)
    results["LSTM"] = evaluate_baseline(m, X_te, y_te, cwpdda_metrics)
    print("done")

    if not skip_gluonts:
        print("  DeepAR...", end=" ", flush=True)
        m = DeepARBaseline(prediction_length=y_tr.shape[1], epochs=10)
        m.fit(X_tr, y_tr)
        results["DeepAR"] = evaluate_baseline(m, X_te, y_te, cwpdda_metrics)
        print("done")

        print("  DRP...",    end=" ", flush=True)
        m = DRPBaseline(prediction_length=y_tr.shape[1], epochs=10)
        m.fit(X_tr, y_tr)
        results["DRP"] = evaluate_baseline(m, X_te, y_te, cwpdda_metrics)
        print("done")

        print("  MQF2...",   end=" ", flush=True)
        m = MQF2Baseline(prediction_length=y_tr.shape[1], epochs=10)
        m.fit(X_tr, y_tr)
        results["MQF2"] = evaluate_baseline(m, X_te, y_te, cwpdda_metrics)
        print("done")

    print("  CWPDDA...", end=" ", flush=True)
    results["CWPDDA"] = evaluate_cwpdda(cwpdda_model, X_te, y_te, device)
    print("done")

    return results


def run_mctl_comparison(
    mctl_model,
    data,
    device="cpu",
    max_test_windows: int | None = None,
    subsample_seed: int = 42,
):
    """Trains and evaluates all MCTL baselines + MCTL."""
    from baselines import (
        ARIMABaseline, LSTMBaseline, GRUBaseline, CNNLSTMBaseline,
        AutoformerBaseline, BHTARIMABaseline, WANNBaseline, TS2VecBaseline,
    )
    X_src = data["src_X"]; y_src = data["src_y"]
    X_tr  = data["tgt_train_X"]; y_tr  = data["tgt_train_y"]
    X_te  = data["tgt_test_X"];  y_te  = data["tgt_test_y"]
    W     = X_tr.shape[1]

    X_te, y_te, sub = _maybe_subsample_test(X_te, y_te, max_test_windows, subsample_seed)
    if sub:
        print(
            f"  Test subsampled to {len(X_te):,} windows (--eval-max-test).",
            flush=True,
        )
    if len(X_te) == 0:
        raise RuntimeError(
            "No target test windows (tgt_test_X is empty). "
            "Check preprocessing / Alibaba val-test splits."
        )
    print(
        f"  Dataset: {len(X_tr):,} train, {len(X_te):,} test windows (W={W}).",
        flush=True,
    )
    kw    = dict(window_size=W, horizon=y_tr.shape[1], epochs=50, device=device)

    results = {}

    for name, cls, extra in [
        ("ARIMA",     ARIMABaseline,     {}),
        ("LSTM",      LSTMBaseline,      kw),
        ("GRU",       GRUBaseline,       kw),
        ("CNN-LSTM",  CNNLSTMBaseline,   kw),
        ("Autoformer",AutoformerBaseline,kw),
        ("BHT-ARIMA", BHTARIMABaseline,  {}),
        ("TS2Vec",    TS2VecBaseline,    kw),
    ]:
        print(f"  {name}...", end=" ", flush=True)
        m = cls(**extra) if extra else cls()
        m.fit(X_tr, y_tr)
        results[name] = evaluate_baseline(m, X_te, y_te, mctl_metrics)
        print("done")

    print("  WANN...", end=" ", flush=True)
    wann = WANNBaseline(**kw)
    wann.fit(X_src, y_src, X_tr, y_tr)
    results["WANN"] = evaluate_baseline(wann, X_te, y_te, mctl_metrics)
    print("done")

    print("  MCTL...", end=" ", flush=True)
    results["MCTL"] = evaluate_mctl(mctl_model, X_te, y_te, device)
    print("done")

    return results


# ─── Print tables ─────────────────────────────────────────────────────────────

def print_cwpdda_table(results, title="CWPDDA — CPU Workload Prediction"):
    print(f"\n{'='*58}", flush=True)
    print(f"  {title}", flush=True)
    print(f"  Target: MAE=2.4183  MAPE=8.66%  RMSE=2.5859", flush=True)
    print(f"{'='*58}", flush=True)
    if not results:
        print("  (no results to show)", flush=True)
        return
    print(f"{'Method':<12}  {'MAE':>8}  {'MAPE %':>8}  {'RMSE':>8}", flush=True)
    print("-" * 45, flush=True)
    for name, m in results.items():
        marker = " ←" if name == "CWPDDA" else ""
        print(f"{name:<12}  {m['MAE']:8.4f}  {m['MAPE_%']:8.2f}  {m['RMSE']:8.4f}{marker}", flush=True)
    print("-" * 45, flush=True)


def print_mctl_table(results, title="MCTL — Few-Shot Workload Prediction"):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"  Target: MAE=7.220E-4  MSE=9.857E-7  MAPE=2.575E-2")
    print(f"{'='*80}")
    print(f"{'Method':<12}  {'MAE':>12}  {'MSE':>12}  {'MAPE':>12}  {'sMAPE':>12}")
    print("-" * 65)
    for name, m in results.items():
        marker = " ←" if name == "MCTL" else ""
        print(f"{name:<12}  {m['MAE']:12.4E}  {m['MSE']:12.4E}  "
              f"{m['MAPE']:12.4E}  {m['sMAPE']:12.4E}{marker}")
    print("-" * 65)


# ─── N-BEATS evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_nbeats(model, X_test, y_test, device="cpu", infer_batch_size: int = 2048):
    """Evaluate N-BEATS using CWPDDA metrics (MAE, MAPE%, RMSE on 0-100 scale)."""
    model.eval()
    pred = model.predict_numpy_batched(X_test, device, batch_size=infer_batch_size)
    return cwpdda_metrics(y_test, pred)


def run_nbeats_comparison(
    nbeats_model,
    data,
    device="cpu",
    skip_gluonts=False,
    max_test_windows=None,
    subsample_seed: int = 42,
    partial_save_path=None,
):
    """
    Evaluate N-BEATS against standard baselines on Alibaba test set.

    N-BEATS is zero-shot: trained on Google, never sees Alibaba during training.
    Baselines (ARIMA, LSTM) are trained on Alibaba target train split so the
    comparison is intentionally favourable to baselines — N-BEATS beating them
    demonstrates the meta-learning / zero-shot transfer capability.

    Uses CWPDDA metrics (MAE, MAPE%, RMSE on 0-100 CPU utilisation scale).

    partial_save_path: if set, results are written to disk after every baseline
                       so a crash/timeout cannot lose already-computed metrics.
    """
    import json as _json
    from pathlib import Path as _Path
    from baselines import ARIMABaseline, LSTMBaseline

    X_tr  = data["tgt_train_X"]; y_tr  = data["tgt_train_y"]
    X_te  = data["tgt_test_X"];  y_te  = data["tgt_test_y"]
    W     = X_tr.shape[1]

    X_te, y_te, sub = _maybe_subsample_test(X_te, y_te, max_test_windows, subsample_seed)
    if sub:
        print(f"  Test subsampled to {len(X_te):,} windows (--eval-max-test).", flush=True)

    print(
        f"  Dataset: {len(X_tr):,} train windows (Alibaba, for baselines only), "
        f"{len(X_te):,} test windows (W={W}).",
        flush=True,
    )
    if len(X_te) == 0:
        raise RuntimeError("No target test windows. Check preprocessing / Alibaba val-test splits.")

    def _save(results):
        if partial_save_path:
            _Path(partial_save_path).write_text(_json.dumps(results, indent=2))

    results = {}

    print("  ARIMA...", end=" ", flush=True)
    m = ARIMABaseline(); m.fit(X_tr, y_tr)
    arima_n = min(500, len(X_te))
    idx = np.random.default_rng(subsample_seed).choice(len(X_te), arima_n, replace=False)
    results["ARIMA"] = evaluate_baseline(m, X_te[idx], y_te[idx], cwpdda_metrics)
    _save(results)
    print(f"done  (sampled {arima_n} windows)", flush=True)

    print("  LSTM...", end=" ", flush=True)
    kw = dict(window_size=W, horizon=y_tr.shape[1], epochs=150, device=device)
    m = LSTMBaseline(**kw); m.fit(X_tr, y_tr)
    results["LSTM"] = evaluate_baseline(m, X_te, y_te, cwpdda_metrics)
    _save(results)
    print("done", flush=True)

    if not skip_gluonts:
        try:
            from baselines import DeepARBaseline
            print("  DeepAR...", end=" ", flush=True)
            m = DeepARBaseline(prediction_length=y_tr.shape[1], epochs=10)
            m.fit(X_tr, y_tr)
            results["DeepAR"] = evaluate_baseline(m, X_te, y_te, cwpdda_metrics)
            _save(results)
            print("done", flush=True)
        except Exception as e:
            print(f"skipped ({e})", flush=True)

    print("  N-BEATS (zero-shot)...", end=" ", flush=True)
    results["N-BEATS"] = evaluate_nbeats(nbeats_model, X_te, y_te, device)
    _save(results)
    print("done", flush=True)

    return results


def print_nbeats_table(results, title="N-BEATS — Zero-Shot Workload Prediction (Google→Alibaba)"):
    print(f"\n{'='*62}", flush=True)
    print(f"  {title}", flush=True)
    print(f"  N-BEATS trained on Google only; applied zero-shot to Alibaba", flush=True)
    print(f"  Baselines trained on Alibaba target data (in-domain advantage)", flush=True)
    print(f"{'='*62}", flush=True)
    if not results:
        print("  (no results to show)", flush=True)
        return
    print(f"{'Method':<22}  {'MAE':>8}  {'MAPE %':>8}  {'RMSE':>8}", flush=True)
    print("-" * 54, flush=True)
    for name, m in results.items():
        marker = " ←" if name == "N-BEATS" else ""
        print(f"{name:<22}  {m['MAE']:8.4f}  {m['MAPE_%']:8.2f}  {m['RMSE']:8.4f}{marker}", flush=True)
    print("-" * 54, flush=True)
