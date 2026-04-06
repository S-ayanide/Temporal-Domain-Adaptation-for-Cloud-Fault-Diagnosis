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
from typing import Dict
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
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))

def smape(y_true, y_pred, eps=1e-8):
    return float(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)))

def variance_err(y_true, y_pred):
    return float(np.var(y_true - y_pred))


def cwpdda_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAE, MAPE (as %), RMSE — matches Table 3/4 of CWPDDA paper."""
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
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


# ─── Full comparison tables ───────────────────────────────────────────────────

def run_cwpdda_comparison(cwpdda_model, data, device="cpu", skip_gluonts=False):
    """
    Trains and evaluates all CWPDDA baselines + CWPDDA on the test set.
    Returns dict of {model_name: metrics}.
    """
    from models.baselines import (
        ARIMABaseline, LSTMBaseline,
        DeepARBaseline, DRPBaseline, MQF2Baseline,
    )

    X_tr  = data["tgt_train_X"]; y_tr  = data["tgt_train_y"]
    X_te  = data["tgt_test_X"];  y_te  = data["tgt_test_y"]
    W     = X_tr.shape[1]

    results = {}
    kw = dict(window_size=W, horizon=y_tr.shape[1], epochs=50, device=device)

    print("  ARIMA...",  end=" ", flush=True)
    m = ARIMABaseline(); m.fit(X_tr, y_tr)
    results["ARIMA"] = evaluate_baseline(m, X_te, y_te, cwpdda_metrics)
    print("done")

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


def run_mctl_comparison(mctl_model, data, device="cpu"):
    """Trains and evaluates all MCTL baselines + MCTL."""
    from models.baselines import (
        ARIMABaseline, LSTMBaseline, GRUBaseline, CNNLSTMBaseline,
        AutoformerBaseline, BHTARIMABaseline, WANNBaseline, TS2VecBaseline,
    )
    X_src = data["src_X"]; y_src = data["src_y"]
    X_tr  = data["tgt_train_X"]; y_tr  = data["tgt_train_y"]
    X_te  = data["tgt_test_X"];  y_te  = data["tgt_test_y"]
    W     = X_tr.shape[1]
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
    print(f"\n{'='*58}")
    print(f"  {title}")
    print(f"  Target: MAE=2.4183  MAPE=8.66%  RMSE=2.5859")
    print(f"{'='*58}")
    print(f"{'Method':<12}  {'MAE':>8}  {'MAPE %':>8}  {'RMSE':>8}")
    print("-" * 45)
    for name, m in results.items():
        marker = " ←" if name == "CWPDDA" else ""
        print(f"{name:<12}  {m['MAE']:8.4f}  {m['MAPE_%']:8.2f}  {m['RMSE']:8.4f}{marker}")
    print("-" * 45)


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
