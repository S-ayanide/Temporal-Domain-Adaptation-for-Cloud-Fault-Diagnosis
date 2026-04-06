"""
evaluate.py
===========
Computes the exact metrics reported in the MCTL paper (Tables 3, 4, 5):

  MAE    — Mean Absolute Error
  MSE    — Mean Squared Error
  MAPE   — Mean Absolute Percentage Error
  sMAPE  — Symmetric MAPE
  Var    — Variance of prediction errors

Also runs all baselines and prints a results table matching the paper layout.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


# ─── Metric functions ─────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)
    ))


def variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    errors = y_true - y_pred
    return float(np.var(errors))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    return {
        "MAE":      mae(y_true, y_pred),
        "MSE":      mse(y_true, y_pred),
        "MAPE":     mape(y_true, y_pred),
        "sMAPE":    smape(y_true, y_pred),
        "Variance": variance(y_true, y_pred),
    }


# ─── Evaluate a trained MCTL model ────────────────────────────────────────────

@torch.no_grad()
def evaluate_mctl(model, X_test: np.ndarray, y_test: np.ndarray,
                  device: str = "cpu") -> Dict[str, float]:
    model.eval()
    x = torch.from_numpy(X_test).float().to(device)
    y_pred = model.predict(x).cpu().numpy()
    return compute_metrics(y_test, y_pred)


# ─── Evaluate a numpy-based baseline (ARIMA etc.) ─────────────────────────────

def evaluate_baseline(model, X_test: np.ndarray,
                       y_test: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    return compute_metrics(y_test, y_pred)


# ─── Print results table ──────────────────────────────────────────────────────

def print_results_table(results: Dict[str, Dict[str, float]], title: str = ""):
    """Print a results table matching Table 3/4/5 layout in the MCTL paper."""
    if title:
        print(f"\n{'=' * 75}")
        print(f"  {title}")
        print(f"{'=' * 75}")

    header = f"{'Method':<14}  {'MAE':>12}  {'MSE':>12}  {'MAPE':>12}  {'sMAPE':>12}  {'Variance':>12}"
    print(header)
    print("-" * 75)

    for name, m in results.items():
        row = (
            f"{name:<14}  "
            f"{m['MAE']:12.4E}  "
            f"{m['MSE']:12.4E}  "
            f"{m['MAPE']:12.4E}  "
            f"{m['sMAPE']:12.4E}  "
            f"{m['Variance']:12.4E}"
        )
        print(row)
    print("-" * 75)


# ─── Run full comparison (all baselines + MCTL) ───────────────────────────────

def run_full_evaluation(
    mctl_model,
    data: dict,
    device: str = "cpu",
    skip_arima: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate all baselines + MCTL on the test set.
    Returns dict of {model_name: metrics_dict}.

    This replicates Tables 3+4+5 from the MCTL paper.
    """
    from models.baselines import (
        ARIMABaseline, LSTMBaseline, GRUBaseline, CNNLSTMBaseline,
        AutoformerBaseline, BHTARIMABaseline, WANNBaseline, TS2VecBaseline,
    )

    X_src   = data["src_X"]
    y_src   = data["src_y"]
    X_tr    = data["tgt_train_X"]
    y_tr    = data["tgt_train_y"]
    X_val   = data["tgt_val_X"]
    y_val   = data["tgt_val_y"]
    X_test  = data["tgt_test_X"]
    y_test  = data["tgt_test_y"]
    W       = X_tr.shape[1]
    H       = y_tr.shape[1]

    results: Dict[str, Dict[str, float]] = {}

    common_kwargs = dict(
        window_size=W, horizon=H,
        epochs=50, batch_size=64, device=device,
    )

    # --- ARIMA ---
    if not skip_arima:
        print("  Training ARIMA...", end=" ", flush=True)
        arima = ARIMABaseline()
        arima.fit(X_tr, y_tr)
        results["ARIMA"] = evaluate_baseline(arima, X_test, y_test)
        print("done")

    # --- LSTM ---
    print("  Training LSTM...", end=" ", flush=True)
    lstm = LSTMBaseline(**common_kwargs)
    lstm.fit(X_tr, y_tr)
    results["LSTM"] = evaluate_baseline(lstm, X_test, y_test)
    print("done")

    # --- GRU ---
    print("  Training GRU...", end=" ", flush=True)
    gru = GRUBaseline(**common_kwargs)
    gru.fit(X_tr, y_tr)
    results["GRU"] = evaluate_baseline(gru, X_test, y_test)
    print("done")

    # --- CNN-LSTM ---
    print("  Training CNN-LSTM...", end=" ", flush=True)
    cnnlstm = CNNLSTMBaseline(**common_kwargs)
    cnnlstm.fit(X_tr, y_tr)
    results["CNN-LSTM"] = evaluate_baseline(cnnlstm, X_test, y_test)
    print("done")

    # --- Autoformer ---
    print("  Training Autoformer...", end=" ", flush=True)
    af = AutoformerBaseline(**common_kwargs)
    af.fit(X_tr, y_tr)
    results["Autoformer"] = evaluate_baseline(af, X_test, y_test)
    print("done")

    # --- BHT-ARIMA ---
    if not skip_arima:
        print("  Training BHT-ARIMA...", end=" ", flush=True)
        bht = BHTARIMABaseline()
        bht.fit(X_tr, y_tr)
        results["BHT-ARIMA"] = evaluate_baseline(bht, X_test, y_test)
        print("done")

    # --- WANN ---
    print("  Training WANN...", end=" ", flush=True)
    wann = WANNBaseline(**common_kwargs, lambda_mmd=0.1)
    wann.fit(X_src, y_src, X_tr, y_tr)
    results["WANN"] = evaluate_baseline(wann, X_test, y_test)
    print("done")

    # --- TS2Vec ---
    print("  Training TS2Vec...", end=" ", flush=True)
    ts2 = TS2VecBaseline(**common_kwargs)
    ts2.fit(X_tr, y_tr)
    results["TS2Vec"] = evaluate_baseline(ts2, X_test, y_test)
    print("done")

    # --- MCTL ---
    print("  Evaluating MCTL...", end=" ", flush=True)
    results["MCTL"] = evaluate_mctl(mctl_model, X_test, y_test, device=device)
    print("done")

    return results
