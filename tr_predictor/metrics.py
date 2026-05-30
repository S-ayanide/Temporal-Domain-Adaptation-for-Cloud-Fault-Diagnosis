"""
metrics.py — Evaluation metrics for Tr-Predictor (Liu et al. 2022).

Paper uses: MAPE, MAE, MSE, R² (Table 2 and throughout).
"""

import numpy as np


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((pred - target) ** 2))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(pred - target)))


def mape(pred: np.ndarray, target: np.ndarray,
         eps: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error (%).

    Note: targets near zero inflate MAPE; we clip denominator by eps.
    """
    return float(np.mean(np.abs((pred - target) / (np.abs(target) + eps))) * 100.0)


def r2(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Coefficient of Determination R².

    R² = 1 - SS_res / SS_tot
    Clamped to [-inf, 1]; negative means worse than predicting the mean.
    """
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)


def evaluate_all(pred: np.ndarray, target: np.ndarray,
                 eps: float = 1e-8) -> dict:
    """
    Compute all metrics for a (pred, target) pair.

    Both arrays should be 1-D or will be flattened.
    Returns dict with keys: mse, mae, mape, r2.
    """
    p = pred.ravel()
    t = target.ravel()
    return {
        "mse":  mse(p, t),
        "mae":  mae(p, t),
        "mape": mape(p, t, eps),
        "r2":   r2(p, t),
    }


def print_metrics(name: str, metrics: dict):
    """Pretty-print a metrics dict."""
    print(f"{name:<30}  "
          f"MSE={metrics['mse']:.5f}  "
          f"MAE={metrics['mae']:.5f}  "
          f"MAPE={metrics['mape']:.2f}%  "
          f"R²={metrics['r2']:.4f}")
