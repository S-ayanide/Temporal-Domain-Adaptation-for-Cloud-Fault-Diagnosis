"""
tr_adaboost.py — Two-Stage TrAdaBoost.R2-LSTM (Algorithm 2, Liu et al. 2022).

Algorithm overview
------------------
Given:
    source data  Ts = {(x_i, y_i)}_{i=1..n}   (large, many time series)
    target data  Tt = {(x_j, y_j)}_{j=1..m}   (small, scarce)
    T = total boosting rounds

Stage 1  (rounds 1 .. ceil(T/2)):
    Freeze source weights (do not update w_s).
    Only update target weights using AdaBoost.R2 error formula.

Stage 2  (rounds ceil(T/2)+1 .. T):
    Freeze target weights (do not update w_t).
    Decrease source weights: w_s ← w_s × β_s
    where β_s = 1 / (1 + sqrt(2 ln(n) / T)).

Final ensemble: majority vote over last ceil(T/2) hypotheses
    (weighted by log(1/β_t^(r))).

Reference: Algorithm 2 in §3.3 of Liu et al. (2022).
"""

import math
import numpy as np
import copy
import tensorflow as tf
from tensorflow import keras

from lstm_model import build_weak_learner, train_weak_learner, predict_weak_learner


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _adaboost_r2_error(pred: np.ndarray, target: np.ndarray,
                        weights: np.ndarray) -> float:
    """
    Weighted relative error for AdaBoost.R2.

    e_t = Σ_i w_i |h_t(x_i) - y_i| / D_max
    where D_max = max |h_t(x_i) - y_i|.
    """
    diff = np.abs(pred - target)
    d_max = diff.max() + 1e-12
    e = float(np.sum(weights * (diff / d_max)))
    return e


def _update_weights(weights: np.ndarray, pred: np.ndarray,
                    target: np.ndarray, beta: float) -> np.ndarray:
    """
    Update rule: w_i ← w_i × β^(1 - |h(x_i) - y_i| / D_max)
    Renormalise to sum = 1.
    """
    diff = np.abs(pred - target)
    d_max = diff.max() + 1e-12
    exponent = 1.0 - diff / d_max
    new_w = weights * (beta ** exponent)
    new_w /= new_w.sum() + 1e-12
    return new_w


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TrAdaBoostLSTM:
    """
    Two-stage TrAdaBoost.R2 with LSTM weak learners.

    Parameters
    ----------
    n_rounds    : total boosting rounds T (paper default 20)
    lstm_units  : hidden size for each LSTM weak learner
    dense_units : dense layer size
    seq_len     : input window length
    n_features  : number of input channels
    n_targets   : forecast horizon
    dropout     : dropout inside weak learner
    lr          : learning rate
    max_epochs  : epochs per weak learner
    batch_size  : mini-batch size
    patience    : early-stopping patience
    verbose     : 0/1 Keras verbosity
    """

    def __init__(
        self,
        n_rounds: int = 20,
        lstm_units: int = 64,
        dense_units: int = 32,
        seq_len: int = 24,
        n_features: int = 1,
        n_targets: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
        max_epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        verbose: int = 0,
    ):
        self.n_rounds    = n_rounds
        self.lstm_units  = lstm_units
        self.dense_units = dense_units
        self.seq_len     = seq_len
        self.n_features  = n_features
        self.n_targets   = n_targets
        self.dropout     = dropout
        self.lr          = lr
        self.max_epochs  = max_epochs
        self.batch_size  = batch_size
        self.patience    = patience
        self.verbose     = verbose

        self._hypotheses  = []   # list of (model, beta_t, round_idx)
        self._fitted      = False

    # ------------------------------------------------------------------
    def _make_model(self) -> keras.Model:
        return build_weak_learner(
            seq_len=self.seq_len,
            n_features=self.n_features,
            n_targets=self.n_targets,
            lstm_units=self.lstm_units,
            dense_units=self.dense_units,
            dropout=self.dropout,
            learning_rate=self.lr,
        )

    # ------------------------------------------------------------------
    def fit(
        self,
        X_src: np.ndarray, Y_src: np.ndarray,
        X_tgt: np.ndarray, Y_tgt: np.ndarray,
        X_val: np.ndarray = None, Y_val: np.ndarray = None,
    ):
        """
        Train the TrAdaBoost ensemble.

        Parameters
        ----------
        X_src : (n, seq_len, n_features)  source training windows
        Y_src : (n, n_targets)            source labels
        X_tgt : (m, seq_len, n_features)  target training windows
        Y_tgt : (m, n_targets)            target labels
        X_val, Y_val : optional validation set (target domain only)
        """
        n = len(X_src)   # source size
        m = len(X_tgt)   # target size
        N = n + m        # combined

        T = self.n_rounds
        stage1_end = math.ceil(T / 2)

        # β_s — fixed decay for source weights in stage 2
        beta_s = 1.0 / (1.0 + math.sqrt(2.0 * math.log(n) / T))

        # Initial uniform weights over all N samples
        w = np.ones(N, dtype=np.float64) / N
        # Convention: w[0..n-1] = source, w[n..n+m-1] = target

        self._hypotheses = []

        X_all = np.concatenate([X_src, X_tgt], axis=0)
        Y_all = np.concatenate([Y_src, Y_tgt], axis=0)

        for t in range(1, T + 1):
            if self.verbose:
                print(f"  Round {t}/{T}  (stage {'1' if t <= stage1_end else '2'})", end="  ")

            # Normalise weights
            w_norm = w / w.sum()

            # ---- Train weak learner on all data with current weights ----
            model = self._make_model()
            train_weak_learner(
                model, X_all, Y_all,
                weights=w_norm,
                X_val=X_val, Y_val=Y_val,
                batch_size=self.batch_size,
                max_epochs=self.max_epochs,
                patience=self.patience,
                verbose=0,
            )

            # ---- Error on target data only ----
            pred_tgt = predict_weak_learner(model, X_tgt)
            # Squeeze to 1-D if single target for error computation
            if self.n_targets == 1:
                pred_tgt_1d = pred_tgt.ravel()
                ytgt_1d     = Y_tgt.ravel()
            else:
                # Use mean across targets for weight update
                pred_tgt_1d = pred_tgt.mean(axis=1)
                ytgt_1d     = Y_tgt.mean(axis=1)

            w_tgt = w_norm[n:]   # target portion of weights
            e_t   = _adaboost_r2_error(pred_tgt_1d, ytgt_1d, w_tgt)

            # Clamp to valid range
            e_t = float(np.clip(e_t, 1e-10, 1.0 - 1e-10))
            beta_t = e_t / (1.0 - e_t)

            if self.verbose:
                print(f"e={e_t:.4f}  beta={beta_t:.4f}")

            # Keep hypothesis
            self._hypotheses.append({
                "model":  model,
                "beta_t": beta_t,
                "round":  t,
            })

            # ---- Weight update ----
            if t <= stage1_end:
                # Stage 1: update target weights only (source frozen)
                pred_all = predict_weak_learner(model, X_all)
                if self.n_targets == 1:
                    pred_all_1d = pred_all.ravel()
                    yall_1d     = Y_all.ravel()
                else:
                    pred_all_1d = pred_all.mean(axis=1)
                    yall_1d     = Y_all.mean(axis=1)

                # Update only target slice
                w_tgt_new = _update_weights(
                    w_norm[n:], pred_tgt_1d, ytgt_1d, beta_t
                )
                w[n:] = w_tgt_new * w_tgt_new.sum()   # re-scale before combined norm
            else:
                # Stage 2: decrease source weights, freeze target
                w[:n] *= beta_s

            # Renormalise
            w /= w.sum() + 1e-12

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Ensemble prediction via weighted median over last ceil(T/2) hypotheses.

        Returns
        -------
        np.ndarray, shape (N, n_targets)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        T = self.n_rounds
        stage1_end = math.ceil(T / 2)

        # Use only stage-2 hypotheses (rounds > stage1_end)
        stage2 = [h for h in self._hypotheses if h["round"] > stage1_end]
        if not stage2:
            stage2 = self._hypotheses   # fallback: use all

        preds   = []
        log_inv = []
        for h in stage2:
            p   = predict_weak_learner(h["model"], X)   # (N, n_targets)
            bt  = h["beta_t"]
            w_h = math.log(1.0 / (bt + 1e-10))
            preds.append(p)
            log_inv.append(w_h)

        preds   = np.stack(preds, axis=0)   # (R, N, n_targets)
        log_inv = np.array(log_inv)

        # Normalise log weights → sum to 1 per sample
        log_inv = np.clip(log_inv, 0.0, None)  # discard negative (bad) hypotheses
        if log_inv.sum() < 1e-12:
            log_inv = np.ones(len(stage2))
        log_inv /= log_inv.sum()

        # Weighted average across hypotheses
        ensemble_pred = np.einsum("r,rni->ni", log_inv, preds)
        return ensemble_pred   # (N, n_targets)

    # ------------------------------------------------------------------
    def predict_cpu(self, X: np.ndarray) -> np.ndarray:
        """Convenience: return only first target column."""
        return self.predict(X)[:, 0]
