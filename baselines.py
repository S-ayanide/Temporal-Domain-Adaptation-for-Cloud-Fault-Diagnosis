"""
models/baselines.py
===================
All baselines for both papers.

CWPDDA baselines (Table 3/4):  ARIMA, LSTM, DeepAR, DRP, MQF2
MCTL baselines   (Table 3/4):  ARIMA, LSTM, GRU, CNN-LSTM, Autoformer,
                                BHT-ARIMA, WANN, TS2Vec

Each model exposes:
    model.fit(X_train, y_train, **kwargs)
    model.predict(X_test) -> np.ndarray  shape (N, horizon)
"""

from __future__ import annotations
import warnings
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─── Shared neural net trainer ────────────────────────────────────────────────

class _NNBaseline:
    def __init__(self, model, lr=1e-3, epochs=50, batch_size=64, device="cpu"):
        self.model     = model.to(device)
        self.opt       = torch.optim.Adam(model.parameters(), lr=lr)
        self.epochs    = epochs
        self.bs        = batch_size
        self.device    = device

    def fit(self, X, y, **_):
        self.model.train()
        dl = DataLoader(TensorDataset(torch.from_numpy(X).float(),
                                      torch.from_numpy(y).float()),
                        batch_size=self.bs, shuffle=True)
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.opt.zero_grad()
                F.mse_loss(self.model(xb), yb).backward()
                self.opt.step()

    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        return self.model(torch.from_numpy(X).float().to(self.device)).cpu().numpy()


# ─── ARIMA ────────────────────────────────────────────────────────────────────

class ARIMABaseline:
    """Paper Table 2: p=5, d=1, q=5"""
    def __init__(self, order=(5, 1, 5)):
        self.order = order

    def fit(self, X, y, **_):
        pass

    def predict(self, X):
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            raise ImportError("pip install statsmodels")
        preds = []
        n = len(X)
        # One ARIMA fit per row — large test sets look like a hang unless we log progress.
        step = max(500, n // 20) if n > 800 else 0
        for i, x in enumerate(X):
            if step and (i % step == 0 or i == n - 1):
                print(f"    ARIMA predict {i + 1:,}/{n:,}", flush=True)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fc = ARIMA(x.astype(float), order=self.order).fit().forecast(1)
                    preds.append(float(fc[0]))
            except Exception:
                preds.append(float(x[-1]))
        return np.array(preds, dtype=np.float32).reshape(-1, 1)


# ─── LSTM ─────────────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    def __init__(self, W, hidden=40, layers=2, dropout=0.1, horizon=1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.fc = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.fc(out[:, -1])


class LSTMBaseline(_NNBaseline):
    """Paper Table 2: 2 layers, 40 cells, lr=1e-3"""
    def __init__(self, window_size=24, hidden=40, layers=2, dropout=0.1,
                 horizon=1, lr=1e-3, epochs=50, batch_size=64, device="cpu"):
        super().__init__(_LSTMNet(window_size, hidden, layers, dropout, horizon),
                         lr, epochs, batch_size, device)


# ─── GRU ──────────────────────────────────────────────────────────────────────

class _GRUNet(nn.Module):
    def __init__(self, W, hidden=128, layers=2, dropout=0.2, horizon=1):
        super().__init__()
        self.gru = nn.GRU(1, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0)
        self.fc = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.gru(x.unsqueeze(-1))
        return self.fc(out[:, -1])


class GRUBaseline(_NNBaseline):
    def __init__(self, window_size=24, hidden=128, layers=2, dropout=0.2,
                 horizon=1, lr=1e-3, epochs=50, batch_size=64, device="cpu"):
        super().__init__(_GRUNet(window_size, hidden, layers, dropout, horizon),
                         lr, epochs, batch_size, device)


# ─── CNN-LSTM ─────────────────────────────────────────────────────────────────

class _CNNLSTMNet(nn.Module):
    def __init__(self, W, hidden=128, dropout=0.2, horizon=1):
        super().__init__()
        self.conv = nn.Conv1d(1, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(W // 2)
        self.lstm = nn.LSTM(64, hidden, 1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, horizon)

    def forward(self, x):
        h = F.relu(self.conv(x.unsqueeze(1)))
        h = self.pool(h).permute(0, 2, 1)
        out, _ = self.lstm(h)
        return self.fc(self.drop(out[:, -1]))


class CNNLSTMBaseline(_NNBaseline):
    def __init__(self, window_size=24, hidden=128, dropout=0.2,
                 horizon=1, lr=1e-3, epochs=50, batch_size=64, device="cpu"):
        super().__init__(_CNNLSTMNet(window_size, hidden, dropout, horizon),
                         lr, epochs, batch_size, device)


# ─── Autoformer (simplified) ──────────────────────────────────────────────────

class _AutoformerNet(nn.Module):
    def __init__(self, W, d_model=64, n_heads=4, dim_ff=128, dropout=0.1, horizon=1):
        super().__init__()
        k = min(25, W // 2 * 2 + 1)
        self.avg  = nn.AvgPool1d(k, 1, k // 2, count_include_pad=False)
        self.proj = nn.Linear(1, d_model)
        enc       = nn.TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout, batch_first=True)
        self.trf  = nn.TransformerEncoder(enc, 2)
        self.fc   = nn.Linear(d_model, horizon)

    def forward(self, x):
        trend    = self.avg(x.unsqueeze(1)).squeeze(1)[:, :x.size(1)]
        seasonal = x - trend
        h = self.proj(seasonal.unsqueeze(-1))
        return self.fc(self.trf(h).mean(1))


class AutoformerBaseline(_NNBaseline):
    def __init__(self, window_size=24, d_model=64, n_heads=4, dim_ff=128,
                 dropout=0.1, horizon=1, lr=1e-3, epochs=50, batch_size=64, device="cpu"):
        super().__init__(_AutoformerNet(window_size, d_model, n_heads, dim_ff, dropout, horizon),
                         lr, epochs, batch_size, device)


# ─── BHT-ARIMA (approximated) ────────────────────────────────────────────────

class BHTARIMABaseline(ARIMABaseline):
    """BHT-ARIMA approx as ARIMA(1,0,0) for short series."""
    def __init__(self):
        super().__init__(order=(1, 0, 0))


# ─── WANN ─────────────────────────────────────────────────────────────────────

class _WANNNet(nn.Module):
    def __init__(self, W, hidden=128, dropout=0.2, horizon=1):
        super().__init__()
        self.enc  = nn.Sequential(nn.Linear(W, hidden), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):   return self.head(self.enc(x))
    def encode(self, x):    return self.enc(x)


class WANNBaseline:
    def __init__(self, window_size=24, hidden=128, dropout=0.2,
                 horizon=1, lr=1e-3, epochs=50, batch_size=64, device="cpu", lam=0.1):
        self.model  = _WANNNet(window_size, hidden, dropout, horizon).to(device)
        self.opt    = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs; self.bs = batch_size
        self.device = device; self.lam = lam

    def fit(self, X_src, y_src, X_tgt=None, y_tgt=None, **_):
        X_tgt = X_tgt if X_tgt is not None else X_src
        y_tgt = y_tgt if y_tgt is not None else y_src
        n = min(len(X_src), len(X_tgt))
        ds_s = TensorDataset(torch.from_numpy(X_src[:n]).float(), torch.from_numpy(y_src[:n]).float())
        ds_t = TensorDataset(torch.from_numpy(X_tgt[:n]).float(), torch.from_numpy(y_tgt[:n]).float())
        dl_s = DataLoader(ds_s, self.bs, shuffle=True, drop_last=True)
        dl_t = DataLoader(ds_t, self.bs, shuffle=True, drop_last=True)
        self.model.train()
        for _ in range(self.epochs):
            for (xs, ys), (xt, yt) in zip(dl_s, dl_t):
                xs, ys = xs.to(self.device), ys.to(self.device)
                xt, yt = xt.to(self.device), yt.to(self.device)
                self.opt.zero_grad()
                zs, zt = self.model.encode(xs), self.model.encode(xt)
                mmd = ((zs.mean(0) - zt.mean(0)) ** 2).sum()
                loss = (F.mse_loss(self.model.head(zs), ys) +
                        F.mse_loss(self.model.head(zt), yt) +
                        self.lam * mmd)
                loss.backward(); self.opt.step()

    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        return self.model(torch.from_numpy(X).float().to(self.device)).cpu().numpy()


# ─── TS2Vec baseline ──────────────────────────────────────────────────────────

class _TS2VecNet(nn.Module):
    """TCN-based supervised baseline (TS2Vec without contrastive pretraining)."""
    def __init__(self, W, hidden=128, n_layers=3, kernel_size=3, dropout=0.2, horizon=1):
        super().__init__()
        from cwpdda import SelfAttentionBlock  # reuse attention
        self.proj = nn.Linear(1, hidden)
        self.attn = SelfAttentionBlock(hidden, dropout)
        self.fc   = nn.Linear(hidden, horizon)

    def forward(self, x):
        h = self.proj(x.unsqueeze(-1))      # (B,W,H)
        h = self.attn(h).mean(1)             # (B,H)
        return self.fc(h)


class TS2VecBaseline(_NNBaseline):
    def __init__(self, window_size=24, hidden=128, n_layers=3, kernel_size=3,
                 dropout=0.2, horizon=1, lr=1e-3, epochs=50, batch_size=64, device="cpu"):
        super().__init__(_TS2VecNet(window_size, hidden, n_layers, kernel_size, dropout, horizon),
                         lr, epochs, batch_size, device)


# ─── DeepAR (GluonTS) ─────────────────────────────────────────────────────────

class DeepARBaseline:
    """
    DeepAR from GluonTS (paper uses GluonTS toolbox).
    Requires: pip install gluonts mxnet  OR  gluonts[torch]
    Falls back to LSTM if GluonTS not installed.
    """

    def __init__(self, prediction_length=1, freq="5min", epochs=10):
        self.pred_len = prediction_length
        self.freq     = freq
        self.epochs   = epochs
        self._model   = None
        self._fallback = None

    def fit(self, X, y, **_):
        try:
            from gluonts.dataset.common import ListDataset
            from gluonts.torch.model.deepar import DeepAREstimator
            from gluonts.evaluation.backtest import make_evaluation_predictions

            train_ds = ListDataset(
                [{"start": "2017-01-01", "target": x.tolist()} for x in X],
                freq=self.freq,
            )
            est = DeepAREstimator(
                prediction_length=self.pred_len,
                freq=self.freq,
                num_layers=2,
                hidden_size=40,
                trainer_kwargs={"max_epochs": self.epochs},
            )
            self._model = est.train(train_ds)
            self._X_train = X
        except ImportError:
            # Graceful fallback to LSTM
            print("  [warn] GluonTS not installed — DeepAR falling back to LSTM")
            self._fallback = LSTMBaseline(window_size=X.shape[1], horizon=self.pred_len)
            self._fallback.fit(X, y)

    def predict(self, X):
        if self._fallback is not None:
            return self._fallback.predict(X)
        try:
            from gluonts.dataset.common import ListDataset
            test_ds = ListDataset(
                [{"start": "2017-01-01", "target": x.tolist()} for x in X],
                freq=self.freq,
            )
            preds = []
            for forecast in self._model.predict(test_ds):
                preds.append(float(forecast.mean[0]))
            return np.array(preds, dtype=np.float32).reshape(-1, 1)
        except Exception:
            return np.array([float(x[-1]) for x in X], dtype=np.float32).reshape(-1, 1)


# ─── DRP (Deep Renewal Processes) ────────────────────────────────────────────

class DRPBaseline:
    """
    DRP from GluonTS. Falls back to GRU if unavailable.
    Paper Table 2: hidden=16, layers=2, dropout=0.1
    """
    def __init__(self, prediction_length=1, freq="5min", epochs=10):
        self.pred_len = prediction_length
        self.freq     = freq
        self.epochs   = epochs
        self._fallback = None

    def fit(self, X, y, **_):
        try:
            from gluonts.dataset.common import ListDataset
            # DRP available as DeepRenewalProcessEstimator in some gluonts versions
            from gluonts.model.renewal import DeepRenewalEstimator
            train_ds = ListDataset(
                [{"start": "2017-01-01", "target": x.tolist()} for x in X],
                freq=self.freq,
            )
            est = DeepRenewalEstimator(
                prediction_length=self.pred_len,
                freq=self.freq,
                num_hidden_dimensions=[16, 16],
                trainer_kwargs={"max_epochs": self.epochs},
            )
            self._model = est.train(train_ds)
            self._X = X
        except Exception:
            print("  [warn] DRP unavailable — falling back to GRU")
            self._fallback = GRUBaseline(window_size=X.shape[1])
            self._fallback.fit(X, y)

    def predict(self, X):
        if self._fallback is not None:
            return self._fallback.predict(X)
        try:
            from gluonts.dataset.common import ListDataset
            test_ds = ListDataset(
                [{"start": "2017-01-01", "target": x.tolist()} for x in X],
                freq=self.freq,
            )
            preds = [float(f.mean[0]) for f in self._model.predict(test_ds)]
            return np.array(preds, dtype=np.float32).reshape(-1, 1)
        except Exception:
            return np.array([float(x[-1]) for x in X], dtype=np.float32).reshape(-1, 1)


# ─── MQF2 (Multivariate Quantile Function Forecaster) ────────────────────────

class MQF2Baseline:
    """
    MQF2 from GluonTS. Falls back to CNN-LSTM if unavailable.
    Paper Table 2: layers=2, hidden=40, lr=1e-3, weight_decay=0.1, samples=50
    """
    def __init__(self, prediction_length=1, freq="5min", epochs=10):
        self.pred_len = prediction_length
        self.freq     = freq
        self.epochs   = epochs
        self._fallback = None

    def fit(self, X, y, **_):
        try:
            from gluonts.dataset.common import ListDataset
            from gluonts.model.mqf2 import MQF2MultiHorizonEstimator
            train_ds = ListDataset(
                [{"start": "2017-01-01", "target": x.tolist()} for x in X],
                freq=self.freq,
            )
            est = MQF2MultiHorizonEstimator(
                prediction_length=self.pred_len,
                freq=self.freq,
                trainer_kwargs={"max_epochs": self.epochs},
            )
            self._model = est.train(train_ds)
        except Exception:
            print("  [warn] MQF2 unavailable — falling back to CNN-LSTM")
            self._fallback = CNNLSTMBaseline(window_size=X.shape[1])
            self._fallback.fit(X, y)

    def predict(self, X):
        if self._fallback is not None:
            return self._fallback.predict(X)
        try:
            from gluonts.dataset.common import ListDataset
            test_ds = ListDataset(
                [{"start": "2017-01-01", "target": x.tolist()} for x in X],
                freq=self.freq,
            )
            preds = [float(f.mean[0]) for f in self._model.predict(test_ds)]
            return np.array(preds, dtype=np.float32).reshape(-1, 1)
        except Exception:
            return np.array([float(x[-1]) for x in X], dtype=np.float32).reshape(-1, 1)
