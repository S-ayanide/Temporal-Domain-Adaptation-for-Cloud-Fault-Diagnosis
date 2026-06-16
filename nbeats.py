"""
nbeats.py
=========
N-BEATS zero-shot workload prediction — Oreshkin et al., AAAI 2021.
"Meta-learning framework with applications to zero-shot time-series forecasting"

Key idea: Train on Google cluster source data only; evaluate directly on
Alibaba cluster data without any fine-tuning (zero-shot cross-domain transfer).

The meta-learning interpretation of N-BEATS (Section 3 of paper):
  Each block implements one inner-loop update step. Shared residual connections
  carry task-specific context (the shifted input mu_l) across blocks.
  After L blocks, the model has effectively run L gradient-like adaptation steps.

Architecture (Appendix D.1):
  - L blocks with doubly residual connections
  - Each block: K fully-connected layers + ReLU → backcast head + forecast head
  - Backcast: reconstructs (explains) the current residual input
  - Forecast: produces a partial prediction for the horizon
  - Next residual: x_{l+1} = x_l − backcast_l
  - Final forecast: y_hat = sum over l of forecast_l

Two variants:
  - NSH (non-shared, default): each block has independent parameters — more
    expressive, learns different pattern types per block
  - SH (shared): all blocks share one set of parameters — pure meta-learning
    effect from residual structure; more parameter-efficient

Cross-domain transfer mechanism — MaxAbs per-window scaling:
  Divide the input window by its own absolute max before passing to the network,
  then multiply the network's output by the same scale factor. This normalises
  amplitude differences between Google and Alibaba workloads so that a model
  trained on Google can be applied to Alibaba windows with very different scales.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─── N-BEATS block ────────────────────────────────────────────────────────────

class NBeatsBlock(nn.Module):
    """
    Single N-BEATS block (Section 3 / Figure 1 of paper).

    FC stack → two linear heads:
      backcast:  projects to input space (explains / removes the current residual)
      forecast:  projects to horizon space (partial prediction)
    """

    def __init__(self, input_size: int, hidden_size: int, n_layers: int,
                 forecast_size: int, dropout: float = 0.0):
        super().__init__()
        fc = []
        for i in range(n_layers):
            in_dim = input_size if i == 0 else hidden_size
            fc.append(nn.Linear(in_dim, hidden_size))
            fc.append(nn.ReLU())
            if dropout > 0.0:
                fc.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*fc)
        self.backcast_head = nn.Linear(hidden_size, input_size)
        self.forecast_head = nn.Linear(hidden_size, forecast_size)

    def forward(self, x: torch.Tensor):
        """x: (B, W) → backcast: (B, W), forecast: (B, H)"""
        h = self.fc(x)
        return self.backcast_head(h), self.forecast_head(h)


# ─── N-BEATS model ────────────────────────────────────────────────────────────

class NBeats(nn.Module):
    """
    N-BEATS for zero-shot cloud workload prediction.

    Training: source domain (Google) only — no target data used.
    Inference: directly on target domain (Alibaba) — no fine-tuning.

    Args:
        window_size:  Input lookback length (default 24 = matches CWPDDA/MCTL)
        horizon:      Steps ahead to predict (default 1)
        n_blocks:     Number of stacked blocks L (paper uses 30; 8-12 is practical)
        n_layers:     FC layers per block K (paper uses 4)
        hidden_size:  Hidden units per FC layer (paper uses 512; 256 is practical)
        shared_weights: If True, all blocks share one parameter set (SH variant);
                        if False (default), each block has own params (NSH variant)
        dropout:      Dropout rate within FC stack (0 = off, as in paper)
    """

    def __init__(
        self,
        window_size: int = 24,
        horizon: int = 1,
        n_blocks: int = 8,
        n_layers: int = 4,
        hidden_size: int = 256,
        shared_weights: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.horizon = horizon
        self.n_blocks = n_blocks
        self.shared_weights = shared_weights

        if shared_weights:
            # SH: one block reused L times; blocks list still holds L references
            # so the doubly-residual loop is clean and parameter count is minimal
            single = NBeatsBlock(window_size, hidden_size, n_layers, horizon, dropout)
            self.blocks = nn.ModuleList([single] * n_blocks)
        else:
            # NSH: each block has its own parameters (default, more expressive)
            self.blocks = nn.ModuleList([
                NBeatsBlock(window_size, hidden_size, n_layers, horizon, dropout)
                for _ in range(n_blocks)
            ])

    # ── MaxAbs per-window scaling (critical for zero-shot cross-domain transfer) ──

    @staticmethod
    def maxabs_scale(x: torch.Tensor):
        """
        Normalise each window by its absolute maximum value.
        Returns (x_norm, scale) where scale has shape (B, 1).
        At inference: multiply the raw output by scale to recover original units.
        """
        scale = x.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        return x / scale, scale

    # ── Forward pass ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, W) — raw (normalised-to-[0,1]) workload windows
        returns forecast: (B, H)

        Doubly residual stack:
          residual_0 = MaxAbs(x)
          for block l in 1..L:
              backcast_l, forecast_l = block_l(residual_{l-1})
              residual_l = residual_{l-1} − backcast_l
              accumulated_forecast += forecast_l
          output = accumulated_forecast * scale
        """
        x_norm, scale = self.maxabs_scale(x)
        residual = x_norm
        forecast = torch.zeros(x.size(0), self.horizon, device=x.device, dtype=x.dtype)

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        # Rescale output back to original [0,1] normalised range
        return forecast * scale

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Same as forward. Required by evaluate.py's evaluate_baseline()."""
        return self.forward(x)

    @torch.no_grad()
    def predict_numpy_batched(
        self,
        X: np.ndarray,
        device: str = "cpu",
        batch_size: int = 2048,
    ) -> np.ndarray:
        """
        Batch inference over a numpy array. Required by evaluate.py's
        evaluate_cwpdda() which calls model.predict_numpy_batched(...).

        Returns: (N, H) float32 numpy array.
        """
        self.eval()
        parts = []
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).float().to(device)
            parts.append(self.forward(xb).cpu().numpy())
        return np.concatenate(parts, axis=0) if parts else np.empty((0, self.horizon), dtype=np.float32)
