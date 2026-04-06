"""
models/tcn.py
=============
Temporal Convolutional Network (TCN) encoder used in the MCTL paper.

Architecture (Section 3.3 of paper):
  - 1-D dilated causal convolution layers
  - Each layer: Conv1d -> ReLU -> Dropout (weight norm applied)
  - Residual connections
  - Global average pooling at the end
  - 3 convolutional layers (paper default)

Input:  (batch, window_size)  — 1-D time series window
Output: (batch, hidden_dim)   — encoded representation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class _CausalConv1d(nn.Module):
    """
    Single dilated causal convolution block.
    Padding is applied only to the left (past) to maintain causality.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        # Weight norm is applied as in the original TCN paper
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.padding,
                dilation=dilation,
            )
        )
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.net     = nn.Sequential(self.conv, self.relu, self.dropout)

        # Residual 1x1 conv if channel sizes differ
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self._init_weights()

    def _init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        out = self.net(x)
        # Remove the left-padding overhang to restore original length
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)


class TCNEncoder(nn.Module):
    """
    TCN encoder from the MCTL paper.

    Paper uses:
      n_layers = 3, hidden_dim = 128 (filter size), kernel_size = 3
      n_filters chosen from [64, 128, 256], kernel from [1, 2, 3]

    Input:  (batch, window_size)          — raw normalised series
    Output: (batch, hidden_dim)           — fixed-size representation
    """

    def __init__(
        self,
        window_size: int = 24,
        hidden_dim: int = 128,
        n_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim  = hidden_dim

        layers = []
        for i in range(n_layers):
            dilation  = 2 ** i
            in_ch     = 1 if i == 0 else hidden_dim
            out_ch    = hidden_dim
            layers.append(
                _CausalConv1d(in_ch, out_ch, kernel_size, dilation, dropout)
            )
        self.tcn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, window_size)  — 1-D input
        Returns: (batch, hidden_dim)
        """
        # TCN expects (batch, channels, time)
        h = x.unsqueeze(1)          # (batch, 1, W)
        h = self.tcn(h)             # (batch, hidden_dim, W)
        # Global average pooling over time (paper Section 3.3)
        return h.mean(dim=2)        # (batch, hidden_dim)


class TCNPredictor(nn.Module):
    """
    Full prediction model: TCN encoder + linear regression head.
    Used as the target model after transfer.
    """

    def __init__(
        self,
        window_size: int = 24,
        hidden_dim: int = 128,
        n_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
        horizon: int = 1,
    ):
        super().__init__()
        self.encoder = TCNEncoder(window_size, hidden_dim, n_layers, kernel_size, dropout)
        self.head    = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, W) → (batch, horizon)"""
        return self.head(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return representation without the prediction head."""
        return self.encoder(x)
