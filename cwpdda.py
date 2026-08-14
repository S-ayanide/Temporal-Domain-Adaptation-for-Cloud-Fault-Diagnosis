"""
models/cwpdda.py
================
Container Workload Prediction using Deep Domain Adaptation (CWPDDA)
Wang et al., Euro-Par 2025 — pp. 322-336

Architecture (Figure 1 of paper):
    Input: source + target CPU/memory time series windows

    1. Feature Extractor  Gf(·)   [Section 3.1]
       - Source branch:        Self-Attention → private source features
       - Target branch:        Self-Attention → private target features
       - Source-Target branch: Cross-Attention → shared features
         (Q from source, K and V from target)
       - Loss: maximise MMD(self-attn, cross-attn) so private ≠ shared

    2. Domain Adversarial Adapter  Gd(·)  [Section 3.2]
       - Takes shared features from Gf
       - Sigmoid classifier: prob feature is from target domain
       - Gradient Reversal Layer (GRL) with coefficient λ_GRL
       - Loss: negative log-likelihood domain classification

    3. Workload Predictor  Gy(·)  [Section 3.3]
       - Multi-layer LSTM on shared features
       - Fully-connected output
       - Loss: MSE on target predictions

Total optimisation:
    min  Ly + Lf
    max  Ld   (via GRL so it becomes min -Ld in the feature extractor)

Metrics reported in paper (Table 3 / 4):
    CPU:    MAE=2.4183, MAPE=8.66%, RMSE=2.5859
    Memory: MAE=4.8528, MAPE=9.11%, RMSE=5.9989
"""

from __future__ import annotations
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Gradient Reversal Layer ──────────────────────────────────────────────────

class GRL(torch.autograd.Function):
    """Multiplies gradient by -λ on the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.lam * grad, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return GRL.apply(x, lam)


def grl_lambda(step: int, total_steps: int,
               alpha: float = 10.0, beta: float = 0.75) -> float:
    """
    Adaptive GRL coefficient from the DANN paper (Eq. 13 in CWPDDA paper).
    α=10, β=0.75 as listed in Table 2 of the paper.
    """
    p = step / max(total_steps, 1)
    return float(2.0 / (1.0 + math.exp(-alpha * p)) - 1.0) ** beta


# ─── Self-Attention block ─────────────────────────────────────────────────────

class SelfAttentionBlock(nn.Module):
    """
    Self-attention (paper Eqs. 2-5). Uses nn.MultiheadAttention for
    memory efficiency — avoids materialising the full (B, seq, seq) matrix.
    Input:  (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # num_heads=1 matches the single-head formulation in the paper
        self.attn = nn.MultiheadAttention(d_model, num_heads=1,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + out)


# ─── Cross-Attention block ────────────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention: Q from target, K/V from source (Eqs. 6-9).
    Q=target preserves temporal diversity across timesteps; K/V=source acts
    as a fixed knowledge bank at inference without collapsing the LSTM input.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=1,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Q=target queries into source K/V; residual on target preserves temporal info
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return self.norm(q + out)


# ─── Feature Extractor  Gf(·) ─────────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """
    Three-branch feature extractor (Figure 2 of paper).

    Source branch:        self-attention on source   → z_src_private
    Target branch:        self-attention on target   → z_tgt_private
    Source-Target branch: cross-attention            → z_shared

    Paper Section 3.1: "three branches with SHARED attention weights" — Eq. 2
    uses W^Q, W^K, W^V without domain subscripts, meaning one SelfAttentionBlock
    is used for both source and target branches.  The cross-attention has its own
    weight matrices W^Qs, W^Kt, W^Vt (Eq. 6), kept separate.

    shared_weights=True  (CWPDDA default): faithful to paper — one proj, one self_attn.
    shared_weights=False (MC-CWPDDA):      separate proj_src/proj_tgt/self_attn_src/
                                           self_attn_tgt for staged-curriculum freezing.

    Input projection: (batch, window_size) → (batch, window_size, d_model)
    Pool over sequence: (batch, d_model)
    """

    def __init__(self, window_size: int, d_model: int = 64, dropout: float = 0.1,
                 shared_weights: bool = True):
        super().__init__()
        self._shared = shared_weights

        if shared_weights:
            # Paper-faithful: one projection and one self-attention shared across branches
            self.proj      = nn.Linear(1, d_model)
            self.self_attn = SelfAttentionBlock(d_model, dropout)
        else:
            # MC-CWPDDA: separate per-domain projections and attention for staged freezing
            self.proj_src      = nn.Linear(1, d_model)
            self.proj_tgt      = nn.Linear(1, d_model)
            self.self_attn_src = SelfAttentionBlock(d_model, dropout)
            self.self_attn_tgt = SelfAttentionBlock(d_model, dropout)

        self.cross_attn = CrossAttentionBlock(d_model, dropout)
        self.out_dim    = d_model

    def forward(
        self,
        x_src: torch.Tensor,   # (batch, W)
        x_tgt: torch.Tensor,   # (batch, W)
    ):
        """
        Returns:
            z_shared:      (batch, W, d_model) — full sequence fed to LSTM predictor
            z_src_private: (batch, d_model)    — pooled, used for MMD and discriminator
            z_tgt_private: (batch, d_model)    — pooled, used for MMD and discriminator
        """
        if self._shared:
            hs = self.proj(x_src.unsqueeze(-1))              # (batch, W, d_model)
            ht = self.proj(x_tgt.unsqueeze(-1))
            hs_sa = self.self_attn(hs)                        # source self-attended
            ht_sa = self.self_attn(ht)                        # target self-attended
            z_src_private = hs_sa.mean(dim=1)                 # (batch, d_model)
            z_tgt_private = ht_sa.mean(dim=1)
        else:
            hs = self.proj_src(x_src.unsqueeze(-1))
            ht = self.proj_tgt(x_tgt.unsqueeze(-1))
            hs_sa = self.self_attn_src(hs)
            ht_sa = self.self_attn_tgt(ht)
            z_src_private = hs_sa.mean(dim=1)
            z_tgt_private = ht_sa.mean(dim=1)

        z_shared = self.cross_attn(ht_sa, hs_sa)             # (batch, W, d_model)

        return z_shared, z_src_private, z_tgt_private

    def mmd_loss(
        self,
        z_private: torch.Tensor,  # src or tgt private
        z_shared: torch.Tensor,
    ) -> torch.Tensor:
        """
        Eq. 10: maximise MMD between private and shared features.
        Uses -mmd/(mmd.detach()+1e-8) for numerical stability; naive 1/(mmd+ε)
        explodes at init and swamps the prediction loss.
        """
        # z_shared is now (batch, W, d_model) — pool over sequence before comparing
        if z_shared.dim() == 3:
            z_shared = z_shared.mean(dim=1)  # (batch, d_model)
        diff = z_private.mean(0) - z_shared.mean(0)
        mmd  = (diff ** 2).sum()
        return -mmd / (mmd.detach() + 1e-8)   # ≈ -1, stable; gradient ∝ -1/mmd


# ─── Domain Adversarial Adapter  Gd(·) ───────────────────────────────────────

class DomainAdversarialAdapter(nn.Module):
    """
    Domain discriminator with GRL (Figure 3, Eqs. 11-13).
    Classifies features as source (0) or target (1).
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        """z: (batch, d_model) → domain probability (batch, 1)"""
        return self.net(grad_reverse(z, lam))

    def loss(
        self,
        z_src: torch.Tensor,
        z_tgt: torch.Tensor,
        lam: float = 1.0,
    ) -> torch.Tensor:
        """
        Negative log-likelihood domain loss (Eq. 12).
        Source label = 0, target label = 1.
        """
        p_src = self.forward(z_src, lam)   # (N, 1)
        p_tgt = self.forward(z_tgt, lam)   # (N, 1)

        # Binary cross-entropy from NLL formulation in Eq. 12
        eps = 1e-7
        loss_src = -torch.log(1 - p_src + eps).mean()
        loss_tgt = -torch.log(p_tgt + eps).mean()
        return loss_src + loss_tgt


# ─── Workload Predictor  Gy(·) ────────────────────────────────────────────────

class WorkloadPredictor(nn.Module):
    """
    Multi-layer LSTM + FC head (Figure 4, Eq. 14-15).
    Input: shared features (batch, d_model)
    Output: predicted workload (batch, horizon)
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 40,
        n_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 1,
    ):
        super().__init__()
        # Paper Table 2: "Number of cells for each layer: 40, Number of layers: 2"
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, W, d_model) → (batch, horizon)"""
        # z is already the full W-step sequence from cross-attention; no unsqueeze needed.
        out, _ = self.lstm(z)                # (batch, W, hidden)
        return self.fc(out[:, -1, :])        # (batch, horizon) — use last timestep


# ─── Full CWPDDA model ────────────────────────────────────────────────────────

class CWPDDA(nn.Module):
    """
    Full CWPDDA model.

    Usage:
        model = CWPDDA(window_size=24)
        loss, info = model.compute_loss(x_src, y_src, x_tgt, y_tgt, step, total)
        pred = model.predict(x_tgt)
    """

    def __init__(
        self,
        window_size: int = 24,
        d_model: int = 64,
        lstm_hidden: int = 40,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 1,
        alpha: float = 10.0,     # GRL schedule param (Table 2)
        beta: float = 0.75,      # GRL schedule param (Table 2)
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

        self.extractor  = FeatureExtractor(window_size, d_model, dropout)
        self.adapter    = DomainAdversarialAdapter(d_model, dropout)
        self.predictor  = WorkloadPredictor(d_model, lstm_hidden, lstm_layers,
                                            dropout, horizon)

        # Source reference batch for inference — set via register_source_ref()
        # after training so that cross-attention at test time uses real source data.
        self._src_ref: Optional[torch.Tensor] = None

    def forward(
        self,
        x_src: torch.Tensor,
        x_tgt: torch.Tensor,
    ):
        z_shared, z_src_priv, z_tgt_priv = self.extractor(x_src, x_tgt)
        workload_pred = self.predictor(z_shared)
        return workload_pred, z_shared, z_src_priv, z_tgt_priv

    def compute_loss(
        self,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        x_tgt: torch.Tensor,
        y_tgt: torch.Tensor,
        step: int = 0,
        total_steps: int = 1000,
    ):
        lam = grl_lambda(step, total_steps, self.alpha, self.beta)

        pred, z_shared, z_src_priv, z_tgt_priv = self.forward(x_src, x_tgt)

        # Ly: prediction MSE on target (Eq. 15)
        Ly = F.mse_loss(pred, y_tgt)

        # Lf: feature disentanglement — push BOTH private features away from
        # shared so private ≠ shared (Eq. 10).  Apply to both source and target.
        Lf = (self.extractor.mmd_loss(z_src_priv, z_shared) +
              self.extractor.mmd_loss(z_tgt_priv, z_shared))

        # Ld: domain adversarial loss (Eq. 12, via GRL).
        # Discriminator sees source-private (label=0) vs target-private (label=1).
        # GRL makes the extractor produce domain-invariant private features.
        Ld = self.adapter.loss(z_src_priv, z_tgt_priv, lam)

        lam1, lam2 = 0.01, 0.01  # not stated in paper; tuned to keep aux losses at same scale as Ly
        loss = Ly + lam1 * Lf + lam2 * Ld

        return loss, {
            "Ly": Ly.item(),
            "Lf": Lf.item(),
            "Ld": Ld.item(),
            "lam": lam,
        }

    def register_source_ref(self, x_src_np: np.ndarray, n: int = 0) -> None:
        """
        Store source window bank for nearest-neighbour retrieval at train/inference.
        n=0 uses all windows; set n=8192 on CPU for a speed/quality tradeoff.
        """
        rng = np.random.default_rng(42)
        if n > 0:
            idx = rng.choice(len(x_src_np), size=min(n, len(x_src_np)), replace=False)
            subset = x_src_np[idx].astype(np.float32)          # (n, W)
        else:
            subset = x_src_np.astype(np.float32)                # all windows
        self._src_ref      = torch.from_numpy(subset).float()           # (n, W)
        self._src_ref_mean = torch.from_numpy(
            subset.mean(axis=0, keepdims=True)).float()                 # (1, W)

    def _match_source(self, x_tgt: torch.Tensor, bank_chunk: int = 4096) -> torch.Tensor:
        """Find nearest source window for each target window by L2 distance (chunked for OOM safety)."""
        if self._src_ref is None:
            return x_tgt   # fallback: self-attention
        bank = self._src_ref.to(x_tgt.device)   # (n, W)
        n = bank.shape[0]
        best_dist = torch.full((x_tgt.shape[0],), float("inf"), device=x_tgt.device)
        best_idx  = torch.zeros(x_tgt.shape[0], dtype=torch.long, device=x_tgt.device)
        for start in range(0, n, bank_chunk):
            chunk = bank[start : start + bank_chunk]          # (c, W)
            dist  = ((x_tgt.unsqueeze(1) - chunk.unsqueeze(0)) ** 2).sum(-1)  # (B, c)
            chunk_min, chunk_argmin = dist.min(dim=1)         # (B,)
            mask = chunk_min < best_dist
            best_dist[mask] = chunk_min[mask]
            best_idx[mask]  = start + chunk_argmin[mask]
        return bank[best_idx]                                  # (B, W)

    @torch.no_grad()
    def predict(self, x_tgt: torch.Tensor, x_src: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict workload for target windows using nearest-source cross-attention."""
        self.eval()
        if x_src is None:
            x_src = self._match_source(x_tgt)
        z_shared, _, _ = self.extractor(x_src, x_tgt)
        return self.predictor(z_shared)

    @torch.no_grad()
    def predict_numpy_batched(
        self,
        x_np: np.ndarray,
        device: str,
        batch_size: int = 2048,
    ) -> np.ndarray:
        """
        Batched inference using nearest-neighbour source retrieval.
        Each target window gets its closest source window as K/V in cross-attention,
        consistent with how source windows are selected during training.
        """
        self.eval()
        n = len(x_np)
        if n == 0:
            h = int(self.predictor.fc.out_features)
            return np.empty((0, h), dtype=np.float32)

        parts: list[np.ndarray] = []
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(x_np[i : i + batch_size]).float().to(device)
            xs = self._match_source(xb)   # nearest-neighbour retrieval per window
            z, _, _ = self.extractor(xs, xb)
            parts.append(self.predictor(z).cpu().numpy())
        return np.concatenate(parts, axis=0)