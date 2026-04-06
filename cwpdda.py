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
    Single-head scaled dot-product self-attention (paper Eqs. 2-5).
    Input:  (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        scale = math.sqrt(self.d_model)
        # (batch, seq, seq)
        attn = torch.softmax(Q @ K.transpose(-2, -1) / scale, dim=-1)
        attn = self.drop(attn)
        out = attn @ V                      # (batch, seq, d_model)
        return self.norm(x + out)           # residual


# ─── Cross-Attention block ────────────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention: Q from source, K/V from target (paper Eqs. 6-9).
    Input:  xs (batch, seq_s, d_model),  xt (batch, seq_t, d_model)
    Output: (batch, seq_s, d_model)  — shared features
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.Wqs = nn.Linear(d_model, d_model, bias=False)
        self.Wkt = nn.Linear(d_model, d_model, bias=False)
        self.Wvt = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, xs: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        Qs = self.Wqs(xs)
        Kt = self.Wkt(xt)
        Vt = self.Wvt(xt)
        scale = math.sqrt(self.d_model)
        attn = torch.softmax(Qs @ Kt.transpose(-2, -1) / scale, dim=-1)
        attn = self.drop(attn)
        out = attn @ Vt                     # (batch, seq_s, d_model)
        return self.norm(xs + out)


# ─── Feature Extractor  Gf(·) ─────────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """
    Three-branch feature extractor (Figure 2 of paper).

    Source branch:        self-attention on source   → z_src_private
    Target branch:        self-attention on target   → z_tgt_private
    Source-Target branch: cross-attention            → z_shared

    Input projection: (batch, window_size) → (batch, window_size, d_model)
    Pool over sequence: (batch, d_model)
    """

    def __init__(self, window_size: int, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.proj_src = nn.Linear(1, d_model)
        self.proj_tgt = nn.Linear(1, d_model)

        self.self_attn_src = SelfAttentionBlock(d_model, dropout)
        self.self_attn_tgt = SelfAttentionBlock(d_model, dropout)
        self.cross_attn    = CrossAttentionBlock(d_model, dropout)

        self.out_dim = d_model

    def forward(
        self,
        x_src: torch.Tensor,   # (batch, W)
        x_tgt: torch.Tensor,   # (batch, W)
    ):
        """
        Returns:
            z_shared: (batch, d_model)  — used by predictor and discriminator
            z_src_private: (batch, d_model)
            z_tgt_private: (batch, d_model)
        """
        # (batch, W, 1) → (batch, W, d_model)
        hs = self.proj_src(x_src.unsqueeze(-1))
        ht = self.proj_tgt(x_tgt.unsqueeze(-1))

        # Private features
        z_src_private = self.self_attn_src(hs).mean(dim=1)   # (batch, d_model)
        z_tgt_private = self.self_attn_tgt(ht).mean(dim=1)

        # Shared features via cross-attention
        z_shared = self.cross_attn(hs, ht).mean(dim=1)       # (batch, d_model)

        return z_shared, z_src_private, z_tgt_private

    def mmd_loss(
        self,
        z_private: torch.Tensor,  # src or tgt private
        z_shared: torch.Tensor,
    ) -> torch.Tensor:
        """
        Feature extractor loss (Eq. 10): 1 / MMD(private, shared).
        We want to MAXIMISE MMD, so the loss is -MMD (minimised by optimiser).
        Guards against zero-division.
        """
        diff = z_private.mean(0) - z_shared.mean(0)
        mmd = (diff ** 2).sum()
        return -mmd / (mmd.detach() + 1e-8)     # normalised, maximised


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
        """z: (batch, d_model) → (batch, horizon)"""
        # Treat the d_model vector as a length-1 sequence for the LSTM
        out, _ = self.lstm(z.unsqueeze(1))   # (batch, 1, hidden)
        return self.fc(out[:, -1, :])         # (batch, horizon)


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

    def forward(
        self,
        x_src: torch.Tensor,
        x_tgt: torch.Tensor,
        lam: float = 1.0,
    ):
        z_shared, z_src_priv, z_tgt_priv = self.extractor(x_src, x_tgt)
        domain_pred = self.adapter(z_shared, lam)
        workload_pred = self.predictor(z_shared)
        return workload_pred, domain_pred, z_shared, z_src_priv, z_tgt_priv

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

        pred, _, z_shared, z_src_priv, z_tgt_priv = self.forward(
            x_src, x_tgt, lam
        )

        # Ly: prediction MSE on target (Eq. 15)
        Ly = F.mse_loss(pred, y_tgt)

        # Lf: feature extractor loss — src private vs shared (Eq. 10)
        Lf = self.extractor.mmd_loss(z_src_priv, z_shared)

        # Ld: domain adversarial loss (Eq. 12, via GRL)
        Ld = self.adapter.loss(z_shared, z_shared, lam)  # symmetrical

        loss = Ly + Lf + Ld

        return loss, {
            "Ly": Ly.item(),
            "Lf": Lf.item(),
            "Ld": Ld.item(),
            "lam": lam,
        }

    @torch.no_grad()
    def predict(self, x_tgt: torch.Tensor, x_src: torch.Tensor = None) -> torch.Tensor:
        """
        Predict workload for target windows.
        If x_src not provided, uses x_tgt as a dummy source (inference mode).
        """
        self.eval()
        if x_src is None:
            x_src = x_tgt
        z_shared, _, _ = self.extractor(x_src, x_tgt)
        return self.predictor(z_shared)
