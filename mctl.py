"""
models/mctl.py
==============
Mixed Contrastive Transfer Learning (MCTL) for few-shot workload prediction.
Implements the paper: "Mixed contrastive transfer learning for few-shot
workload prediction in the cloud" (Zuo et al., Computing 2025).

Two-stage training:

Stage 1 — Data Mixup + Source Encoder pretraining (Section 3.3)
  - Mixup augmentation on source data
  - Train source TCN encoder with prediction loss

Stage 2 — Contrastive Representation Transfer (Section 3.4)
  - Freeze source encoder
  - For each target window x̃_t (mixed), positive samples = the two
    original subsamples x^(1), x^(2); negatives = other series subsamples
  - Minimise KL divergence between source and target representation
    relationship distributions  (Eq. 8 in paper)
  - Then fine-tune target regression head (Eq. 9)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tcn import TCNEncoder, TCNPredictor


# ─── Mixup ────────────────────────────────────────────────────────────────────

def mixup_batch(
    x: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply mixup to a batch of windows (paper Section 3.3, Eq. 1).

    λ ~ Beta(α, α)
    x̃ = λ * x_i + (1 - λ) * x_j

    Returns:
        x_mixed: (B, W)  — augmented samples
        x1:      (B, W)  — first original samples  (positive)
        x2:      (B, W)  — second original samples (positive)
        lam:     scalar  — mixing parameter λ
    """
    B = x.size(0)
    lam = float(np.random.beta(alpha, alpha))

    idx = torch.randperm(B, device=x.device)
    x1, x2 = x, x[idx]
    x_mixed = lam * x1 + (1 - lam) * x2
    return x_mixed, x1, x2, lam


# ─── Representation relationship distribution (paper Eq. 6 / 7) ───────────────

def _papn(
    z_mixed: torch.Tensor,   # (B, H)
    z_pos1:  torch.Tensor,   # (B, H)
    z_pos2:  torch.Tensor,   # (B, H)
    z_neg:   torch.Tensor,   # (B, K, H)
    lam: float,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Compute PAPN — the representation relationship distribution.
    Paper Eq. 6:

      PAPN(Z) = λ * sim(z̃, z^(1)) / Σ_k [sim(z̃, z^(1)_k) + sim(z̃, z^(2)_k)]
              + (1-λ) * sim(z̃, z^(2)) / Σ_k [...]

    where sim uses the Student-t kernel: (1 + cos_sim / τ)^{-μ}, μ = (τ+1)/2

    Returns: (B,) probability values
    """
    mu = (tau + 1) / 2

    def _student(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """(B,) or (B,K) student-t similarity."""
        # a: (B,H), b: (B,H) or (B,K,H)
        a_n = F.normalize(a, dim=-1)
        b_n = F.normalize(b, dim=-1)
        if b_n.dim() == 3:
            # (B,K)
            cos = (a_n.unsqueeze(1) * b_n).sum(-1)   # (B,K)
        else:
            cos = (a_n * b_n).sum(-1)                 # (B,)
        return (1.0 + cos / tau) ** (-mu)             # (B,) or (B,K)

    sim1 = _student(z_mixed, z_pos1)   # (B,)
    sim2 = _student(z_mixed, z_pos2)   # (B,)

    # neg: (B, K, H)
    sim_neg1 = _student(z_mixed, z_neg)                               # (B,K)
    sim_neg2 = _student(z_mixed, z_neg)                               # (B,K) (same for both)
    denom = (sim_neg1 + sim_neg2).mean(dim=1) + 1e-8                  # (B,)

    papn = lam * sim1 / denom + (1 - lam) * sim2 / denom              # (B,)
    return papn.clamp(1e-7, 1.0 - 1e-7)


def contrastive_kl_loss(
    z_src_mixed: torch.Tensor,  # (B, H)
    z_src_pos1:  torch.Tensor,  # (B, H)
    z_src_pos2:  torch.Tensor,  # (B, H)
    z_src_neg:   torch.Tensor,  # (B, K, H)
    z_tgt_mixed: torch.Tensor,
    z_tgt_pos1:  torch.Tensor,
    z_tgt_pos2:  torch.Tensor,
    z_tgt_neg:   torch.Tensor,
    lam: float,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    KL divergence between source and target representation distributions.
    Paper Eq. 8:  min Σ KL( [P_src, 1-P_src] || [P_tgt, 1-P_tgt] )
    """
    p_src = _papn(z_src_mixed, z_src_pos1, z_src_pos2, z_src_neg, lam, tau)
    p_tgt = _papn(z_tgt_mixed, z_tgt_pos1, z_tgt_pos2, z_tgt_neg, lam, tau)

    # Bernoulli KL: p_src * log(p_src/p_tgt) + (1-p_src)*log((1-p_src)/(1-p_tgt))
    kl = (
        p_src * (p_src / p_tgt).log()
        + (1 - p_src) * ((1 - p_src) / (1 - p_tgt)).log()
    )
    return kl.mean()


# ─── MCTL model ───────────────────────────────────────────────────────────────

class MCTL(nn.Module):
    """
    Mixed Contrastive Transfer Learning model.

    Wraps:
      - source_encoder: TCNEncoder (trained on source, then frozen)
      - target_encoder: TCNEncoder (trained via KL alignment)
      - regression_head: Linear (trained on target with MSE)
    """

    def __init__(
        self,
        window_size: int = 24,
        hidden_dim: int = 128,
        n_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
        horizon: int = 1,
        alpha_mixup: float = 1.0,
        tau: float = 1.0,
        n_negatives: int = 8,
    ):
        super().__init__()
        self.window_size  = window_size
        self.hidden_dim   = hidden_dim
        self.alpha_mixup  = alpha_mixup
        self.tau          = tau
        self.n_negatives  = n_negatives

        self.source_encoder = TCNEncoder(window_size, hidden_dim, n_layers, kernel_size, dropout)
        self.target_encoder = TCNEncoder(window_size, hidden_dim, n_layers, kernel_size, dropout)
        self.regression_head = nn.Linear(hidden_dim, horizon)

    # ── Stage 1: pretrain source encoder ──────────────────────────────────────

    def source_forward(self, x_src: torch.Tensor) -> torch.Tensor:
        """x_src: (B, W) → (B, H)"""
        return self.source_encoder(x_src)

    # ── Stage 2: contrastive transfer ────────────────────────────────────────

    def compute_transfer_loss(
        self,
        x_src: torch.Tensor,    # (B, W)
        x_tgt: torch.Tensor,    # (B, W)
    ) -> torch.Tensor:
        """
        Full MCTL transfer loss (paper Eq. 8).

        1. Mixup both source and target batches
        2. Encode with (frozen) source encoder and (trainable) target encoder
        3. KL divergence between representation distributions
        """
        # --- Mixup ---
        xm_s, x1_s, x2_s, lam_s = mixup_batch(x_src, self.alpha_mixup)
        xm_t, x1_t, x2_t, lam_t = mixup_batch(x_tgt, self.alpha_mixup)
        lam = (lam_s + lam_t) / 2  # average λ across domains

        # --- Negative samples ---
        B = x_src.size(0)
        K = min(self.n_negatives, B - 1)
        neg_idx = torch.stack(
            [torch.randperm(B, device=x_src.device)[:K] for _ in range(B)]
        )  # (B, K)
        x_neg_s = x_src[neg_idx]    # (B, K, W)
        x_neg_t = x_tgt[neg_idx]    # (B, K, W)

        # --- Encode (source is frozen in stage 2) ---
        with torch.no_grad():
            zm_s  = self.source_encoder(xm_s)                           # (B,H)
            z1_s  = self.source_encoder(x1_s)                           # (B,H)
            z2_s  = self.source_encoder(x2_s)                           # (B,H)
            B_, K_ = x_neg_s.shape[:2]
            zneg_s = self.source_encoder(
                x_neg_s.view(B_ * K_, -1)
            ).view(B_, K_, -1)                                           # (B,K,H)

        zm_t  = self.target_encoder(xm_t)
        z1_t  = self.target_encoder(x1_t)
        z2_t  = self.target_encoder(x2_t)
        zneg_t = self.target_encoder(
            x_neg_t.view(B_ * K_, -1)
        ).view(B_, K_, -1)

        return contrastive_kl_loss(
            zm_s, z1_s, z2_s, zneg_s,
            zm_t, z1_t, z2_t, zneg_t,
            lam, self.tau,
        )

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W) → (B, horizon)"""
        z = self.target_encoder(x)
        return self.regression_head(z)

    def freeze_source(self):
        for p in self.source_encoder.parameters():
            p.requires_grad = False

    def unfreeze_source(self):
        for p in self.source_encoder.parameters():
            p.requires_grad = True
