"""
models/mctl.py
==============
Mixed Contrastive Transfer Learning (MCTL) — Zuo et al., Computing 2025.
Google (long series, source) → Alibaba (short series, target).

Two-stage training:
  Stage 1: Train source TCN encoder with prediction loss on Google data.
  Stage 2: Freeze source encoder; align target encoder via contrastive KL
           loss; fine-tune regression head on target data.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ─── TCN encoder ─────────────────────────────────────────────────────────────

class _CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      padding=self.pad, dilation=dilation)
        )
        self.net = nn.Sequential(self.conv, nn.ReLU(), nn.Dropout(dropout))
        self.ds  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]
        return F.relu(out + (x if self.ds is None else self.ds(x)))


class TCNEncoder(nn.Module):
    """TCN encoder used in MCTL (Section 3.3). Input: (B,W) → Output: (B,H)"""

    def __init__(self, window_size=24, hidden_dim=128, n_layers=3,
                 kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch  = 1 if i == 0 else hidden_dim
            layers.append(_CausalConv1d(in_ch, hidden_dim, kernel_size, 2**i, dropout))
        self.tcn     = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, x):
        return self.tcn(x.unsqueeze(1)).mean(dim=2)


# ─── Mixup + contrastive KL loss ─────────────────────────────────────────────

def mixup(x, alpha=1.0):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], x, x[idx], lam


def _papn(zm, zp1, zp2, zneg, lam, tau=1.0):
    """
    Compute positive probability for mixed anchor zm via softmax over
    [positive, neg_1, ..., neg_K] — InfoNCE-style.

    Why not the original ratio formula:
      The original used student-t kernel (1+cos/τ)^(-μ) which maps cos≈0
      (typical for random 128-dim vectors) to ≈1.0 for ALL pairs.
      Both s_pos and s_neg collapse to 1 → ratio = 1 → clamped to 1-1e-7
      for BOTH source and target → KL(p||p) = 0 → no gradient ever.

    Softmax fix: with K=8 negatives, random init gives p ≈ 1/9 ≈ 0.11.
    After source encoder pretraining, ps rises toward 1 (positives closer
    than negatives in representation space).  Target pt starts at 0.11 →
    KL(ps || pt) >> 0 → gradient pushes target encoder toward source structure.
    """
    def cos_sim(a, b):
        an = F.normalize(a, dim=-1)
        bn = F.normalize(b, dim=-1)
        if bn.dim() == 3:
            return (an.unsqueeze(1) * bn).sum(-1)   # (B, K)
        return (an * bn).sum(-1)                     # (B,)

    s1 = cos_sim(zm, zp1).unsqueeze(1) / tau   # (B, 1)
    s2 = cos_sim(zm, zp2).unsqueeze(1) / tau   # (B, 1)
    sn = cos_sim(zm, zneg) / tau                # (B, K)

    # Probability that anchor is closer to positive than to any negative
    p1 = F.softmax(torch.cat([s1, sn], dim=1), dim=1)[:, 0]   # (B,)
    p2 = F.softmax(torch.cat([s2, sn], dim=1), dim=1)[:, 0]   # (B,)

    return (lam * p1 + (1 - lam) * p2).clamp(1e-7, 1 - 1e-7)


def contrastive_kl_loss(xm_s, x1_s, x2_s, xn_s,
                         xm_t, x1_t, x2_t, xn_t,
                         enc_src, enc_tgt, lam, tau=1.0):
    B, K, W = xn_s.shape

    with torch.no_grad():
        zm_s = enc_src(xm_s)
        z1_s = enc_src(x1_s)
        z2_s = enc_src(x2_s)
        zn_s = enc_src(xn_s.view(B * K, W)).view(B, K, -1)

    zm_t = enc_tgt(xm_t)
    z1_t = enc_tgt(x1_t)
    z2_t = enc_tgt(x2_t)
    zn_t = enc_tgt(xn_t.view(B * K, W)).view(B, K, -1)

    ps = _papn(zm_s, z1_s, z2_s, zn_s, lam, tau)
    pt = _papn(zm_t, z1_t, z2_t, zn_t, lam, tau)

    kl = ps * (ps / pt).log() + (1 - ps) * ((1 - ps) / (1 - pt)).log()
    return kl.mean()


# ─── Full MCTL model ─────────────────────────────────────────────────────────

class MCTL(nn.Module):
    """
    MCTL full model.
    source_encoder: pretrained on Google, then frozen.
    target_encoder: aligned to source via KL, then fine-tuned.
    regression_head: MSE on target labels.
    """

    def __init__(self, window_size=24, hidden_dim=128, n_layers=3,
                 kernel_size=3, dropout=0.2, horizon=1,
                 alpha_mixup=1.0, tau=0.1, n_neg=32):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.alpha_mixup = alpha_mixup
        self.tau         = tau
        self.n_neg       = n_neg

        self.source_encoder  = TCNEncoder(window_size, hidden_dim, n_layers, kernel_size, dropout)
        self.target_encoder  = TCNEncoder(window_size, hidden_dim, n_layers, kernel_size, dropout)
        self.regression_head = nn.Linear(hidden_dim, horizon)

    def freeze_source(self):
        for p in self.source_encoder.parameters():
            p.requires_grad = False

    def unfreeze_source(self):
        for p in self.source_encoder.parameters():
            p.requires_grad = True

    def transfer_loss(self, x_src, x_tgt):
        B = x_src.size(0)
        K = min(self.n_neg, B - 1)

        xm_s, x1_s, x2_s, lam_s = mixup(x_src, self.alpha_mixup)
        xm_t, x1_t, x2_t, lam_t = mixup(x_tgt, self.alpha_mixup)
        lam = (lam_s + lam_t) / 2

        neg_idx = torch.stack([torch.randperm(B, device=x_src.device)[:K]
                                for _ in range(B)])
        xn_s = x_src[neg_idx]
        xn_t = x_tgt[neg_idx]

        return contrastive_kl_loss(
            xm_s, x1_s, x2_s, xn_s,
            xm_t, x1_t, x2_t, xn_t,
            self.source_encoder, self.target_encoder, lam, self.tau,
        )

    def predict(self, x):
        return self.regression_head(self.target_encoder(x))
