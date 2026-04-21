"""
mc_cwpdda.py
============
MC-CWPDDA — Mixed Contrastive Container Workload Prediction with Deep Domain Adaptation.

Novel combination of:
  - CWPDDA (Wang et al., Euro-Par 2025): three-branch attention + GRL adversarial adapter
  - MCTL   (Zuo et al., Computing 2025): PAPN contrastive loss + KL distribution alignment

Architecture (four modules):
    Gf — Feature Extractor:     three-branch attention (source self-attn,
                                  target self-attn, cross-attn) → z_shared,
                                  z_src_priv, z_tgt_priv
    Gc — Contrastive Head:      two-layer MLP from z_shared → unit sphere
    Gd — Domain Adv. Adapter:   GRL binary discriminator on private features
    Gy — Workload Predictor:    two-layer LSTM on z_shared → y_hat

Full training objective (Stage 3):
    L = Ly + λ1·Lf + λ2·Ld + λ3·Lc + λ4·Lkl
    λ1=0.1, λ2=0.1, λ3=0.1, λ4=0.05   (Table 1 of MC-CWPDDA paper)

Three-stage curriculum:
    Stage 1 — Source pre-training:  proj_src + self_attn_src + temp head,
                                     MSE on Google data, E1 epochs
    Stage 2 — Contrastive alignment: freeze source branch; train target branch,
                                     cross-attn, Gc on Lc + λ4·Lkl, E2 epochs
    Stage 3 — Joint fine-tuning:    unfreeze all; full loss; early-stop on val MSE
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse CWPDDA building-blocks directly — no code duplication
from cwpdda import (
    grl_lambda,
    SelfAttentionBlock,
    CrossAttentionBlock,
    FeatureExtractor,
    DomainAdversarialAdapter,
    WorkloadPredictor,
)


# ─── Contrastive Projection Head  Gc(·) ──────────────────────────────────────

class ContrastiveHead(nn.Module):
    """
    Gc: two-layer MLP → L2-normalised projection (unit sphere).
    Input:  (batch, d_model)
    Output: (batch, proj_dim)  — unit-norm vectors for cosine similarity
    """

    def __init__(self, d_model: int, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


# ─── PAPN contrastive loss helpers ────────────────────────────────────────────

def _student_sim(
    a: torch.Tensor,   # (B, D) or (B, D)
    b: torch.Tensor,   # (B, D) or (B, K, D)
    tau: float = 1.0,
) -> torch.Tensor:
    """Student-t kernel f(a,b) = (1 + cos(a,b)/τ)^(-μ),  μ=(τ+1)/2."""
    mu = (tau + 1) / 2
    if b.dim() == 3:                          # negative bank (B, K, D)
        cos = (a.unsqueeze(1) * b).sum(-1)    # (B, K)
    else:
        cos = (a * b).sum(-1)                 # (B,)
    return (1.0 + cos / tau).clamp(min=1e-6) ** (-mu)


def papn_contrastive_loss(
    h_anchor:  torch.Tensor,   # (B, proj_dim) — Gc(z_mix)
    h_pos_src: torch.Tensor,   # (B, proj_dim) — Gc(z_shared)  [source positive]
    h_pos_tgt: torch.Tensor,   # (B, proj_dim) — Gc(z_tgt_only) [target positive]
    lam: float,                 # mixup λ  (float in [0,1])
    n_neg: int = 8,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    PAPN loss (Eq. 3 of MC-CWPDDA paper; derived from MCTL Eq. 6).

    Attracts h_anchor to h_pos_src (weight λ) and h_pos_tgt (weight 1-λ),
    repels from K random in-batch negatives drawn from h_pos_src.
    """
    B = h_anchor.size(0)
    K = min(n_neg, B - 1)

    # Draw K negative indices per anchor
    neg_idx = torch.stack([
        torch.randperm(B, device=h_anchor.device)[:K] for _ in range(B)
    ])                                            # (B, K)
    h_neg = h_pos_src[neg_idx]                   # (B, K, proj_dim)

    f1 = _student_sim(h_anchor, h_pos_src, tau)  # (B,)
    f2 = _student_sim(h_anchor, h_pos_tgt, tau)  # (B,)
    fn = _student_sim(h_anchor, h_neg,     tau).mean(dim=1)  # (B,)

    numerator   = f1.pow(lam) * f2.pow(1.0 - lam) + 1e-9
    denominator = numerator + fn.clamp(min=1e-9)
    return -torch.log(numerator / denominator).mean()


def kl_alignment_loss(
    h_src: torch.Tensor,   # (B, proj_dim) Gc-projected source features
    h_tgt: torch.Tensor,   # (B, proj_dim) Gc-projected target features
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    KL divergence between batch-level soft feature distributions.
    Encourages target features to match source distribution in contrastive space.
    q(z) = softmax over batch dimension (each feature dim is a distribution over B).
    """
    q_src = F.softmax(h_src / temperature, dim=0)  # (B, proj_dim)
    q_tgt = F.softmax(h_tgt / temperature, dim=0)
    return F.kl_div(q_tgt.log(), q_src, reduction="batchmean")


# ─── Full MC-CWPDDA model ─────────────────────────────────────────────────────

class MCCWPDDA(nn.Module):
    """
    MC-CWPDDA: Mixed Contrastive CWPDDA.

    Usage:
        model = MCCWPDDA(window_size=24)
        # Stage 3 (joint):
        loss, info = model.compute_loss(x_src, y_src, x_tgt, y_tgt, step, total)
        # Inference:
        pred = model.predict(x_tgt)
    """

    def __init__(
        self,
        window_size:  int   = 24,
        d_model:      int   = 64,
        lstm_hidden:  int   = 40,
        lstm_layers:  int   = 2,
        dropout:      float = 0.1,
        horizon:      int   = 1,
        proj_dim:     int   = 64,    # Gc output dimension
        alpha:        float = 10.0,  # GRL schedule (Table 2 of CWPDDA)
        beta:         float = 0.75,
        lam1:         float = 0.1,   # Lf weight (disentanglement)
        lam2:         float = 0.1,   # Ld weight (adversarial)
        lam3:         float = 0.1,   # Lc weight (PAPN contrastive)
        lam4:         float = 0.05,  # Lkl weight (KL alignment)
        alpha_mixup:  float = 1.0,   # Beta param for mixup λ
        n_neg:        int   = 8,     # PAPN negatives per anchor
        tau:          float = 1.0,   # Student-t temperature
    ):
        super().__init__()
        self.alpha       = alpha
        self.beta        = beta
        self.lam1        = lam1
        self.lam2        = lam2
        self.lam3        = lam3
        self.lam4        = lam4
        self.alpha_mixup = alpha_mixup
        self.n_neg       = n_neg
        self.tau         = tau

        self.extractor        = FeatureExtractor(window_size, d_model, dropout)
        self.contrastive_head = ContrastiveHead(d_model, proj_dim)
        self.adapter          = DomainAdversarialAdapter(d_model, dropout)
        self.predictor        = WorkloadPredictor(d_model, lstm_hidden,
                                                   lstm_layers, dropout, horizon)

    # ── Freeze / unfreeze helpers for staged training ─────────────────────────

    def freeze_source_branch(self):
        """Freeze source-only parameters (proj_src, self_attn_src)."""
        for p in self.extractor.proj_src.parameters():
            p.requires_grad = False
        for p in self.extractor.self_attn_src.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x_src: torch.Tensor,
        x_tgt: torch.Tensor,
        lam_grl: float = 1.0,
    ) -> Tuple[torch.Tensor, ...]:
        z_shared, z_src_priv, z_tgt_priv = self.extractor(x_src, x_tgt)
        domain_pred  = self.adapter(z_shared, lam_grl)
        workload_pred = self.predictor(z_shared)
        return workload_pred, domain_pred, z_shared, z_src_priv, z_tgt_priv

    # ── Stage 3 full loss ─────────────────────────────────────────────────────

    def compute_loss(
        self,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        x_tgt: torch.Tensor,
        y_tgt: torch.Tensor,
        step:        int = 0,
        total_steps: int = 1000,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Full MC-CWPDDA loss (Stage 3 joint training).

        L = Ly + λ1·Lf + λ2·Ld + λ3·Lc + λ4·Lkl
        """
        lam_grl = grl_lambda(step, total_steps, self.alpha, self.beta)

        # ── Main forward ──────────────────────────────────────────────────────
        pred, _, z_shared, z_src_priv, z_tgt_priv = self.forward(
            x_src, x_tgt, lam_grl
        )

        # Ly: target prediction MSE
        Ly = F.mse_loss(pred, y_tgt)

        # Lf: MMD disentanglement — push private features away from shared
        Lf = (self.extractor.mmd_loss(z_src_priv, z_shared) +
              self.extractor.mmd_loss(z_tgt_priv, z_shared))

        # Ld: domain adversarial via GRL on private features
        Ld = self.adapter.loss(z_src_priv, z_tgt_priv, lam_grl)

        # ── Contrastive forward ───────────────────────────────────────────────
        # Cross-domain Mixup anchor
        lam_mix = float(np.random.beta(self.alpha_mixup, self.alpha_mixup))
        x_mix   = lam_mix * x_src + (1.0 - lam_mix) * x_tgt

        # Source positive: Gc(z_shared) from main forward
        h_pos_src = self.contrastive_head(z_shared)          # (B, proj_dim)

        # Anchor: Gc of z_shared from mixup input
        with torch.no_grad() if False else torch.enable_grad():
            z_mix, _, _ = self.extractor(x_mix, x_tgt)
        h_anchor = self.contrastive_head(z_mix)               # (B, proj_dim)

        # Target positive: Gc of z_shared when source=target
        z_tgt_only, _, _ = self.extractor(x_tgt, x_tgt)
        h_pos_tgt = self.contrastive_head(z_tgt_only)         # (B, proj_dim)

        Lc  = papn_contrastive_loss(
            h_anchor, h_pos_src, h_pos_tgt, lam_mix, self.n_neg, self.tau
        )
        Lkl = kl_alignment_loss(h_pos_src, h_pos_tgt)

        loss = Ly + self.lam1 * Lf + self.lam2 * Ld + self.lam3 * Lc + self.lam4 * Lkl

        return loss, {
            "Ly":      Ly.item(),
            "Lf":      Lf.item(),
            "Ld":      Ld.item(),
            "Lc":      Lc.item(),
            "Lkl":     Lkl.item(),
            "lam_grl": lam_grl,
            "lam_mix": lam_mix,
        }

    # ── Stage 2 contrastive-only loss ─────────────────────────────────────────

    def contrastive_alignment_loss(
        self,
        x_src: torch.Tensor,
        x_tgt: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Stage 2 loss: Lc + λ4·Lkl only (source branch frozen).
        Does NOT compute Ly, Lf, Ld to keep Stage 2 focused on alignment.
        """
        lam_mix = float(np.random.beta(self.alpha_mixup, self.alpha_mixup))
        x_mix   = lam_mix * x_src + (1.0 - lam_mix) * x_tgt

        z_shared, _, _    = self.extractor(x_src,  x_tgt)
        z_mix, _, _       = self.extractor(x_mix,  x_tgt)
        z_tgt_only, _, _  = self.extractor(x_tgt,  x_tgt)

        h_pos_src = self.contrastive_head(z_shared)
        h_anchor  = self.contrastive_head(z_mix)
        h_pos_tgt = self.contrastive_head(z_tgt_only)

        Lc  = papn_contrastive_loss(
            h_anchor, h_pos_src, h_pos_tgt, lam_mix, self.n_neg, self.tau
        )
        Lkl = kl_alignment_loss(h_pos_src, h_pos_tgt)

        loss = Lc + self.lam4 * Lkl
        return loss, {"Lc": Lc.item(), "Lkl": Lkl.item(), "lam_mix": lam_mix}

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        x_tgt: torch.Tensor,
        x_src: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict workload.  Uses x_tgt as dummy source if x_src not provided."""
        self.eval()
        if x_src is None:
            x_src = x_tgt
        z_shared, _, _ = self.extractor(x_src, x_tgt)
        return self.predictor(z_shared)

    @torch.no_grad()
    def predict_numpy_batched(
        self,
        x_np:       np.ndarray,
        device:     str,
        batch_size: int = 2048,
    ) -> np.ndarray:
        """Chunked inference — avoids CUDA SDPA OOM on large test sets."""
        self.eval()
        n = len(x_np)
        if n == 0:
            h = int(self.predictor.fc.out_features)
            return np.empty((0, h), dtype=np.float32)
        parts: list[np.ndarray] = []
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(x_np[i : i + batch_size]).float().to(device)
            z, _, _ = self.extractor(xb, xb)
            parts.append(self.predictor(z).cpu().numpy())
        return np.concatenate(parts, axis=0)
