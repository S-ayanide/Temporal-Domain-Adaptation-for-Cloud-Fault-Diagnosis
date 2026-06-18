"""
deepjdot/model.py
=================
DeepJDOT adapted for time-series workload prediction (regression).

Original paper: "DeepJDOT: Deep Joint Distribution Optimal Transport for
Unsupervised Domain Adaptation" — Damodaran et al., ECCV 2018.

The paper solves unsupervised domain adaptation for image classification.
This implementation adapts it for cloud workload prediction (Google → Alibaba):

  Image CNN encoder   → LSTM time-series encoder
  Softmax classifier  → MLP regression head
  Cross-entropy loss  → MSE regression loss

Core idea (unchanged from paper):
  Find an optimal transport coupling γ between source and target samples in
  the joint (feature, label) space, then use γ as importance weights to train
  the shared encoder+predictor to simultaneously:
    (a) Align source and target feature distributions (feature term)
    (b) Propagate source labels to paired target samples (label term)

The JDOT cost per pair (i=source, j=target):
  C_ij = α · ‖g(x_i^s) − g(x_j^t)‖²   ← feature alignment
        + λ_t · L(y_i^s, f(g(x_j^t)))  ← label consistency (MSE for regression)

Full loss (Eq. 6 of paper, adapted):
  L_total = (1/m) Σ_i MSE(y_i^s, f(g(x_i^s)))         ← source supervised loss
          + Σ_{i,j} γ̂_ij · α · ‖g(x_i^s) − g(x_j^t)‖²   ← feature alignment
          + Σ_{i,j} γ̂_ij · λ_t · MSE(y_i^s, f(g(x_j^t))) ← label propagation
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Encoder g: maps time-series window → latent embedding ────────────────────

class TSEncoder(nn.Module):
    """
    LSTM-based time-series encoder — replaces the CNN image encoder from the
    original DeepJDOT paper.

    Architecture:
      - MaxAbs per-window normalization (same as N-BEATS): divide each input
        window by its own absolute max before the LSTM. This makes the encoder
        scale-invariant: Google windows [0.01-0.05] and Alibaba windows [0.09-0.30]
        both become [0,1] before entering the LSTM, removing the amplitude mismatch
        that was causing the LSTM to learn Google-specific scale features.
      - 2-layer LSTM, hidden_dim units
      - Take the final hidden state h_T as the sequence representation
      - Project to d_embed and L2-normalize onto the unit hypersphere
    """

    def __init__(self, window_size: int = 24, hidden_dim: int = 128,
                 n_layers: int = 2, d_embed: int = 128, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim, d_embed)
        self.d_embed = d_embed

    @staticmethod
    def maxabs_scale(x: torch.Tensor):
        """Normalize each window by its own absolute maximum (per N-BEATS paper)."""
        scale = x.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        return x / scale, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W) raw → z: (B, d_embed), unit-norm"""
        x_norm, _ = self.maxabs_scale(x)              # scale-invariant input
        h, _ = self.lstm(x_norm.unsqueeze(-1))         # (B, W, hidden_dim)
        z = self.proj(h[:, -1, :])                     # (B, d_embed)
        return F.normalize(z, p=2, dim=-1)             # unit hypersphere


# ─── Predictor f: maps embedding → workload forecast ──────────────────────────

class Predictor(nn.Module):
    """
    MLP regression head — replaces the softmax classifier from the paper.
    Outputs one step ahead CPU utilisation prediction in [0,1] normalised space.
    """

    def __init__(self, d_embed: int = 64, horizon: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_embed, d_embed // 2),
            nn.ReLU(),
            nn.Linear(d_embed // 2, horizon),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_embed) → y_hat: (B, horizon)"""
        return self.net(z)


# ─── DeepJDOT model ───────────────────────────────────────────────────────────

class DeepJDOT(nn.Module):
    """
    DeepJDOT for unsupervised domain adaptation in workload prediction.

    Shared encoder g + predictor f applied to both source (Google) and
    target (Alibaba) domains. No domain-specific layers — adaptation happens
    entirely through the OT-weighted loss.

    Args:
        window_size:  Input lookback length (default 24, matches CWPDDA/MCTL)
        horizon:      Steps ahead to predict (default 1)
        hidden_dim:   LSTM hidden units (128, same as CWPDDA)
        n_layers:     LSTM layers (2, same depth as CWPDDA)
        d_embed:      Embedding dimension — dimensionality of the OT feature space
        dropout:      LSTM dropout
        alpha:        Weight on feature alignment term ‖z_s − z_t‖² in OT cost.
                      MSE source loss is ~0.02 on normalised data; squared L2 of
                      tanh-normalised 128-dim vectors is ~0.5–2.0, so alpha=1.0
                      makes the feature alignment term the same order as source MSE.
                      (Original paper: 0.001 for raw pixel CE; much larger needed for MSE.)
        lambda_t:     Weight on label consistency term L_t in OT cost.
                      0.5 makes label propagation contribute ~half the source MSE.
    """

    def __init__(
        self,
        window_size: int = 24,
        horizon: int = 1,
        hidden_dim: int = 128,
        n_layers: int = 2,
        d_embed: int = 128,
        dropout: float = 0.1,
        alpha: float = 0.001,
        lambda_t: float = 0.1,
    ):
        super().__init__()
        self.encoder  = TSEncoder(window_size, hidden_dim, n_layers, d_embed, dropout)
        self.predictor = Predictor(d_embed, horizon)
        self.alpha    = alpha
        self.lambda_t = lambda_t
        self.horizon  = horizon

    # ── Forward ────────────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W) → z: (B, d_embed), unit-norm"""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, W) → y_hat: (B, horizon)

        The predictor outputs in scale-normalised space. We rescale the output
        back to the original [0,1] range using the same per-window max used by
        the encoder — matching the N-BEATS MaxAbs rescaling at inference.
        """
        _, scale = TSEncoder.maxabs_scale(x)           # (B, 1)
        z = self.encode(x)
        return self.predictor(z) * scale               # rescale to [0,1] space

    # ── OT cost matrix ─────────────────────────────────────────────────────────

    def compute_cost_matrix(
        self,
        z_src: torch.Tensor,   # (m, d_embed)
        z_tgt: torch.Tensor,   # (m, d_embed)
        y_src: torch.Tensor,   # (m, horizon)  — source labels
        y_hat_tgt: torch.Tensor,  # (m, horizon) — current target predictions
    ) -> torch.Tensor:
        """
        Compute the m×m OT cost matrix (Eq. 8 of paper, adapted for regression):

          C_ij = α · ‖z_i^s − z_j^t‖²
               + λ_t · MSE(y_i^s, ŷ_j^t)

        The feature term pushes matched pairs' embeddings together.
        The label term pairs source samples with target samples whose predictions
        agree with the source label — implicit pseudo-label propagation.

        Both terms are computed with fixed/frozen embeddings (called before the
        network update step, matching Algorithm 1 of the paper).
        """
        m = z_src.size(0)

        # Feature cost: squared L2 distance (m, m)
        # ‖z_i^s − z_j^t‖² = ‖z_s‖² + ‖z_t‖² − 2 z_s z_t^T
        sq_src = (z_src ** 2).sum(1, keepdim=True)        # (m, 1)
        sq_tgt = (z_tgt ** 2).sum(1, keepdim=True)        # (m, 1)
        feat_cost = sq_src + sq_tgt.T - 2 * z_src @ z_tgt.T  # (m, m)
        feat_cost = feat_cost.clamp(min=0.0)               # numerical safety

        # Label cost: MSE(y_i^s, ŷ_j^t) for each (i, j) pair (m, m)
        # y_src:     (m, H) → expand to (m, 1, H)
        # y_hat_tgt: (m, H) → expand to (1, m, H)
        label_cost = ((y_src.unsqueeze(1) - y_hat_tgt.unsqueeze(0)) ** 2).mean(-1)  # (m, m)

        return self.alpha * feat_cost + self.lambda_t * label_cost

    # ── Full loss (Eq. 9 of paper) ─────────────────────────────────────────────

    def compute_loss(
        self,
        x_src: torch.Tensor,    # (m, W)
        y_src: torch.Tensor,    # (m, H)
        x_tgt: torch.Tensor,    # (m, W)
        gamma: torch.Tensor,    # (m, m) — OT coupling, treated as constant
    ):
        """
        Compute the full DeepJDOT loss (Eq. 9) with differentiable g and f.
        gamma is the OT coupling from the previous step — treated as a fixed
        constant (no gradient flows through it).

        Returns: (total_loss, info_dict)
        """
        z_src    = self.encode(x_src)            # (m, d_embed)
        z_tgt    = self.encode(x_tgt)            # (m, d_embed)
        # Use forward() so MaxAbs rescaling is applied — predictions are in [0,1] space
        y_hat_s  = self.forward(x_src)           # (m, H)
        y_hat_t  = self.forward(x_tgt)           # (m, H)

        # Term 1: source supervised loss
        L_src = F.mse_loss(y_hat_s, y_src)

        # Term 2: feature alignment (γ-weighted squared L2)
        sq_src   = (z_src ** 2).sum(1, keepdim=True)
        sq_tgt   = (z_tgt ** 2).sum(1, keepdim=True)
        feat_mat = (sq_src + sq_tgt.T - 2 * z_src @ z_tgt.T).clamp(min=0.0)  # (m, m)
        L_feat   = (gamma * feat_mat).sum()

        # Term 3: label propagation to target (γ-weighted MSE)
        label_mat = ((y_src.unsqueeze(1) - y_hat_t.unsqueeze(0)) ** 2).mean(-1)  # (m, m)
        L_label   = (gamma * label_mat).sum()

        total = L_src + self.alpha * L_feat + self.lambda_t * L_label

        return total, {
            "L_src":   L_src.item(),
            "L_feat":  L_feat.item(),
            "L_label": L_label.item(),
        }

    # ── Inference ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_numpy_batched(
        self,
        X: np.ndarray,
        device: str = "cpu",
        batch_size: int = 2048,
    ) -> np.ndarray:
        """Batch inference over numpy array. Returns (N, H) float32."""
        self.eval()
        parts = []
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).float().to(device)
            parts.append(self.forward(xb).cpu().numpy())
        return np.concatenate(parts) if parts else np.empty((0, self.horizon), dtype=np.float32)
