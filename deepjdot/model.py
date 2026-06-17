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
      - 2-layer LSTM, hidden_dim units
      - Take the final hidden state h_T as the sequence representation
      - Project through FC + Tanh to get the d_embed-dimensional embedding z

    The tanh normalises the embedding space, which helps keep the squared
    L2 distance in the OT cost matrix on a stable scale.
    """

    def __init__(self, window_size: int = 24, hidden_dim: int = 64,
                 n_layers: int = 2, d_embed: int = 64, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, d_embed),
            nn.Tanh(),
        )
        self.d_embed = d_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W) → z: (B, d_embed)"""
        h, _ = self.lstm(x.unsqueeze(-1))   # (B, W, hidden_dim)
        return self.proj(h[:, -1, :])        # (B, d_embed)


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
        hidden_dim:   LSTM hidden units (paper uses 128 for image feature maps;
                      64 is sufficient for 1D time-series)
        n_layers:     LSTM layers (2, same depth as CWPDDA)
        d_embed:      Embedding dimension — dimensionality of the OT feature space
        dropout:      LSTM dropout
        alpha:        Weight on feature alignment term ‖z_s − z_t‖² in OT cost
                      (paper uses 0.001 for image features; 0.01 default here
                       since LSTM embeddings are lower-dimensional)
        lambda_t:     Weight on label consistency term L_t in OT cost
                      (paper uses 0.0001 for CE; 0.1 default here for MSE
                       since MSE on [0,1] scale is much smaller than CE)
    """

    def __init__(
        self,
        window_size: int = 24,
        horizon: int = 1,
        hidden_dim: int = 64,
        n_layers: int = 2,
        d_embed: int = 64,
        dropout: float = 0.1,
        alpha: float = 0.01,
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
        """x: (B, W) → z: (B, d_embed)"""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W) → y_hat: (B, horizon)"""
        return self.predictor(self.encode(x))

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
        y_hat_s  = self.predictor(z_src)         # (m, H)
        y_hat_t  = self.predictor(z_tgt)         # (m, H)

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
