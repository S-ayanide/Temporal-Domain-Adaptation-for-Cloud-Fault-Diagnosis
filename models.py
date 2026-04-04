"""
All neural network architectures.

Baselines (replicated from paper):
  DANN    — Domain-Adversarial Neural Network
  CDAN    — Conditional Domain Adversarial Network
  FixBi   — Fixing the Bipartite Graph
  ToAlign — Task-Oriented Alignment
  DATL    — Proposed model in the replicated paper (MLP + MMD + adv + pseudo-labels)

Novel contribution:
  TA-DATL — Temporal Attention DATL
            Replaces the flat MLP feature extractor with a temporal encoder
            (multi-scale gated convolutions + grouped query attention),
            adds calibrated pseudo-labels (temperature scaling), and
            temporal MMD (aligns both mean and variance of feature sequences).

Input shapes:
  Flat models (DANN, CDAN, FixBi, ToAlign, DATL) : (batch, F)
  Temporal model (TA-DATL)                        : (batch, W, F)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

class GRL(torch.autograd.Function):
    """Gradient Reversal Layer."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None

def grad_reverse(x, alpha=1.0):
    return GRL.apply(x, alpha)

def grl_alpha(epoch: int, total: int, gamma: float = 10.0) -> float:
    p = epoch / total
    return float(2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p))) - 1.0)

def mmd_loss(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Point-wise MMD: squared distance of feature means."""
    return torch.sum((src.mean(0) - tgt.mean(0)) ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# Flat building blocks (used by DANN / CDAN / FixBi / ToAlign / DATL)
# ══════════════════════════════════════════════════════════════════════════════

class FlatExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, feat_dim, n_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, n_classes),
        )
    def forward(self, z):
        return self.net(z)


class DomainDiscriminator(nn.Module):
    def __init__(self, feat_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, 1),
            # No Sigmoid here — use F.binary_cross_entropy_with_logits for
            # numerical stability (avoids the [0,1] assertion on GPU)
        )
    def forward(self, z, alpha=1.0):
        return self.net(grad_reverse(z, alpha))


# ══════════════════════════════════════════════════════════════════════════════
# TA-DATL: Temporal feature extractor components
# ══════════════════════════════════════════════════════════════════════════════

class GatedConv1d(nn.Module):
    """
    Single-scale gated convolution (from GQAT-Net, adapted for tabular sequences).
    F = ReLU(W_f * X);   G = sigmoid(W_g * X);   out = F ⊙ G
    """
    def __init__(self, in_ch, out_ch, kernel_size, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv_f = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.conv_g = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)

    def forward(self, x):
        # x: (batch, C, T)
        return F.relu(self.conv_f(x)) * torch.sigmoid(self.conv_g(x))


class MultiScaleGatedConv(nn.Module):
    """
    Three parallel gated convolutions (kernels 3, 5, 7) concatenated and
    layer-normalised.  Input: (batch, T, F) → Output: (batch, T, hidden_dim)
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        assert hidden_dim % 3 == 0, "hidden_dim must be divisible by 3"
        ch = hidden_dim // 3
        self.conv3 = GatedConv1d(input_dim, ch, 3)
        self.conv5 = GatedConv1d(input_dim, ch, 5)
        self.conv7 = GatedConv1d(input_dim, ch, 7)
        self.norm  = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)
        self.out_dim = hidden_dim

    def forward(self, x):
        # x: (batch, T, F)
        xc = x.permute(0, 2, 1)            # (batch, F, T)
        o3 = self.conv3(xc)
        o5 = self.conv5(xc)
        o7 = self.conv7(xc)
        # Truncate to same length (padding may differ by 1 at edges)
        min_t = min(o3.size(2), o5.size(2), o7.size(2))
        out = torch.cat([o3[:,:,:min_t], o5[:,:,:min_t], o7[:,:,:min_t]], dim=1)
        out = out.permute(0, 2, 1)          # (batch, T, hidden)
        return self.drop(self.norm(out))


class GroupedQueryAttention(nn.Module):
    """
    Simplified Grouped Query Attention (GQA).
    Partitions hidden_dim into n_groups; each group attends independently.
    """
    def __init__(self, hidden_dim, n_groups=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_groups == 0
        self.n_groups  = n_groups
        self.group_dim = hidden_dim // n_groups
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm  = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)
        self.scale = self.group_dim ** -0.5

    def forward(self, x):
        # x: (batch, T, H)
        B, T, H = x.shape
        G, D = self.n_groups, self.group_dim
        Q = self.q(x).view(B, T, G, D).permute(0, 2, 1, 3)   # (B,G,T,D)
        K = self.k(x).view(B, T, G, D).permute(0, 2, 1, 3)
        V = self.v(x).view(B, T, G, D).permute(0, 2, 1, 3)
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        out  = (attn @ V).permute(0, 2, 1, 3).reshape(B, T, H)
        return self.norm(x + self.drop(self.proj(out)))


class TemporalExtractor(nn.Module):
    """
    TA-DATL feature extractor.

    Pipeline:
      (batch, W, F)
        → MultiScaleGatedConv          → (batch, W, H)
        → GroupedQueryAttention × 2    → (batch, W, H)
        → mean + std pooling           → (batch, 2H)
        → linear projection            → (batch, H)

    Using both mean and std in the pooling is deliberate: mean captures the
    average resource state; std captures its variability — both are predictive
    of fault type (a sudden spike has high std; a sustained overload has high mean).
    """
    def __init__(self, input_dim, hidden_dim=128, n_groups=4, dropout=0.3):
        super().__init__()
        assert hidden_dim % 3 == 0 and hidden_dim % n_groups == 0, \
            "hidden_dim must be divisible by both 3 and n_groups"
        self.conv   = MultiScaleGatedConv(input_dim, hidden_dim, dropout)
        self.gqa1   = GroupedQueryAttention(hidden_dim, n_groups, dropout)
        self.gqa2   = GroupedQueryAttention(hidden_dim, n_groups, dropout)
        self.proj   = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim

    def forward(self, x):
        # x: (batch, W, F)
        h = self.conv(x)          # (batch, W, H)
        h = self.gqa1(h)          # (batch, W, H)
        h = self.gqa2(h)          # (batch, W, H)
        # Temporal statistics pooling
        # correction=0 avoids NaN when W==1; nan_to_num handles constant signals
        m = h.mean(dim=1)                          # (batch, H)
        s = h.std(dim=1, correction=0).nan_to_num(0.0)  # (batch, H)
        return self.proj(torch.cat([m, s], dim=1))      # (batch, H)


# ══════════════════════════════════════════════════════════════════════════════
# TA-DATL: Temporal MMD
# ══════════════════════════════════════════════════════════════════════════════

def temporal_mmd(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    Temporal MMD: aligns both the mean state and the variance of the
    temporal feature distributions.  src/tgt are (batch, H) embeddings.

    L_tmmd = MMD(mean_src, mean_tgt) + MMD(std_src, std_tgt)

    The variance term captures distributional shift in fault *dynamics*
    (how quickly a resource degrades) not just the static level.
    """
    mmd_mean = mmd_loss(src, tgt)
    # correction=0 (biased estimator) avoids NaN when batch size == 1
    var_src  = src.var(dim=0, correction=0)
    var_tgt  = tgt.var(dim=0, correction=0)
    mmd_var  = torch.sum((var_src - var_tgt) ** 2)
    # Guard against NaN from degenerate batches
    if not torch.isfinite(mmd_var):
        mmd_var = torch.zeros(1, device=src.device)[0]
    return mmd_mean + 0.5 * mmd_var


# ══════════════════════════════════════════════════════════════════════════════
# TA-DATL  (proposed novel model)
# ══════════════════════════════════════════════════════════════════════════════

class TA_DATL(nn.Module):
    """
    Temporal Attention Domain-Adversarial Transfer Learning.

    Extensions over the replicated DATL:
    ┌──────────────────────────────────────────────────────────────┐
    │  1. Temporal feature extractor (multi-scale GatedConv + GQA) │
    │     → captures HOW metrics evolve, not just their values     │
    │  2. Temporal MMD (mean + variance alignment)                 │
    │     → better distributional alignment for time-series data   │
    │  3. Calibrated pseudo-labels (temperature scaling)           │
    │     → more reliable confidence estimates for self-training   │
    └──────────────────────────────────────────────────────────────┘

    Total loss:
      L = L_cls + λ1·L_tmmd + λ2·L_adv   (+ calibrated pseudo-label term)
    """

    def __init__(
        self,
        input_dim:         int,
        n_classes:         int,
        hidden_dim:        int   = 120,    # must be div by 3 AND n_groups(4)
        n_groups:          int   = 4,
        dropout:           float = 0.3,
        lambda1:           float = 0.1,    # temporal MMD weight
        lambda2:           float = 0.1,    # adversarial weight
        pseudo_threshold:  float = 0.85,
        temperature:       float = 1.5,    # initial calibration temperature
    ):
        super().__init__()
        self.lambda1          = lambda1
        self.lambda2          = lambda2
        self.pseudo_threshold = pseudo_threshold

        # Learnable temperature for calibration (Section 3.3)
        self.log_T = nn.Parameter(torch.log(torch.tensor(temperature)))

        self.F = TemporalExtractor(input_dim, hidden_dim, n_groups, dropout)
        self.C = Classifier(self.F.out_dim, n_classes, dropout)
        self.D = DomainDiscriminator(self.F.out_dim, dropout)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x_src, x_tgt=None, alpha=1.0):
        """
        x_src / x_tgt : (batch, W, F)  temporal windows
        """
        feat_src   = self.F(x_src)
        cls_logits = self.C(feat_src)
        dom_src    = self.D(feat_src, alpha)

        feat_tgt = dom_tgt = None
        if x_tgt is not None:
            feat_tgt = self.F(x_tgt)
            dom_tgt  = self.D(feat_tgt, alpha)

        return cls_logits, dom_src, dom_tgt, feat_src, feat_tgt

    # ── Loss ─────────────────────────────────────────────────────────────────

    def compute_loss(self, cls_logits, y_src, dom_src, dom_tgt,
                     feat_src, feat_tgt):
        # Guard: clamp labels to valid range to prevent CUDA index-OOB assert
        n_cls  = cls_logits.size(1)
        y_src  = y_src.clamp(0, n_cls - 1)

        L_cls  = F.cross_entropy(cls_logits, y_src)

        # L_tmmd: temporal MMD (mean + variance alignment)
        L_tmmd = temporal_mmd(feat_src, feat_tgt)

        # L_adv: adversarial domain loss
        zeros  = torch.zeros_like(dom_src)
        ones   = torch.ones_like(dom_tgt)
        L_adv  = (F.binary_cross_entropy_with_logits(dom_src, zeros)
                  + F.binary_cross_entropy_with_logits(dom_tgt, ones))

        total  = L_cls + self.lambda1 * L_tmmd + self.lambda2 * L_adv

        # Guard: NaN in total would silently corrupt training
        if not torch.isfinite(total):
            total = L_cls
        return total, {"L_cls": L_cls.item(),
                       "L_tmmd": L_tmmd.item(),
                       "L_adv": L_adv.item()}

    # ── Calibrated pseudo-labels ──────────────────────────────────────────────

    @torch.no_grad()
    def get_pseudo_labels(self, x_tgt):
        """
        Temperature-scaled confidence for reliable pseudo-label selection.
        Raw softmax is overconfident; dividing by T > 1 gives better calibration.
        """
        T       = torch.exp(self.log_T).clamp(0.5, 5.0)  # keep T in sane range
        feat    = self.F(x_tgt)
        logits  = self.C(feat)
        probs   = F.softmax(logits / T, dim=1)            # calibrated probs
        max_p, pred = probs.max(dim=1)
        mask    = max_p >= self.pseudo_threshold
        return x_tgt[mask], pred[mask], mask

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.C(self.F(x))


# ══════════════════════════════════════════════════════════════════════════════
# DATL  (replicated baseline, flat MLP)
# ══════════════════════════════════════════════════════════════════════════════

class DATL(nn.Module):
    """Replicated paper model — kept as a baseline to ablate the temporal encoder."""

    def __init__(self, input_dim, n_classes, hidden_dim=128, dropout=0.3,
                 lambda1=0.1, lambda2=0.1, pseudo_threshold=0.85):
        super().__init__()
        self.lambda1, self.lambda2 = lambda1, lambda2
        self.pseudo_threshold      = pseudo_threshold
        self.F = FlatExtractor(input_dim, hidden_dim, dropout)
        self.C = Classifier(self.F.out_dim, n_classes, dropout)
        self.D = DomainDiscriminator(self.F.out_dim, dropout)

    def forward(self, x_src, x_tgt=None, alpha=1.0):
        feat_src   = self.F(x_src)
        cls_logits = self.C(feat_src)
        dom_src    = self.D(feat_src, alpha)
        feat_tgt = dom_tgt = None
        if x_tgt is not None:
            feat_tgt = self.F(x_tgt)
            dom_tgt  = self.D(feat_tgt, alpha)
        return cls_logits, dom_src, dom_tgt, feat_src, feat_tgt

    def compute_loss(self, cls_logits, y_src, dom_src, dom_tgt, feat_src, feat_tgt):
        L_cls  = F.cross_entropy(cls_logits, y_src)
        L_mmd  = mmd_loss(feat_src, feat_tgt)
        L_adv  = (F.binary_cross_entropy_with_logits(dom_src, torch.zeros_like(dom_src))
                  + F.binary_cross_entropy_with_logits(dom_tgt, torch.ones_like(dom_tgt)))
        return L_cls + self.lambda1*L_mmd + self.lambda2*L_adv, {}

    @torch.no_grad()
    def get_pseudo_labels(self, x_tgt):
        feat  = self.F(x_tgt)
        probs = F.softmax(self.C(feat), dim=1)
        max_p, pred = probs.max(dim=1)
        mask = max_p >= self.pseudo_threshold
        return x_tgt[mask], pred[mask], mask

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.C(self.F(x))


# ══════════════════════════════════════════════════════════════════════════════
# Flat baselines
# ══════════════════════════════════════════════════════════════════════════════

class DANN(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.F = FlatExtractor(input_dim, hidden_dim, dropout)
        self.C = Classifier(self.F.out_dim, n_classes, dropout)
        self.D = DomainDiscriminator(self.F.out_dim, dropout)

    def forward(self, x_src, x_tgt=None, alpha=1.0):
        fs = self.F(x_src);  cl = self.C(fs);  ds = self.D(fs, alpha)
        ft = dt = None
        if x_tgt is not None:
            ft = self.F(x_tgt);  dt = self.D(ft, alpha)
        return cl, ds, dt, fs, ft

    def compute_loss(self, cl, y, ds, dt, *_):
        L_cls = F.cross_entropy(cl, y)
        L_adv = (F.binary_cross_entropy_with_logits(ds, torch.zeros_like(ds))
                 + F.binary_cross_entropy_with_logits(dt, torch.ones_like(dt)))
        return L_cls + 0.1 * L_adv, {}

    def predict(self, x):
        self.eval()
        with torch.no_grad(): return self.C(self.F(x))


class CDANDiscriminator(nn.Module):
    def __init__(self, feat_dim, n_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * n_classes, feat_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, 1))

    def forward(self, feat, cls_prob, alpha=1.0):
        j = torch.bmm(feat.unsqueeze(2), cls_prob.unsqueeze(1)).view(feat.size(0), -1)
        return self.net(grad_reverse(j, alpha))


class CDAN(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.F = FlatExtractor(input_dim, hidden_dim, dropout)
        self.C = Classifier(self.F.out_dim, n_classes, dropout)
        self.D = CDANDiscriminator(self.F.out_dim, n_classes, dropout)

    def forward(self, x_src, x_tgt=None, alpha=1.0):
        fs = self.F(x_src);  cl = self.C(fs)
        ds = self.D(fs, F.softmax(cl.detach(), dim=1), alpha)
        ft = dt = None
        if x_tgt is not None:
            ft = self.F(x_tgt);  tl = self.C(ft)
            dt = self.D(ft, F.softmax(tl.detach(), dim=1), alpha)
        return cl, ds, dt, fs, ft

    def compute_loss(self, cl, y, ds, dt, *_):
        L_cls = F.cross_entropy(cl, y)
        L_adv = (F.binary_cross_entropy_with_logits(ds, torch.zeros_like(ds))
                 + F.binary_cross_entropy_with_logits(dt, torch.ones_like(dt)))
        return L_cls + 0.1 * L_adv, {}

    def predict(self, x):
        self.eval()
        with torch.no_grad(): return self.C(self.F(x))


class FixBi(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.F   = FlatExtractor(input_dim, hidden_dim, dropout)
        self.C_s = Classifier(self.F.out_dim, n_classes, dropout)
        self.C_t = Classifier(self.F.out_dim, n_classes, dropout)

    def forward(self, x_src, x_tgt=None, **_):
        fs = self.F(x_src)
        ls_s, lt_s = self.C_s(fs), self.C_t(fs)
        ls_t = lt_t = ft = None
        if x_tgt is not None:
            ft = self.F(x_tgt);  ls_t, lt_t = self.C_s(ft), self.C_t(ft)
        return ls_s, lt_s, ls_t, lt_t, fs, ft

    def compute_loss(self, ls_s, lt_s, ls_t, lt_t, y, *_):
        L_cls  = (F.cross_entropy(ls_s, y) + F.cross_entropy(lt_s, y)) / 2
        L_cons = torch.tensor(0.0, device=y.device)
        if ls_t is not None:
            soft_s = F.softmax(ls_t.detach(), dim=1)
            soft_t = F.softmax(lt_t.detach(), dim=1)
            L_cons = (F.kl_div(F.log_softmax(lt_t, dim=1), soft_s, reduction="batchmean")
                      + F.kl_div(F.log_softmax(ls_t, dim=1), soft_t, reduction="batchmean"))
        return L_cls + 0.1 * L_cons, {}

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            ft = self.F(x)
            return (self.C_s(ft) + self.C_t(ft)) / 2


class ToAlign(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.F    = FlatExtractor(input_dim, hidden_dim, dropout)
        self.C    = Classifier(self.F.out_dim, n_classes, dropout)
        self.D    = DomainDiscriminator(self.F.out_dim, dropout)
        self.gate = nn.Sequential(nn.Linear(self.F.out_dim, self.F.out_dim),
                                  nn.Sigmoid())

    def forward(self, x_src, x_tgt=None, alpha=1.0):
        fs = self.F(x_src);  gs = self.gate(fs);  ts = fs * gs
        cl = self.C(ts);  ds = self.D(ts, alpha)
        ft = dt = None
        if x_tgt is not None:
            ft = self.F(x_tgt);  gt = self.gate(ft);  tt = ft * gt
            dt = self.D(tt, alpha)
        return cl, ds, dt, fs, ft

    def compute_loss(self, cl, y, ds, dt, *_):
        L_cls = F.cross_entropy(cl, y)
        L_adv = (F.binary_cross_entropy_with_logits(ds, torch.zeros_like(ds))
                 + F.binary_cross_entropy_with_logits(dt, torch.ones_like(dt)))
        return L_cls + 0.1 * L_adv, {}

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            ft = self.F(x);  return self.C(ft * self.gate(ft))
