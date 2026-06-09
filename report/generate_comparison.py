"""
generate_comparison.py — Generate method comparison document with visualizations.

Compares: CWPDDA, MC-CWPDDA, MCTL, Tr-Predictor (TrAdaBoost.R2-LSTM).

Run:
    python report/generate_comparison.py
Output:
    report/method_comparison.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

OUT_DIR = os.path.join(os.path.dirname(__file__))
OUT_PDF = os.path.join(OUT_DIR, "method_comparison.pdf")

# ── Colour palette ──────────────────────────────────────────────────────────
C = {
    "cwpdda":   "#2563EB",   # blue
    "mc":       "#7C3AED",   # purple
    "mctl":     "#059669",   # green
    "tr":       "#DC2626",   # red
    "neutral":  "#6B7280",
    "bg":       "#F8FAFC",
    "light":    "#E2E8F0",
    "text":     "#1E293B",
    "gold":     "#F59E0B",
}

METHODS = ["CWPDDA", "MC-CWPDDA", "MCTL", "Tr-Predictor"]
METHOD_COLORS = [C["cwpdda"], C["mc"], C["mctl"], C["tr"]]


# ── Helper: rounded rectangle ────────────────────────────────────────────────
def rbox(ax, x, y, w, h, color, text, fontsize=8, text_color="white", alpha=0.92):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor="white",
                         linewidth=0.8, alpha=alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold", zorder=4,
            wrap=True)


def arrow(ax, x0, y0, x1, y1, color="#475569", lw=1.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=10),
                zorder=5)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Title + overview table
# ════════════════════════════════════════════════════════════════════════════
def page_title_table(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(C["bg"])

    # ── Title block ────────────────────────────────────────────────────────
    ax_title = fig.add_axes([0.05, 0.82, 0.90, 0.14])
    ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1)
    ax_title.axis("off")
    ax_title.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                      boxstyle="round,pad=0.02",
                                      facecolor=C["cwpdda"], alpha=0.9))
    ax_title.text(0.5, 0.62,
                  "Cloud Workload Prediction: Method Comparison",
                  ha="center", va="center", fontsize=18,
                  color="white", fontweight="bold")
    ax_title.text(0.5, 0.22,
                  "CWPDDA  ·  MC-CWPDDA  ·  MCTL  ·  Tr-Predictor",
                  ha="center", va="center", fontsize=11, color="#BFDBFE")

    # ── Overview table ─────────────────────────────────────────────────────
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.75])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    cols = ["Attribute", "CWPDDA", "MC-CWPDDA", "MCTL", "Tr-Predictor"]
    col_w = [0.22, 0.195, 0.195, 0.195, 0.195]
    col_x = [0.0]
    for w in col_w[:-1]:
        col_x.append(col_x[-1] + w)

    rows = [
        ("Problem",
         "CPU/Memory\nprediction\n(Google→Alibaba)",
         "CPU/Memory\nprediction\n(hybrid domains)",
         "Few-shot\nworkload\n(Google→Alibaba)",
         "Small-sample\nworkload\n(any→any)"),
        ("Transfer\nStrategy",
         "Adversarial\n+ MMD\ndisentanglement",
         "Curriculum\n(pretraining →\ncontrastive →\njoint)",
         "Contrastive\nKL alignment\n(frozen source)",
         "Two-stage\nTrAdaBoost\nreweighting"),
        ("Supervision\nLevel",
         "Full supervision\n(source + target\nlabels)",
         "Full supervision\n(3-stage\ncurriculum)",
         "Few-shot\n(minimal\ntarget labels)",
         "Full supervision\n(source heavy,\ntarget sparse)"),
        ("Learning\nParadigm",
         "Zero-shot transfer\nat test time\n(nearest-neighbour\nsource retrieval)",
         "Curriculum\nlearning\n(staged\nfine-tuning)",
         "Few-shot\ncontrastive\nlearning",
         "Instance-based\ntransfer\n(sample reweighting)"),
        ("Core\nArchitecture",
         "3-branch attention\n+ LSTM predictor\n+ GRL discriminator",
         "3-branch attention\n+ contrastive head\n+ GRL discriminator",
         "TCN encoder\n+ regression head\n(source frozen)",
         "Ensemble of\nLSTM weak\nlearners"),
        ("Loss\nFunction",
         "MSE + MMD\ndisentanglement\n+ adversarial (GRL)",
         "MSE + MMD +\nGRL + InfoNCE\ncontrastive + KL",
         "InfoNCE\ncontrastive\n+ KL divergence",
         "Weighted MSE\n(AdaBoost.R2\nsample weights)"),
        ("Key\nInnovation",
         "Cross-attention\nK/V = NN source\nretrieval",
         "Combines CWPDDA\n+ MCTL via\ncurriculum stages",
         "Mixup + PAPN\nfor distribution\nalignment",
         "Stage 1: boost\ntarget; Stage 2:\ndecay source"),
        ("Source\nSelection",
         "All source data\n(no ranking)",
         "All source data\n(no ranking)",
         "All source data\n(no ranking)",
         "TWED + Transfer\nEntropy ranking\n(top-k selection)"),
    ]

    row_h = 0.095
    header_y = 0.97

    # Header
    for j, (col, cx, cw) in enumerate(zip(cols, col_x, col_w)):
        fc = C["neutral"] if j == 0 else METHOD_COLORS[j - 1]
        ax.add_patch(FancyBboxPatch((cx + 0.003, header_y - 0.042),
                                    cw - 0.006, 0.042,
                                    boxstyle="round,pad=0.005",
                                    facecolor=fc, alpha=0.92,
                                    edgecolor="white", lw=0.8, zorder=3))
        ax.text(cx + cw / 2, header_y - 0.021, col,
                ha="center", va="center", fontsize=8.5,
                color="white", fontweight="bold", zorder=4)

    # Rows
    for i, row in enumerate(rows):
        y = header_y - 0.045 - (i + 1) * row_h
        bg = "#F1F5F9" if i % 2 == 0 else "white"
        for j, (cell, cx, cw) in enumerate(zip(row, col_x, col_w)):
            fc = bg if j > 0 else "#E2E8F0"
            ax.add_patch(FancyBboxPatch((cx + 0.002, y + 0.003),
                                        cw - 0.004, row_h - 0.005,
                                        boxstyle="round,pad=0.003",
                                        facecolor=fc, edgecolor="#CBD5E1",
                                        lw=0.4, zorder=2))
            fc_txt = C["text"] if j > 0 else "#334155"
            fw = "bold" if j == 0 else "normal"
            ax.text(cx + cw / 2, y + row_h / 2 + 0.003, cell,
                    ha="center", va="center", fontsize=6.8,
                    color=fc_txt, fontweight=fw, zorder=3,
                    linespacing=1.35)

    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Architecture diagrams (all 4 methods)
# ════════════════════════════════════════════════════════════════════════════
def _draw_cwpdda(ax, title, color, contrastive=False):
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor(C["bg"])

    ax.text(5, 6.6, title, ha="center", va="center",
            fontsize=9, fontweight="bold", color=color)

    # Input row
    rbox(ax, 0.2, 5.5, 2.0, 0.7, "#64748B", "Source\nWindows", 7)
    rbox(ax, 7.8, 5.5, 2.0, 0.7, "#64748B", "Target\nWindows", 7)

    # Self-attention
    rbox(ax, 0.3, 4.2, 1.8, 0.8, color, "Self-Attn\n(source)", 7)
    rbox(ax, 7.9, 4.2, 1.8, 0.8, color, "Self-Attn\n(target)", 7)

    # Cross-attention
    rbox(ax, 3.8, 3.0, 2.4, 0.9, color, "Cross-Attention\nQ=target, K/V=source", 7)

    # NN Retrieval
    rbox(ax, 0.3, 2.8, 1.8, 0.6, "#475569", "NN Source\nRetrieval", 6.5)

    # Private features
    rbox(ax, 0.3, 1.6, 1.6, 0.7, "#94A3B8", "Private\nFeature", 7)
    rbox(ax, 7.9, 1.6, 1.6, 0.7, "#94A3B8", "Private\nFeature", 7)

    # Shared + predictor
    rbox(ax, 3.8, 1.5, 2.4, 0.9, color, "z_shared\nFeatures", 7)

    # GRL
    rbox(ax, 5.8, 2.8, 1.8, 0.6, C["gold"], "GRL\nDiscriminator", 6.5)

    # Predictor
    rbox(ax, 3.8, 0.2, 2.4, 0.9, "#1E293B", "LSTM Predictor\n→ Forecast", 7)

    # Contrastive head (MC only)
    if contrastive:
        rbox(ax, 7.6, 1.4, 2.1, 0.8, C["mctl"], "Contrastive\nHead (InfoNCE)", 6.5)
        arrow(ax, 5.0, 1.5, 5.0, 1.1)
        arrow(ax, 8.7, 1.4, 8.7, 0.8)
        ax.text(9.5, 0.6, "Lc + Lkl", fontsize=6.5, color=C["mctl"], ha="center")

    # Arrows
    arrow(ax, 1.2, 5.5, 1.2, 5.0)
    arrow(ax, 8.8, 5.5, 8.8, 5.0)
    arrow(ax, 1.2, 4.2, 1.2, 3.4)
    arrow(ax, 8.8, 4.2, 8.8, 3.4)  # → cross attn target
    arrow(ax, 8.8, 3.4, 6.2, 3.4)  # target → cross attn
    arrow(ax, 1.2, 2.8, 3.8, 3.4)  # nn retrieval → cross attn source
    arrow(ax, 5.0, 3.0, 5.0, 2.4)  # cross → z_shared
    arrow(ax, 1.2, 1.6, 5.0, 2.4)  # private src → z_shared
    arrow(ax, 8.7, 1.6, 5.8, 3.4)  # private tgt → GRL
    arrow(ax, 0.9, 1.6, 0.9, 0.9)
    arrow(ax, 9.5, 1.6, 9.5, 0.9)
    ax.text(0.9, 0.7, "MMD\nLoss", fontsize=6, color=color, ha="center")
    ax.text(9.5, 0.7, "Adv\nLoss", fontsize=6, color=C["gold"], ha="center")
    arrow(ax, 5.0, 1.5, 5.0, 1.1)
    arrow(ax, 5.0, 0.2, 5.0, -0.1)
    ax.text(5.0, -0.3, "Prediction + MSE Loss", ha="center", fontsize=7,
            color="#334155", style="italic")


def _draw_mctl(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor(C["bg"])

    ax.text(5, 6.6, "MCTL", ha="center", va="center",
            fontsize=9, fontweight="bold", color=C["mctl"])

    # Stage labels
    for lbl, y in [("Stage 1: Source\nPretraining", 5.5),
                   ("Stage 2: Contrastive\nAlignment", 3.2)]:
        ax.add_patch(FancyBboxPatch((0.1, y - 0.15), 9.8, 1.7,
                                    boxstyle="round,pad=0.05",
                                    facecolor="#F0FDF4", edgecolor=C["mctl"],
                                    lw=0.8, alpha=0.5))
        ax.text(0.4, y + 1.4, lbl, fontsize=7.5, color=C["mctl"],
                fontweight="bold", va="top")

    # Stage 1
    rbox(ax, 0.5, 5.5, 1.8, 0.7, "#64748B", "Source\nData", 7)
    rbox(ax, 3.5, 5.5, 2.2, 0.7, C["mctl"], "TCN Encoder\n(source)", 7)
    rbox(ax, 7.2, 5.5, 2.2, 0.7, "#1E293B", "Regression\nHead", 7)
    arrow(ax, 2.3, 5.85, 3.5, 5.85)
    arrow(ax, 5.7, 5.85, 7.2, 5.85)
    ax.text(8.3, 5.1, "MSE Loss", fontsize=6.5, color=C["mctl"], ha="center")

    # Stage 2
    rbox(ax, 0.3, 3.6, 1.5, 0.7, "#64748B", "Target\nData", 7)
    rbox(ax, 0.3, 2.4, 1.5, 0.7, "#64748B", "Source\nData", 7)
    rbox(ax, 2.6, 3.0, 2.0, 0.8, C["mctl"], "Mixup\nAugment", 7)
    rbox(ax, 5.4, 3.5, 2.0, 0.7, C["mctl"], "TCN Encoder\n(target, trainable)", 6.5)
    rbox(ax, 5.4, 2.4, 2.0, 0.7, "#94A3B8", "TCN Encoder\n(source, frozen)", 6.5)
    rbox(ax, 8.0, 3.0, 1.7, 0.8, C["mctl"], "InfoNCE +\nKL Loss", 7)

    arrow(ax, 1.8, 3.3, 2.6, 3.3)
    arrow(ax, 1.8, 2.7, 2.6, 2.9)
    arrow(ax, 4.6, 3.2, 5.4, 3.8)
    arrow(ax, 4.6, 3.2, 5.4, 2.7)
    arrow(ax, 7.4, 3.85, 8.0, 3.3)
    arrow(ax, 7.4, 2.75, 8.0, 3.1)

    # Stage 3 note
    ax.add_patch(FancyBboxPatch((0.1, 0.1), 9.8, 1.8,
                                boxstyle="round,pad=0.05",
                                facecolor="#F0F9FF", edgecolor=C["mctl"],
                                lw=0.8, alpha=0.5))
    ax.text(0.4, 1.8, "Stage 2 output: Target encoder aligned to source distribution",
            fontsize=7, color=C["mctl"], va="top")
    rbox(ax, 1.0, 0.2, 3.5, 0.9, C["mctl"], "Trained target encoder", 7)
    rbox(ax, 5.5, 0.2, 3.5, 0.9, "#1E293B", "Fine-tuned\nRegression Head → Forecast", 7)
    arrow(ax, 4.5, 0.65, 5.5, 0.65)


def _draw_tr_predictor(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor(C["bg"])

    ax.text(5, 6.6, "Tr-Predictor (TrAdaBoost.R2-LSTM)",
            ha="center", va="center",
            fontsize=9, fontweight="bold", color=C["tr"])

    # Source selection
    rbox(ax, 0.2, 5.6, 2.0, 0.7, "#64748B", "All Source\nDomains", 7)
    rbox(ax, 0.2, 4.5, 2.0, 0.7, "#64748B", "Target\nDomain", 7)
    rbox(ax, 3.0, 5.0, 2.2, 0.9, C["tr"], "TWED + TE\nSource Ranking", 7)
    rbox(ax, 6.4, 5.0, 2.0, 0.9, C["tr"], "Top-k\nSources", 7)
    arrow(ax, 2.2, 5.95, 3.0, 5.4)
    arrow(ax, 2.2, 4.85, 3.0, 5.1)
    arrow(ax, 5.2, 5.45, 6.4, 5.45)

    # Stage 1
    ax.add_patch(FancyBboxPatch((0.1, 2.5), 4.5, 2.0,
                                boxstyle="round,pad=0.05",
                                facecolor="#FFF1F2", edgecolor=C["tr"],
                                lw=0.8, alpha=0.5))
    ax.text(0.3, 4.45, "Stage 1: Boost target weights (source frozen)",
            fontsize=6.8, color=C["tr"], fontweight="bold")
    rbox(ax, 0.3, 3.5, 1.8, 0.7, C["tr"], "LSTM\nWeak Learner", 7)
    rbox(ax, 2.5, 3.5, 1.8, 0.7, "#94A3B8", "Error on\ntarget →  βt", 7)
    rbox(ax, 0.3, 2.6, 1.8, 0.7, "#94A3B8", "Update\ntarget w↑", 7)
    arrow(ax, 2.1, 3.85, 2.5, 3.85)
    arrow(ax, 2.5, 3.5, 1.3, 3.3)

    # Stage 2
    ax.add_patch(FancyBboxPatch((5.0, 2.5), 4.7, 2.0,
                                boxstyle="round,pad=0.05",
                                facecolor="#FFF7ED", edgecolor="#D97706",
                                lw=0.8, alpha=0.5))
    ax.text(5.2, 4.45, "Stage 2: Decay source weights (target frozen)",
            fontsize=6.8, color="#D97706", fontweight="bold")
    rbox(ax, 5.2, 3.5, 1.8, 0.7, C["tr"], "LSTM\nWeak Learner", 7)
    rbox(ax, 7.4, 3.5, 1.8, 0.7, "#94A3B8", "βs × source\nweights↓", 7)
    rbox(ax, 5.2, 2.6, 1.8, 0.7, "#94A3B8", "Source w\nfrozen on tgt", 7)
    arrow(ax, 7.0, 3.85, 7.4, 3.85)
    arrow(ax, 7.4, 3.5, 6.2, 3.3)

    # Arrow source selection → stage 1+2
    arrow(ax, 8.4, 5.0, 3.0, 4.5)
    arrow(ax, 8.4, 5.0, 7.2, 4.5)

    # Ensemble
    rbox(ax, 2.8, 1.0, 4.4, 0.9, "#1E293B",
         "Weighted Ensemble (log(1/βt) weights)\nover last ⌈T/2⌉ hypotheses → Forecast", 7)
    arrow(ax, 2.2, 2.9, 3.0, 1.9)
    arrow(ax, 7.1, 2.9, 6.8, 1.9)

    ax.text(5.0, 0.4, "βs = 1 / (1 + √(2·ln(n)/T))",
            ha="center", fontsize=7.5, color=C["tr"],
            style="italic", fontweight="bold")


def page_architectures(pdf):
    fig = plt.figure(figsize=(11, 17))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Architecture Diagrams", fontsize=14, fontweight="bold",
                 color=C["text"], y=0.99)

    gs = gridspec.GridSpec(4, 1, hspace=0.18,
                           left=0.03, right=0.97,
                           top=0.97, bottom=0.01)

    ax0 = fig.add_subplot(gs[0]); _draw_cwpdda(ax0, "CWPDDA", C["cwpdda"])
    ax1 = fig.add_subplot(gs[1]); _draw_cwpdda(ax1, "MC-CWPDDA", C["mc"], contrastive=True)
    ax2 = fig.add_subplot(gs[2]); _draw_mctl(ax2)
    ax3 = fig.add_subplot(gs[3]); _draw_tr_predictor(ax3)

    for ax, col in zip([ax0, ax1, ax2, ax3], METHOD_COLORS):
        for spine in ax.spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(1.5)

    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Transfer learning taxonomy + few-shot vs zero-shot spectrum
# ════════════════════════════════════════════════════════════════════════════
def page_transfer_spectrum(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Transfer Learning Paradigms & Few-Shot Spectrum",
                 fontsize=13, fontweight="bold", color=C["text"])

    # ── Top: spectrum bar ──────────────────────────────────────────────────
    ax1 = fig.add_axes([0.06, 0.72, 0.88, 0.20])
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 3)
    ax1.axis("off")
    ax1.set_facecolor(C["bg"])

    ax1.text(5, 2.8, "Supervision Spectrum: Source Labels vs Target Labels",
             ha="center", va="center", fontsize=10, fontweight="bold",
             color=C["text"])

    # Gradient bar
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("spec",
                                             ["#DC2626", "#7C3AED",
                                              "#2563EB", "#059669"])
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax1.imshow(grad, aspect="auto", extent=[0.5, 9.5, 1.0, 1.8],
               cmap=cmap, alpha=0.85)

    labels = [
        (1.5, "Zero-Shot\n(no target labels)"),
        (3.5, "Few-Shot\n(few target labels)"),
        (6.5, "Semi-supervised\n(some target labels)"),
        (8.5, "Full supervision\n(all labels)"),
    ]
    for xp, lbl in labels:
        ax1.plot([xp, xp], [1.0, 1.8], color="white", lw=1, alpha=0.6)
        ax1.text(xp, 0.9, lbl, ha="center", va="top", fontsize=7,
                 color=C["text"])

    positions = {"Tr-Predictor": 2.2, "MCTL": 3.8,
                 "MC-CWPDDA": 6.5, "CWPDDA": 8.2}
    colors_map = {"Tr-Predictor": C["tr"], "MCTL": C["mctl"],
                  "MC-CWPDDA": C["mc"], "CWPDDA": C["cwpdda"]}
    for name, xp in positions.items():
        ax1.plot(xp, 1.4, "v", color=colors_map[name], ms=10, zorder=5)
        ax1.text(xp, 2.0, name, ha="center", va="bottom",
                 fontsize=7.5, color=colors_map[name], fontweight="bold")

    # ── Middle: taxonomy tree ──────────────────────────────────────────────
    ax2 = fig.add_axes([0.03, 0.32, 0.94, 0.37])
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 4)
    ax2.axis("off")
    ax2.set_facecolor(C["bg"])
    ax2.text(5, 3.7, "Transfer Learning Taxonomy", ha="center",
             fontsize=10, fontweight="bold", color=C["text"])

    # Root
    rbox(ax2, 3.8, 3.0, 2.4, 0.55, "#334155", "Transfer\nLearning", 8)
    # Level 1
    for x, lbl, col in [(0.5, "Instance\nTransfer", C["tr"]),
                         (3.0, "Feature\nTransfer", C["cwpdda"]),
                         (6.0, "Parameter\nTransfer", C["mctl"]),
                         (8.5, "Relational\nTransfer", C["neutral"])]:
        rbox(ax2, x, 1.8, 1.8, 0.65, col, lbl, 7.5)
        arrow(ax2, 5.0, 3.0, x + 0.9, 2.45)

    # Level 2 — methods
    methods_l2 = [
        (0.3, 0.3, C["tr"],    "Tr-Predictor\n(TrAdaBoost)"),
        (2.5, 0.3, C["cwpdda"],"CWPDDA\n(MMD + GRL)"),
        (5.2, 0.3, C["mctl"],  "MCTL\n(contrastive)"),
        (7.8, 0.3, C["mc"],    "MC-CWPDDA\n(curriculum)"),
    ]
    for x, y, col, lbl in methods_l2:
        rbox(ax2, x, y, 2.0, 0.65, col, lbl, 7)

    arrow(ax2, 1.4, 1.8, 1.3, 0.95)
    arrow(ax2, 3.9, 1.8, 3.5, 0.95)
    arrow(ax2, 6.9, 1.8, 6.2, 0.95)
    arrow(ax2, 6.9, 1.8, 8.8, 0.95)

    # ── Bottom: stage comparison ───────────────────────────────────────────
    ax3 = fig.add_axes([0.03, 0.02, 0.94, 0.28])
    ax3.set_xlim(0, 10); ax3.set_ylim(0, 3)
    ax3.axis("off")
    ax3.set_facecolor(C["bg"])
    ax3.text(5, 2.8, "Training Stages per Method", ha="center",
             fontsize=10, fontweight="bold", color=C["text"])

    stages = {
        "CWPDDA":      [("Source + Target\nJoint Training", C["cwpdda"])],
        "MC-CWPDDA":   [("Stage 1\nSource Pretrain", C["mc"]),
                        ("Stage 2\nContrastive Align", C["mc"]),
                        ("Stage 3\nJoint Fine-tune", C["mc"])],
        "MCTL":        [("Stage 1\nSource Pretrain", C["mctl"]),
                        ("Stage 2\nContrastive KL", C["mctl"])],
        "Tr-Predictor":[("Source\nSelection (TWED+TE)", C["tr"]),
                        ("TrAdaBoost\nStage 1", C["tr"]),
                        ("TrAdaBoost\nStage 2", C["tr"])],
    }
    y_row = [2.2, 1.4, 0.6, -0.2]
    for i, (mname, stg_list) in enumerate(stages.items()):
        y = y_row[i] if i < len(y_row) else 0.1
        ax3.text(0.2, y + 0.5, mname, fontsize=8, fontweight="bold",
                 va="center", color=METHOD_COLORS[i])
        x0 = 1.8
        for j, (slbl, scol) in enumerate(stg_list):
            w = 2.2
            rbox(ax3, x0 + j * (w + 0.3), y + 0.1, w, 0.7, scol, slbl, 7)
            if j < len(stg_list) - 1:
                arrow(ax3, x0 + j * (w + 0.3) + w,
                      y + 0.45,
                      x0 + (j + 1) * (w + 0.3),
                      y + 0.45)

    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Algorithm pseudo-code
# ════════════════════════════════════════════════════════════════════════════
def page_algorithms(pdf):
    fig = plt.figure(figsize=(11, 17))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Algorithm Pseudocode", fontsize=14, fontweight="bold",
                 color=C["text"], y=0.99)

    blocks = [
        {
            "title": "Algorithm 1: CWPDDA — Domain-Adversarial Workload Prediction",
            "color": C["cwpdda"],
            "lines": [
                "Input: Source windows Xs, labels Ys; Target windows Xt",
                "Output: Forecast ŷt for target domain",
                "",
                "1.  FOR each training step:",
                "2.    Retrieve nearest-neighbour source window x̃s ← NN(xt, Xs)  [L2]",
                "3.    z_src_priv ← SelfAttn(proj_src(Xs))",
                "4.    z_tgt_priv ← SelfAttn(proj_tgt(Xt))",
                "5.    z_shared  ← CrossAttn(Q=Xt, K=x̃s, V=x̃s)",
                "",
                "6.    Ly ← MSE(LSTM(z_shared), Ys)                    [prediction]",
                "7.    Lf ← -mmd(z_priv, z_shared)                     [disentangle]",
                "8.    Ld ← GRL_discriminator(z_src_priv, z_tgt_priv)  [adversarial]",
                "9.    L  ← Ly + λ1·Lf + λ2·Ld",
                "10.   Backpropagate L; update all parameters",
                "",
                "11. RETURN LSTM(CrossAttn(Q=Xt_test, K=x̃s, V=x̃s))",
            ],
        },
        {
            "title": "Algorithm 2: MC-CWPDDA — Curriculum Contrastive Domain Adaptation",
            "color": C["mc"],
            "lines": [
                "Input: Source (Xs, Ys), Target (Xt, Yt_few)",
                "Output: Forecast ŷt",
                "",
                "// Stage 1: Source encoder pretraining",
                "1.  Train {proj_src, SelfAttn_src} with MSE(LSTM(z_src), Ys)",
                "",
                "// Stage 2: Contrastive alignment (source frozen)",
                "2.  FREEZE source branch parameters",
                "3.  FOR each batch:",
                "4.    λ ~ Beta(α, α)",
                "5.    x_mix ← λ·xs + (1-λ)·xt                         [cross-domain mixup]",
                "6.    h_anc ← normalize(ContrastiveHead(CrossAttn(x_mix)))",
                "7.    h_pos ← λ·Gc(z_shared_src) + (1-λ)·Gc(z_shared_tgt)",
                "8.    Lc ← InfoNCE(h_anc, h_pos, {h_neg_k})           [K=8 negatives]",
                "9.    Lkl ← KL(q_tgt || q_src)                        [distribution align]",
                "",
                "// Stage 3: Joint fine-tuning",
                "10. UNFREEZE all parameters",
                "11. L ← Ly + λ1·Lf + λ2·Ld + λ3·Lc + λ4·Lkl",
                "12. Train until early-stop on validation MSE",
            ],
        },
        {
            "title": "Algorithm 3: MCTL — Mixed Contrastive Transfer Learning",
            "color": C["mctl"],
            "lines": [
                "Input: Source (Xs, Ys), Target Xt (few-shot, minimal labels)",
                "Output: Forecast ŷt",
                "",
                "// Stage 1: Train source TCN encoder",
                "1.  Train TCN_src end-to-end: min MSE(RegrHead(TCN_src(Xs)), Ys)",
                "2.  FREEZE TCN_src",
                "",
                "// Stage 2: Align target encoder to source distribution",
                "3.  FOR each batch (xs, xt):",
                "4.    λ ~ Beta(α, α)                                   [α=1.0]",
                "5.    x_mix ← λ·xs + (1-λ)·xt                         [mixup]",
                "6.    z_mix_s ← TCN_src(x_mix)  [frozen]",
                "7.    z_mix_t ← TCN_tgt(x_mix)  [trainable]",
                "8.    Sample K=32 in-batch negatives",
                "9.    p_s ← PAPN(z_mix_s, z_pos, {z_neg}) / τ         [InfoNCE, τ=0.1]",
                "10.   p_t ← PAPN(z_mix_t, z_pos, {z_neg}) / τ",
                "11.   Lkl ← p_s·log(p_s/p_t) + (1-p_s)·log((1-p_s)/(1-p_t))",
                "12.   Backprop Lkl; update TCN_tgt only",
                "",
                "// Inference",
                "13. ŷt ← RegrHead(TCN_tgt(Xt))",
            ],
        },
        {
            "title": "Algorithm 4: Two-Stage TrAdaBoost.R2-LSTM (Tr-Predictor)",
            "color": C["tr"],
            "lines": [
                "Input: Source data Ts (n samples), Target data Tt (m samples), T rounds",
                "Output: Ensemble forecast ŷt",
                "",
                "// Pre-step: Source domain selection",
                "0.  FOR each candidate source domain s:",
                "      score(s) ← rank(TWED(tgt,s)) + rank(-TE(s→tgt))",
                "    SELECT top-k sources by score",
                "",
                "// Boosting",
                "1.  w ← uniform(1/N)  where N = n + m",
                "2.  β_s ← 1 / (1 + √(2·ln(n)/T))",
                "3.  FOR t = 1, …, T:",
                "4.    ht ← LSTM trained on (Ts ∪ Tt) with weights w",
                "5.    e_t ← Σ_{i∈Tt} w_i · |ht(xi) - yi| / max|error|",
                "6.    β_t ← e_t / (1 - e_t)",
                "7.    IF t ≤ ⌈T/2⌉:  [Stage 1 — target weight update]",
                "8.      w_i ← w_i · β_t^(1 - |error_i|/D_max)  ∀ i ∈ Tt",
                "        # source weights unchanged",
                "9.    ELSE:          [Stage 2 — source weight decay]",
                "10.     w_i ← w_i · β_s  ∀ i ∈ Ts",
                "        # target weights unchanged",
                "11.   Renormalise w",
                "",
                "12. ŷ ← Σ_{t=⌈T/2⌉+1}^{T} log(1/β_t) · ht(x)  [weighted average]",
            ],
        },
    ]

    axes_h = [0.22, 0.22, 0.22, 0.27]
    tops   = [0.975, 0.735, 0.495, 0.235]

    for blk, h, top in zip(blocks, axes_h, tops):
        ax = fig.add_axes([0.04, top - h, 0.92, h])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")

        # Header
        ax.add_patch(FancyBboxPatch((0, 0.88), 1, 0.12,
                                    boxstyle="round,pad=0.01",
                                    facecolor=blk["color"], alpha=0.9,
                                    edgecolor="none"))
        ax.text(0.5, 0.94, blk["title"], ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold")

        # Code body
        ax.add_patch(FancyBboxPatch((0, 0), 1, 0.88,
                                    boxstyle="round,pad=0.01",
                                    facecolor="#F8FAFC",
                                    edgecolor=blk["color"],
                                    lw=1.2))
        n = len(blk["lines"])
        for li, line in enumerate(blk["lines"]):
            y = 0.84 - li * (0.82 / max(n, 1))
            fc = "#1E293B"
            fw = "normal"
            style = "normal"
            if line.strip().startswith("//"):
                fc = blk["color"]; fw = "bold"
            elif line.strip().startswith("Input:") or line.strip().startswith("Output:"):
                fc = "#475569"; style = "italic"
            ax.text(0.02, y, line, ha="left", va="top",
                    fontsize=6.8, color=fc, fontweight=fw,
                    style=style, family="monospace")

    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Loss function comparison + hyperparameter radar
# ════════════════════════════════════════════════════════════════════════════
def page_losses_and_radar(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Loss Functions & Hyperparameter Profiles",
                 fontsize=13, fontweight="bold", color=C["text"])

    # ── Left: loss component stacked bar ──────────────────────────────────
    ax1 = fig.add_axes([0.05, 0.45, 0.42, 0.48])
    ax1.set_facecolor(C["bg"])

    methods_short = ["CWPDDA", "MC-CWPDDA", "MCTL", "Tr-Pred"]
    # Normalised contribution of each loss component (illustrative weights)
    loss_components = {
        "Prediction\n(MSE/NLL)":  [0.50, 0.40, 0.00, 0.70],
        "Adversarial\n(GRL)":     [0.25, 0.20, 0.00, 0.00],
        "Disentangle\n(MMD)":     [0.25, 0.15, 0.00, 0.00],
        "Contrastive\n(InfoNCE)": [0.00, 0.15, 0.55, 0.00],
        "KL\nDivergence":         [0.00, 0.10, 0.45, 0.00],
        "Reweighting\n(AdaBoost)":[0.00, 0.00, 0.00, 0.30],
    }
    loss_colors = ["#2563EB","#DC2626","#F59E0B","#059669","#7C3AED","#EF4444"]
    bottom = np.zeros(4)
    xs = np.arange(4)
    for (lbl, vals), lc in zip(loss_components.items(), loss_colors):
        v = np.array(vals)
        ax1.bar(xs, v, bottom=bottom, color=lc, label=lbl,
                width=0.6, edgecolor="white", lw=0.8)
        for xi, (b, vi) in enumerate(zip(bottom, v)):
            if vi > 0.05:
                ax1.text(xi, b + vi / 2, f"{vi:.0%}",
                         ha="center", va="center", fontsize=7,
                         color="white", fontweight="bold")
        bottom += v

    ax1.set_xticks(xs)
    ax1.set_xticklabels(methods_short, fontsize=9)
    ax1.set_ylabel("Relative Loss Contribution", fontsize=8)
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Loss Component Breakdown", fontsize=9, fontweight="bold",
                  color=C["text"])
    ax1.legend(fontsize=6.5, loc="upper right",
               framealpha=0.9, ncol=1)
    ax1.spines[["top","right"]].set_visible(False)

    # ── Right: radar chart ─────────────────────────────────────────────────
    categories = ["Data\nEfficiency", "Architecture\nComplexity",
                  "Interpretability", "Transfer\nFlexibility",
                  "Source\nSelection", "Scalability"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    scores = {
        "CWPDDA":     [3, 4, 2, 3, 1, 4],
        "MC-CWPDDA":  [3, 5, 2, 4, 1, 3],
        "MCTL":       [5, 3, 3, 3, 1, 4],
        "Tr-Predictor": [4, 2, 5, 5, 5, 3],
    }

    ax2 = fig.add_axes([0.55, 0.45, 0.42, 0.48], polar=True)
    ax2.set_facecolor(C["bg"])
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, size=7)
    ax2.set_ylim(0, 5)
    ax2.set_yticks([1, 2, 3, 4, 5])
    ax2.set_yticklabels(["1","2","3","4","5"], size=6, color="gray")
    ax2.grid(color="#CBD5E1", lw=0.5)

    for (mname, sc), col in zip(scores.items(), METHOD_COLORS):
        vals = sc + [sc[0]]
        ax2.plot(angles, vals, "o-", color=col, lw=2, ms=4, label=mname)
        ax2.fill(angles, vals, color=col, alpha=0.08)

    ax2.set_title("Method Profile (1=low, 5=high)",
                  fontsize=9, fontweight="bold", color=C["text"], pad=15)
    ax2.legend(loc="lower right", bbox_to_anchor=(1.3, -0.15),
               fontsize=7.5, framealpha=0.9)

    # ── Bottom: hyperparameter table ───────────────────────────────────────
    ax3 = fig.add_axes([0.03, 0.02, 0.94, 0.38])
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.axis("off")
    ax3.set_facecolor(C["bg"])
    ax3.text(0.5, 0.97, "Key Hyperparameters", ha="center", va="top",
             fontsize=10, fontweight="bold", color=C["text"])

    hp_rows = [
        ("Window size",           "24",       "24",        "24",       "24"),
        ("Hidden dim / units",    "64",        "64",       "128 (TCN)", "64 (LSTM)"),
        ("Forecast horizon",      "1 step",    "1 step",   "1 step",   "1 step"),
        ("Dropout",               "0.1",       "0.1",      "0.2",      "0.1"),
        ("GRL α / β",             "10, 0.75",  "10, 0.75", "—",        "—"),
        ("Loss weights λ",        "λ1=λ2=0.01","λ1–4=0.1/0.05","—",   "βs = f(n,T)"),
        ("Mixup α",               "—",         "1.0",      "1.0",      "—"),
        ("Contrastive τ",         "—",         "1.0",      "0.1",      "—"),
        ("Negatives K",           "—",         "8",        "32",       "—"),
        ("Boosting rounds T",     "—",         "—",        "—",        "20"),
        ("Source selection",      "None",      "None",     "None",     "TWED+TE top-5"),
        ("Optimiser",             "Adam 1e-3","Adam 1e-3","Adam 1e-3","Adam 1e-3"),
    ]

    hdr = ["Hyperparameter", "CWPDDA", "MC-CWPDDA", "MCTL", "Tr-Predictor"]
    col_x = [0.0, 0.22, 0.40, 0.58, 0.76]
    col_w = [0.21, 0.17, 0.17, 0.17, 0.23]
    header_y = 0.90
    row_h    = 0.063

    for j, (h, cx, cw) in enumerate(zip(hdr, col_x, col_w)):
        fc = C["neutral"] if j == 0 else METHOD_COLORS[j - 1]
        ax3.add_patch(FancyBboxPatch((cx+0.003, header_y-0.036), cw-0.006, 0.036,
                                     boxstyle="round,pad=0.003",
                                     facecolor=fc, alpha=0.9,
                                     edgecolor="white", lw=0.6))
        ax3.text(cx+cw/2, header_y-0.018, h, ha="center", va="center",
                 fontsize=7.5, color="white", fontweight="bold")

    for i, row in enumerate(hp_rows):
        y = header_y - 0.04 - (i+1)*row_h
        bg = "#F1F5F9" if i % 2 == 0 else "white"
        for j, (cell, cx, cw) in enumerate(zip(row, col_x, col_w)):
            fc = "#E2E8F0" if j == 0 else bg
            ax3.add_patch(FancyBboxPatch((cx+0.002, y+0.003), cw-0.004, row_h-0.005,
                                         boxstyle="round,pad=0.002",
                                         facecolor=fc, edgecolor="#CBD5E1",
                                         lw=0.3))
            ax3.text(cx+cw/2, y+row_h/2+0.003, cell,
                     ha="center", va="center", fontsize=6.8,
                     color=C["text"],
                     fontweight="bold" if j == 0 else "normal")

    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — TWED + TrAdaBoost weight dynamics (visual explainer)
# ════════════════════════════════════════════════════════════════════════════
def page_tr_predictor_detail(pdf):
    np.random.seed(42)
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Tr-Predictor Detail: Source Selection + Weight Dynamics",
                 fontsize=13, fontweight="bold", color=C["text"])

    # ── Top-left: TWED illustration ────────────────────────────────────────
    ax1 = fig.add_axes([0.05, 0.55, 0.43, 0.37])
    t = np.linspace(0, 4*np.pi, 100)
    ts_target = 0.5 + 0.4*np.sin(t) + 0.05*np.random.randn(100)
    ts_close  = 0.5 + 0.38*np.sin(t + 0.3) + 0.05*np.random.randn(100)
    ts_far    = 0.5 + 0.4*np.cos(t*1.5)   + 0.05*np.random.randn(100)

    ax1.plot(ts_target, color=C["neutral"], lw=2.0, label="Target", zorder=4)
    ax1.plot(ts_close,  color=C["mctl"],    lw=1.5, ls="--",
             label="Source A (TWED=0.12, TE=0.41)", zorder=3)
    ax1.plot(ts_far,    color="#F59E0B",    lw=1.5, ls=":",
             label="Source B (TWED=0.89, TE=0.08)", zorder=3)

    # Warp paths (illustrative dashed lines)
    for i in range(0, 100, 15):
        ax1.plot([i, i], [ts_target[i], ts_close[i]],
                 color=C["mctl"], lw=0.6, alpha=0.4)

    ax1.set_title("TWED: Time Warp Edit Distance", fontsize=9,
                  fontweight="bold", color=C["text"])
    ax1.legend(fontsize=7, loc="lower right")
    ax1.set_xlabel("Time (5-min bins)", fontsize=8)
    ax1.set_ylabel("CPU Utilisation", fontsize=8)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.set_facecolor(C["bg"])

    # ── Top-right: TE illustration ─────────────────────────────────────────
    ax2 = fig.add_axes([0.55, 0.55, 0.43, 0.37])
    n_pts = 200
    x_strong = np.cumsum(np.random.randn(n_pts) * 0.1)
    y_caused  = np.roll(x_strong, 2) + np.random.randn(n_pts) * 0.05
    x_weak    = np.cumsum(np.random.randn(n_pts) * 0.1)

    ax2.scatter(x_strong[:-1], y_caused[1:], s=8, alpha=0.4, color=C["mctl"],
                label="Source A → Target (TE=0.41, strong)")
    ax2.scatter(x_weak[:-1],   y_caused[1:], s=8, alpha=0.4, color=C["gold"],
                label="Source B → Target (TE=0.08, weak)")

    # Regression lines
    for xs, col in [(x_strong, C["mctl"]), (x_weak, C["gold"])]:
        m, b = np.polyfit(xs[:-1], y_caused[1:], 1)
        xr = np.linspace(xs.min(), xs.max(), 50)
        ax2.plot(xr, m*xr + b, color=col, lw=2, alpha=0.8)

    ax2.set_title("Transfer Entropy: X_t → Y_{t+1} influence", fontsize=9,
                  fontweight="bold", color=C["text"])
    ax2.legend(fontsize=7)
    ax2.set_xlabel("Source value X_t", fontsize=8)
    ax2.set_ylabel("Target value Y_{t+1}", fontsize=8)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.set_facecolor(C["bg"])

    # ── Bottom: weight dynamics across rounds ──────────────────────────────
    ax3 = fig.add_axes([0.07, 0.05, 0.86, 0.43])
    T_rounds = 20
    n_src, n_tgt = 100, 20
    beta_s = 1 / (1 + np.sqrt(2 * np.log(n_src) / T_rounds))

    w_src = np.ones(T_rounds + 1)
    w_tgt = np.ones(T_rounds + 1)
    # Simulate weight evolution
    for t in range(1, T_rounds + 1):
        if t <= T_rounds // 2:
            # Stage 1: target weights decrease/increase (AdaBoost.R2 update)
            e = 0.3 - t * 0.01          # mock decreasing error
            beta_t = e / (1 - e + 1e-9)
            w_tgt[t] = w_tgt[t-1] * (beta_t ** 0.5)
            w_src[t] = w_src[t-1]
        else:
            # Stage 2: source weights decay
            w_src[t] = w_src[t-1] * beta_s
            w_tgt[t] = w_tgt[t-1]

    # Normalise
    total = w_src + w_tgt
    w_src_n = w_src / total
    w_tgt_n = w_tgt / total

    rounds = np.arange(T_rounds + 1)
    ax3.fill_between(rounds, 0, w_src_n, alpha=0.35, color=C["cwpdda"],
                     label="Source weight proportion")
    ax3.fill_between(rounds, w_src_n, 1, alpha=0.35, color=C["tr"],
                     label="Target weight proportion")
    ax3.plot(rounds, w_src_n, color=C["cwpdda"], lw=2)
    ax3.plot(rounds, w_tgt_n, color=C["tr"], lw=2)

    ax3.axvline(T_rounds // 2, color=C["neutral"], lw=1.5, ls="--")
    ax3.text(T_rounds // 2 + 0.3, 0.92, "Stage 1 → Stage 2",
             fontsize=8, color=C["neutral"], va="top")
    ax3.text(3, 0.2, "Stage 1:\nBoost target weights\n(source frozen)",
             fontsize=8, color=C["tr"], ha="center",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["tr"], alpha=0.8))
    ax3.text(15, 0.75, "Stage 2:\nDecay source weights\n(target frozen)",
             fontsize=8, color=C["cwpdda"], ha="center",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["cwpdda"], alpha=0.8))

    ax3.set_xlabel("Boosting Round", fontsize=9)
    ax3.set_ylabel("Normalised Weight Share", fontsize=9)
    ax3.set_title(
        f"TrAdaBoost.R2 Weight Dynamics  "
        f"(n_src={n_src}, n_tgt={n_tgt}, T={T_rounds}, "
        f"β_s={beta_s:.3f})",
        fontsize=9, fontweight="bold", color=C["text"])
    ax3.set_xlim(0, T_rounds)
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=8, loc="center right")
    ax3.spines[["top","right"]].set_visible(False)
    ax3.set_facecolor(C["bg"])

    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 7 — MC-CWPDDA curriculum stages + contrastive loss explainer
# ════════════════════════════════════════════════════════════════════════════
def page_mc_cwpdda_detail(pdf):
    np.random.seed(7)
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("MC-CWPDDA & MCTL: Contrastive Learning Detail",
                 fontsize=13, fontweight="bold", color=C["text"])

    # ── Left: embedding space before/after alignment ───────────────────────
    ax1 = fig.add_axes([0.05, 0.52, 0.42, 0.40])
    ax1.set_facecolor(C["bg"])
    ax1.set_xlim(-3, 3); ax1.set_ylim(-3, 3)
    ax1.set_aspect("equal")

    n = 60
    src = np.random.randn(n, 2) * 0.7 + np.array([1.2,  0.8])
    tgt = np.random.randn(n, 2) * 0.5 + np.array([-1.0, -0.5])

    ax1.scatter(src[:, 0], src[:, 1], s=20, alpha=0.5,
                color=C["cwpdda"], label="Source (before)")
    ax1.scatter(tgt[:, 0], tgt[:, 1], s=20, alpha=0.5, marker="^",
                color=C["tr"], label="Target (before)")

    src2 = np.random.randn(n, 2) * 0.5 + np.array([0.2, 0.3])
    tgt2 = np.random.randn(n, 2) * 0.5 + np.array([0.0, 0.0])
    ax1.scatter(src2[:, 0], src2[:, 1], s=20, alpha=0.4,
                color=C["mctl"], label="Source (after)", marker="s")
    ax1.scatter(tgt2[:, 0], tgt2[:, 1], s=20, alpha=0.4, marker="D",
                color=C["mc"], label="Target (after)")

    # KDE-style ellipses
    for ctr, col, ls in [([1.2, 0.8], C["cwpdda"], "--"),
                          ([-1.0, -0.5], C["tr"], "--"),
                          ([0.2, 0.3], C["mctl"], "-"),
                          ([0.0, 0.0], C["mc"], "-")]:
        ell = matplotlib.patches.Ellipse(ctr, width=1.8, height=1.4,
                                          angle=30, fill=False,
                                          edgecolor=col, lw=1.2, ls=ls,
                                          alpha=0.7)
        ax1.add_patch(ell)

    ax1.axhline(0, color="#CBD5E1", lw=0.5)
    ax1.axvline(0, color="#CBD5E1", lw=0.5)
    ax1.set_title("Feature Space: Before vs After\nContrastive Alignment",
                  fontsize=9, fontweight="bold", color=C["text"])
    ax1.legend(fontsize=6.5, ncol=2, loc="upper left")
    ax1.set_xlabel("Embedding dim 1", fontsize=8)
    ax1.set_ylabel("Embedding dim 2", fontsize=8)
    ax1.spines[["top","right"]].set_visible(False)

    # ── Right: InfoNCE illustration ────────────────────────────────────────
    ax2 = fig.add_axes([0.55, 0.52, 0.42, 0.40])
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 6)
    ax2.axis("off")
    ax2.set_facecolor(C["bg"])
    ax2.text(5, 5.7, "InfoNCE / PAPN Contrastive Loss", ha="center",
             fontsize=9, fontweight="bold", color=C["text"])

    # Anchor
    rbox(ax2, 3.8, 4.2, 2.4, 0.8, C["mc"], "Anchor\nx_mix = λ·xs + (1-λ)·xt", 7.5)

    # Positive
    rbox(ax2, 6.5, 2.8, 2.8, 0.7, C["mctl"], "Positive\n(same domain mix)", 7)
    # Negatives
    for k, y in enumerate([1.7, 0.6]):
        rbox(ax2, 6.5, y, 2.8, 0.6, "#94A3B8", f"Negative {k+1}\n(different sample)", 6.5)

    # Loss formula
    ax2.add_patch(FancyBboxPatch((0.2, 0.0), 5.8, 1.2,
                                  boxstyle="round,pad=0.05",
                                  facecolor="#F0F9FF", edgecolor=C["mc"],
                                  lw=1.0, alpha=0.8))
    ax2.text(3.1, 0.6,
             "L = -log( exp(sim(a,pos)/τ) /\n"
             "          [exp(sim(a,pos)/τ) + Σk exp(sim(a,neg_k)/τ)] )",
             ha="center", va="center", fontsize=7,
             color=C["text"], family="monospace")

    arrow(ax2, 5.0, 4.2, 6.5, 3.15)
    arrow(ax2, 5.0, 4.2, 6.5, 2.0)
    arrow(ax2, 5.0, 4.2, 6.5, 0.9)

    ax2.text(5.5, 3.5, "sim = cos(·)/τ", fontsize=7,
             color=C["mc"], rotation=30)

    # ── Bottom: curriculum timeline ────────────────────────────────────────
    ax3 = fig.add_axes([0.05, 0.03, 0.90, 0.42])
    ax3.set_xlim(0, 12); ax3.set_ylim(0, 4)
    ax3.axis("off")
    ax3.set_facecolor(C["bg"])
    ax3.text(6, 3.8, "MC-CWPDDA Curriculum Training Timeline",
             ha="center", fontsize=10, fontweight="bold", color=C["text"])

    stage_data = [
        (0.3,  3.2, C["cwpdda"],  "Stage 1\nSource Pretraining",
         "• Train source branch\n  (proj_src + SelfAttn_src)\n• MSE loss on source labels\n• Target branch not touched"),
        (4.3,  3.2, C["mctl"],    "Stage 2\nContrastive Alignment",
         "• Freeze source branch\n• Cross-domain mixup\n• InfoNCE + KL loss\n• Align target to source dist."),
        (8.3,  3.2, C["mc"],      "Stage 3\nJoint Fine-tuning",
         "• Unfreeze all parameters\n• Full loss: Ly+Lf+Ld+Lc+Lkl\n• Early stop on val MSE\n• All components optimised"),
    ]

    for x, y, col, title, desc in stage_data:
        ax3.add_patch(FancyBboxPatch((x, y - 2.6), 3.5, 2.8,
                                     boxstyle="round,pad=0.1",
                                     facecolor=col, alpha=0.15,
                                     edgecolor=col, lw=1.5))
        ax3.text(x + 1.75, y + 0.05, title, ha="center",
                 fontsize=8.5, fontweight="bold", color=col)
        ax3.text(x + 0.2, y - 0.3, desc, ha="left", va="top",
                 fontsize=7.5, color=C["text"], linespacing=1.5)

    # Arrows between stages
    for x_arrow in [3.8, 7.8]:
        ax3.annotate("", xy=(x_arrow + 0.5, 1.8), xytext=(x_arrow, 1.8),
                     arrowprops=dict(arrowstyle="-|>", color=C["neutral"],
                                     lw=2, mutation_scale=14))

    ax3.text(4.0, 2.05, "source\nfrozen", ha="center", fontsize=7,
             color=C["neutral"])
    ax3.text(8.0, 2.05, "all\nunfrozen", ha="center", fontsize=7,
             color=C["neutral"])

    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating {OUT_PDF} …")

    with PdfPages(OUT_PDF) as pdf:
        # Metadata
        d = pdf.infodict()
        d["Title"]   = "Cloud Workload Prediction: Method Comparison"
        d["Author"]  = "Dissertation Research"
        d["Subject"] = "CWPDDA, MC-CWPDDA, MCTL, Tr-Predictor"

        print("  Page 1: Overview table …")
        page_title_table(pdf)

        print("  Page 2: Architecture diagrams …")
        page_architectures(pdf)

        print("  Page 3: Transfer learning spectrum …")
        page_transfer_spectrum(pdf)

        print("  Page 4: Algorithm pseudocode …")
        page_algorithms(pdf)

        print("  Page 5: Loss functions + radar chart …")
        page_losses_and_radar(pdf)

        print("  Page 6: Tr-Predictor detail …")
        page_tr_predictor_detail(pdf)

        print("  Page 7: MC-CWPDDA contrastive detail …")
        page_mc_cwpdda_detail(pdf)

    print(f"Done → {OUT_PDF}")
