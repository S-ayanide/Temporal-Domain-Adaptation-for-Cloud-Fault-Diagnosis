"""
plot_mctl.py
============
Generates publication-quality figures for the MCTL replication study.

Run:
    python plot_mctl.py

Outputs (saved in ./figures/):
    mctl_comparison.png        — full baseline comparison bar chart (MAE + MSE)
    mctl_vs_transfer.png       — MCTL vs transfer baselines only (WANN, MCTL)
    mctl_training_curve.png    — KL loss curve during Stage 2a
    mctl_improvement.png       — before/after fix comparison
    mctl_architecture.png      — MCTL 3-stage architecture diagram
    mctl_all.png               — combined 2-panel figure (dissertation-ready)
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   10,
    "legend.fontsize":   9.5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

os.makedirs("figures", exist_ok=True)

# ─── Experimental data ────────────────────────────────────────────────────────

METHODS = ["ARIMA", "LSTM", "GRU", "CNN-LSTM", "Autoformer",
           "BHT-ARIMA", "TS2Vec", "WANN", "MCTL"]

MAE = {
    "ARIMA":      2.0972e-01,
    "LSTM":       1.7883e-01,
    "GRU":        1.7374e-01,
    "CNN-LSTM":   1.7284e-01,
    "Autoformer": 1.9922e-01,
    "BHT-ARIMA":  1.7543e-01,
    "TS2Vec":     1.7607e-01,
    "WANN":       1.7916e-01,
    "MCTL":       1.7292e-01,
}

MSE = {
    "ARIMA":      9.7417e-02,
    "LSTM":       5.4819e-02,
    "GRU":        5.4938e-02,
    "CNN-LSTM":   5.4708e-02,
    "Autoformer": 6.3649e-02,
    "BHT-ARIMA":  5.7122e-02,
    "TS2Vec":     5.5796e-02,
    "WANN":       5.9760e-02,
    "MCTL":       5.3643e-02,
}

MAPE = {
    "ARIMA":      1.3859e+00,
    "LSTM":       1.2159e+00,
    "GRU":        1.0697e+00,
    "CNN-LSTM":   1.0716e+00,
    "Autoformer": 1.3613e+00,
    "BHT-ARIMA":  1.1404e+00,
    "TS2Vec":     1.0912e+00,
    "WANN":       1.1753e+00,
    "MCTL":       1.1125e+00,
}

SMAPE = {
    "ARIMA":      8.6436e-01,
    "LSTM":       8.8877e-01,
    "GRU":        8.9064e-01,
    "CNN-LSTM":   8.9222e-01,
    "Autoformer": 9.2364e-01,
    "BHT-ARIMA":  8.9461e-01,
    "TS2Vec":     9.0289e-01,
    "WANN":       9.0049e-01,
    "MCTL":       8.8730e-01,
}

# Colour: grey for classical, blue for neural non-transfer, red for MCTL
COLORS = {
    "ARIMA":      "#aaaaaa",
    "LSTM":       "#5590c8",
    "GRU":        "#4a7eb5",
    "CNN-LSTM":   "#3d6fa0",
    "Autoformer": "#7ebdc2",
    "BHT-ARIMA":  "#999999",
    "TS2Vec":     "#6baed6",
    "WANN":       "#f4a261",
    "MCTL":       "#d62728",
}

TRANSFER = {"WANN", "MCTL"}


def save(fname):
    plt.savefig(f"figures/{fname}")
    plt.close()
    print(f"  Saved figures/{fname}")


# ─── 1. Full comparison bar chart (MAE + MSE side by side) ──────────────────

def full_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(METHODS))
    mae_vals = [MAE[m] for m in METHODS]
    mse_vals = [MSE[m] for m in METHODS]
    colors   = [COLORS[m] for m in METHODS]
    hatches  = ["xx" if m == "MCTL" else ("///" if m in TRANSFER else "") for m in METHODS]

    for ax, vals, ylabel, title in [
        (ax1, mae_vals, "MAE (normalised)", "Mean Absolute Error"),
        (ax2, mse_vals, "MSE (normalised)", "Mean Squared Error"),
    ]:
        bars = ax.bar(x, vals, color=colors, hatch=hatches,
                      edgecolor="white", linewidth=0.6, alpha=0.9, width=0.65)
        # Highlight MCTL bar with border
        bars[-1].set_edgecolor("#a00000")
        bars[-1].set_linewidth(1.8)

        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{v:.4f}",
                    ha="center", va="bottom", fontsize=7, color="#333333",
                    rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(vals) * 1.22)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#aaaaaa", label="Classical (no transfer)"),
        mpatches.Patch(facecolor="#4a7eb5", label="Neural (no transfer)"),
        mpatches.Patch(facecolor="#f4a261", hatch="///", label="WANN (transfer)"),
        mpatches.Patch(facecolor="#d62728", hatch="xx",  label="MCTL (ours)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.03), framealpha=0.9)
    fig.suptitle("MCTL vs All Baselines — Alibaba Container CPU Prediction",
                 y=1.08, fontsize=13, fontweight="bold")

    plt.tight_layout()
    save("mctl_comparison.png")


full_comparison()


# ─── 2. All 4 metrics ────────────────────────────────────────────────────────

def four_metric_chart():
    metrics = [
        (MAE,   "MAE",    "Mean Absolute Error"),
        (MSE,   "MSE",    "Mean Squared Error"),
        (MAPE,  "MAPE",   "Mean Abs % Error"),
        (SMAPE, "sMAPE",  "Symmetric MAPE"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    x = np.arange(len(METHODS))
    colors  = [COLORS[m] for m in METHODS]
    hatches = ["xx" if m == "MCTL" else ("///" if m in TRANSFER else "") for m in METHODS]

    for ax, (data, ylabel, title) in zip(axes, metrics):
        vals = [data[m] for m in METHODS]
        bars = ax.bar(x, vals, color=colors, hatch=hatches,
                      edgecolor="white", linewidth=0.6, alpha=0.9, width=0.65)
        bars[-1].set_edgecolor("#a00000")
        bars[-1].set_linewidth(1.8)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5,
                    color="#333333", rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(vals) * 1.25)

    legend_elements = [
        mpatches.Patch(facecolor="#aaaaaa", label="Classical"),
        mpatches.Patch(facecolor="#4a7eb5", label="Neural (no transfer)"),
        mpatches.Patch(facecolor="#f4a261", hatch="///", label="WANN (transfer)"),
        mpatches.Patch(facecolor="#d62728", hatch="xx",  label="MCTL"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.03), framealpha=0.9)
    fig.suptitle("MCTL — All Metrics vs All Baselines",
                 y=1.08, fontsize=13, fontweight="bold")
    plt.tight_layout()
    save("mctl_all_metrics.png")


four_metric_chart()


# ─── 3. KL training curve (Stage 2a) ────────────────────────────────────────

def kl_curve():
    epochs = list(range(10, 101, 10))
    kl_vals = [0.00505, 0.00475, 0.00461, 0.00448, 0.00431,
               0.00449, 0.00431, 0.00432, 0.00426, 0.00417]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, kl_vals, "o-", color="#d62728", linewidth=2,
            markersize=6, markerfacecolor="white", markeredgewidth=1.8)
    ax.fill_between(epochs, kl_vals, alpha=0.12, color="#d62728")

    # Annotate start and end
    ax.annotate(f"Start: {kl_vals[0]:.5f}",
                xy=(epochs[0], kl_vals[0]),
                xytext=(epochs[0] + 5, kl_vals[0] + 0.00015),
                fontsize=9, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))
    ax.annotate(f"End: {kl_vals[-1]:.5f}",
                xy=(epochs[-1], kl_vals[-1]),
                xytext=(epochs[-1] - 25, kl_vals[-1] - 0.00025),
                fontsize=9, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))

    ax.set_xlabel("Stage 2a Epoch")
    ax.set_ylabel("Contrastive KL Loss")
    ax.set_title("Stage 2a: KL Alignment Loss During Contrastive Transfer\n"
                 "(decreasing = target encoder aligning to source structure)")
    ax.yaxis.grid(True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(5, 105)

    plt.tight_layout()
    save("mctl_kl_curve.png")


kl_curve()


# ─── 4. Before/after fix improvement ────────────────────────────────────────

def improvement_chart():
    versions = ["Broken\n(KL≡0)", "Fixed\n50 epochs", "Fixed\n100 epochs"]
    mae_v  = [1.7437e-01, 1.7322e-01, 1.7292e-01]
    mse_v  = [5.3794e-02, 5.3724e-02, 5.3643e-02]
    mape_v = [1.1646e+00, 1.1303e+00, 1.1125e+00]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
    bar_colors = ["#bbbbbb", "#f4a261", "#d62728"]

    for ax, vals, ylabel, title in [
        (axes[0], mae_v,  "MAE",  "MAE"),
        (axes[1], mse_v,  "MSE",  "MSE"),
        (axes[2], mape_v, "MAPE", "MAPE"),
    ]:
        bars = ax.bar(versions, vals, color=bar_colors,
                      edgecolor="white", linewidth=0.8, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.005,
                    f"{v:.5f}",
                    ha="center", va="bottom", fontsize=8.5)

        # Improvement arrow
        improvement = vals[0] - vals[-1]
        pct = improvement / vals[0] * 100
        ax.annotate(f"−{pct:.2f}%",
                    xy=(2, vals[-1]),
                    xytext=(1, vals[0] * 0.97),
                    fontsize=9, color="#006400", fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color="#006400", lw=1.2))

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(vals) * 1.15)

    fig.suptitle("MCTL: Improvement from Contrastive Loss Fix",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save("mctl_improvement.png")


improvement_chart()


# ─── 5. Architecture diagram ─────────────────────────────────────────────────

def architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(bx, by, bw, bh, label, color="#4C72B0", fontsize=9,
            text_color="white", style="round,pad=0.1", lw=1.2, alpha=1.0):
        patch = FancyBboxPatch((bx - bw / 2, by - bh / 2), bw, bh,
                               boxstyle=style, facecolor=color,
                               edgecolor="white", linewidth=lw,
                               zorder=3, alpha=alpha)
        ax.add_patch(patch)
        ax.text(bx, by, label, ha="center", va="center",
                fontsize=fontsize, color=text_color,
                fontweight="bold", zorder=4, multialignment="center")

    def arrow(x1, y1, x2, y2, color="#555555", lw=1.5, ms=12):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=lw, mutation_scale=ms),
                    zorder=2)

    def lbl(lx, ly, text, fontsize=8.5, color="#333333",
            ha="center", style="normal"):
        ax.text(lx, ly, text, ha=ha, va="center",
                fontsize=fontsize, color=color, zorder=5, style=style)

    def stage_box(sx, sy, sw, sh, label, color):
        patch = FancyBboxPatch((sx, sy), sw, sh,
                               boxstyle="round,pad=0.15",
                               facecolor=color, edgecolor=color,
                               linewidth=1.5, zorder=1, alpha=0.10)
        ax.add_patch(patch)
        ax.text(sx + sw / 2, sy + sh + 0.18, label,
                ha="center", va="bottom", fontsize=9,
                color=color, fontweight="bold", zorder=5)

    # ── Stage background boxes ────────────────────────────────────────────────
    stage_box(0.2, 0.4, 3.8, 5.9, "Stage 1: Source Pretraining", "#2ca02c")
    stage_box(4.3, 0.4, 5.2, 5.9, "Stage 2a: Contrastive KL Alignment", "#1f77b4")
    stage_box(9.7, 0.4, 4.1, 5.9, "Stage 2b: Fine-tuning", "#d62728")

    # ── STAGE 1 ───────────────────────────────────────────────────────────────
    box(2.1, 5.8, 3.0, 0.65,
        "Google Source Windows\n(>= 80 pts, W=24)",
        color="#2ca02c", fontsize=8.5)
    box(2.1, 4.6, 2.6, 0.6,
        "Source TCN Encoder\n(3 layers, H=128)",
        color="#1a7a1a")
    arrow(2.1, 5.47, 2.1, 4.9)

    box(2.1, 3.4, 2.2, 0.55,
        "Temp. Linear Head\n(H -> 1)",
        color="#3aaa3a", fontsize=8.5)
    arrow(2.1, 4.3, 2.1, 3.68)

    box(2.1, 2.3, 1.8, 0.5,
        "MSE Loss  L1",
        color="#57c957", fontsize=8.5, text_color="black")
    arrow(2.1, 3.12, 2.1, 2.55)

    lbl(2.1, 1.75, "Pretrains source encoder\nto predict Google workloads",
        fontsize=7.5, color="#1a7a1a", style="italic")

    # ── STAGE 2a ──────────────────────────────────────────────────────────────
    # Source (frozen)
    box(5.3, 5.8, 2.4, 0.55,
        "Source Windows\n(frozen encoder input)",
        color="#2ca02c", fontsize=8)
    box(5.3, 4.75, 2.4, 0.55,
        "Frozen Source Encoder\nf_s  (no gradient)",
        color="#888888")
    arrow(5.3, 5.52, 5.3, 5.03)

    # Target
    box(8.2, 5.8, 2.4, 0.55,
        "Alibaba Target Windows\n(<= 100 pts, W=24)",
        color="#9467bd", fontsize=8)
    box(8.2, 4.75, 2.4, 0.55,
        "Target TCN Encoder\nf_t  (trainable)",
        color="#6a3d9a")
    arrow(8.2, 5.52, 8.2, 5.03)

    # Mixup
    box(5.3, 3.65, 2.4, 0.6,
        "Mixup Augmentation\nx_m = lam*x1 + (1-lam)*x2\nlam ~ Beta(1,1)",
        color="#aec7e8", text_color="black", fontsize=7.5)
    box(8.2, 3.65, 2.4, 0.6,
        "Mixup Augmentation\nx_m = lam*x1 + (1-lam)*x2\nlam ~ Beta(1,1)",
        color="#c5b0d5", text_color="black", fontsize=7.5)
    arrow(5.3, 4.47, 5.3, 3.95)
    arrow(8.2, 4.47, 8.2, 3.95)

    # PAPN probability
    box(5.3, 2.5, 2.4, 0.6,
        "PAPN Probability  p_s\nSoftmax over K=32 negatives\ntemperature tau=0.1",
        color="#aec7e8", text_color="black", fontsize=7.5)
    box(8.2, 2.5, 2.4, 0.6,
        "PAPN Probability  p_t\nSoftmax over K=32 negatives\ntemperature tau=0.1",
        color="#c5b0d5", text_color="black", fontsize=7.5)
    arrow(5.3, 3.35, 5.3, 2.8)
    arrow(8.2, 3.35, 8.2, 2.8)

    # KL loss
    box(6.75, 1.5, 2.0, 0.55,
        "KL Loss\nKL( p_s || p_t )",
        color="#1f77b4")
    arrow(5.3, 2.2, 6.2, 1.72)
    arrow(8.2, 2.2, 7.3, 1.72)

    lbl(6.75, 0.9, "Aligns target encoder\nto source similarity structure",
        fontsize=7.5, color="#1f77b4", style="italic")

    # ── STAGE 2b ──────────────────────────────────────────────────────────────
    box(11.7, 5.8, 2.2, 0.55,
        "Alibaba Target Windows\n(labelled, W=24)",
        color="#9467bd", fontsize=8)
    box(11.7, 4.7, 2.2, 0.6,
        "Target Encoder  f_t\n(fine-tuned jointly)",
        color="#6a3d9a")
    box(11.7, 3.6, 1.8, 0.55,
        "Regression Head\n(H -> 1)",
        color="#c5003b")
    box(11.7, 2.55, 1.6, 0.5,
        "MSE Loss  L2",
        color="#e377c2", text_color="black", fontsize=8.5)
    box(11.7, 1.5, 1.8, 0.5,
        "Prediction  y_hat",
        color="#bcbd22", text_color="black")

    arrow(11.7, 5.52, 11.7, 5.0)
    arrow(11.7, 4.4,  11.7, 3.88)
    arrow(11.7, 3.32, 11.7, 2.8)
    arrow(11.7, 2.3,  11.7, 1.75)

    lbl(11.7, 0.9, "Prediction on target domain\n(source encoder discarded)",
        fontsize=7.5, color="#c5003b", style="italic")

    # ── Stage-to-stage arrows ─────────────────────────────────────────────────
    ax.annotate("Frozen weights\ntransferred",
                xy=(4.3, 4.75), xytext=(3.7, 4.75),
                fontsize=7.5, color="#555555", ha="right",
                arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.2))

    ax.annotate("Trained weights\ntransferred",
                xy=(9.7, 4.75), xytext=(9.1, 4.75),
                fontsize=7.5, color="#555555", ha="right",
                arrowprops=dict(arrowstyle="-|>", color="#6a3d9a", lw=1.2))

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(7.0, 6.78,
            "MCTL Architecture  —  Mixed Contrastive Transfer Learning (3-Stage Curriculum)",
            ha="center", va="center", fontsize=11.5,
            fontweight="bold", color="#222222")

    plt.tight_layout()
    save("mctl_architecture.png")


architecture_diagram()

print("\nAll MCTL figures saved to ./figures/")
