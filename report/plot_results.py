"""
plot_results.py
===============
Generates publication-quality figures for the CWPDDA replication study.

Run:
    python plot_results.py

Outputs (saved in ./figures/):
    few_shot_mae.png       — grouped bar chart: MAE vs training windows
    few_shot_mape.png      — grouped bar chart: MAPE vs training windows
    few_shot_rmse.png      — grouped bar chart: RMSE vs training windows
    few_shot_all.png       — 3-panel combined figure (dissertation-ready)
    cwpdda_advantage.png   — CWPDDA MAE/MAPE reduction over LSTM
    architecture.png       — CWPDDA architecture diagram
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
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

os.makedirs("figures", exist_ok=True)

# ─── Experimental data ────────────────────────────────────────────────────────

N_TRAIN = [200, 500, 1000]

DATA = {
    "ARIMA":  {"mae": [19.3475, 19.3475, 19.3475],
               "mape":[148.34,  148.34,  148.34],
               "rmse":[29.3543, 29.3543, 29.3543]},
    "LSTM":   {"mae": [17.6528, 16.8678, 16.7915],
               "mape":[141.94,  135.93,  134.53],
               "rmse":[22.7938, 22.2495, 22.1620]},
    "CWPDDA": {"mae": [17.3589, 16.5322, 16.3801],
               "mape":[122.72,  122.33,  126.07],
               "rmse":[22.7725, 22.2661, 22.0359]},
}

COLORS = {
    "ARIMA":  "#7f7f7f",
    "LSTM":   "#1f77b4",
    "CWPDDA": "#d62728",
}
HATCHES = {
    "ARIMA":  "//",
    "LSTM":   "",
    "CWPDDA": "xx",
}

METHODS = ["ARIMA", "LSTM", "CWPDDA"]
x = np.arange(len(N_TRAIN))
width = 0.25


def save(filename: str) -> None:
    plt.savefig(f"figures/{filename}")
    plt.close()
    print(f"  Saved figures/{filename}")


# ─── Individual bar charts ────────────────────────────────────────────────────

def grouped_bar(metric_key: str, ylabel: str, filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    offsets = [-width, 0, width]
    for method, offset in zip(METHODS, offsets):
        vals = DATA[method][metric_key]
        bars = ax.bar(x + offset, vals, width * 0.92,
                      label=method,
                      color=COLORS[method],
                      hatch=HATCHES[method],
                      edgecolor="white",
                      linewidth=0.5,
                      alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{v:.1f}",
                    ha="center", va="bottom",
                    fontsize=7.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in N_TRAIN])
    ax.set_xlabel("Target Training Windows (few-shot budget)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save(filename)


grouped_bar("mae",  "MAE (per-series min-max normalised ×100)",
            "few_shot_mae.png",
            "Few-Shot CPU Workload Prediction: MAE\n(CWPDDA vs baselines, Google → Alibaba transfer)")

grouped_bar("mape", "MAPE (%)",
            "few_shot_mape.png",
            "Few-Shot CPU Workload Prediction: MAPE\n(CWPDDA vs baselines, Google → Alibaba transfer)")

grouped_bar("rmse", "RMSE (per-series min-max normalised ×100)",
            "few_shot_rmse.png",
            "Few-Shot CPU Workload Prediction: RMSE\n(CWPDDA vs baselines, Google → Alibaba transfer)")


# ─── Combined 3-panel figure ─────────────────────────────────────────────────

def combined_figure() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    configs = [
        ("mae",  "MAE"),
        ("mape", "MAPE (%)"),
        ("rmse", "RMSE"),
    ]

    for ax, (metric_key, ylabel) in zip(axes, configs):
        offsets = [-width, 0, width]
        for method, offset in zip(METHODS, offsets):
            vals = DATA[method][metric_key]
            bars = ax.bar(x + offset, vals, width * 0.92,
                          label=method,
                          color=COLORS[method],
                          hatch=HATCHES[method],
                          edgecolor="white",
                          linewidth=0.5,
                          alpha=0.88)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.15,
                        f"{v:.1f}",
                        ha="center", va="bottom",
                        fontsize=6.5, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in N_TRAIN])
        ax.set_xlabel("Train Windows")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.14)
        ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
        ax.set_axisbelow(True)

    handles = [mpatches.Patch(facecolor=COLORS[m], hatch=HATCHES[m],
                               edgecolor="gray", label=m) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.9)
    fig.suptitle("CWPDDA Few-Shot Scaling: Google → Alibaba Container CPU Prediction",
                 y=1.07, fontsize=13, fontweight="bold")

    plt.tight_layout()
    save("few_shot_all.png")


combined_figure()


# ─── Advantage plot ───────────────────────────────────────────────────────────

def advantage_plot() -> None:
    mae_gap  = [DATA["LSTM"]["mae"][i]  - DATA["CWPDDA"]["mae"][i]  for i in range(3)]
    mape_gap = [DATA["LSTM"]["mape"][i] - DATA["CWPDDA"]["mape"][i] for i in range(3)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    bar_kw = dict(width=0.4, color="#d62728", alpha=0.85, edgecolor="white")

    ax1.bar(x, mae_gap, **bar_kw)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in N_TRAIN])
    ax1.set_xlabel("Train Windows")
    ax1.set_ylabel("MAE Reduction (LSTM − CWPDDA)")
    ax1.set_title("MAE Advantage of CWPDDA over LSTM")
    for i, v in enumerate(mae_gap):
        ax1.text(i, v + 0.01, f"{v:+.3f}", ha="center", va="bottom", fontsize=9)
    ax1.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax1.set_axisbelow(True)

    ax2.bar(x, mape_gap, **bar_kw)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in N_TRAIN])
    ax2.set_xlabel("Train Windows")
    ax2.set_ylabel("MAPE Reduction (pp)")
    ax2.set_title("MAPE Advantage of CWPDDA over LSTM")
    for i, v in enumerate(mape_gap):
        ax2.text(i, v + 0.1, f"{v:+.2f}pp", ha="center", va="bottom", fontsize=9)
    ax2.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    save("cwpdda_advantage.png")


advantage_plot()


# ─── Architecture diagram ────────────────────────────────────────────────────

def architecture_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(ax, bx, by, bw, bh, label, color="#4C72B0", fontsize=9,
            text_color="white", style="round,pad=0.1", lw=1.2):
        patch = FancyBboxPatch((bx - bw / 2, by - bh / 2), bw, bh,
                               boxstyle=style,
                               facecolor=color, edgecolor="white",
                               linewidth=lw, zorder=3)
        ax.add_patch(patch)
        ax.text(bx, by, label, ha="center", va="center",
                fontsize=fontsize, color=text_color,
                fontweight="bold", zorder=4, multialignment="center")

    def arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.5, mutation_scale=12):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw, mutation_scale=mutation_scale),
                    zorder=2)

    def lbl(ax, lx, ly, text, fontsize=8.5, color="#333333", ha="center"):
        ax.text(lx, ly, text, ha=ha, va="center",
                fontsize=fontsize, color=color, zorder=5)

    # ── Inputs ────────────────────────────────────────────────────────────────
    box(ax, 2.5, 6.3, 2.2, 0.6,
        "Source Windows\n(Google Cluster, W=24)",
        color="#2ca02c", fontsize=8.5)
    box(ax, 10.5, 6.3, 2.2, 0.6,
        "Target Windows\n(Alibaba Container, W=24)",
        color="#9467bd", fontsize=8.5)

    ax.text(2.5, 5.78, "NN-matched from 125k-window bank",
            ha="center", va="top", fontsize=7.5, color="#2ca02c", style="italic")

    # ── Projections ───────────────────────────────────────────────────────────
    box(ax, 2.5, 5.0, 2.0, 0.55, "Linear Projection\n(1 \u2192 d=64)", color="#1f77b4")
    box(ax, 10.5, 5.0, 2.0, 0.55, "Linear Projection\n(1 \u2192 d=64)", color="#1f77b4")

    arrow(ax, 2.5, 5.97, 2.5, 5.28)
    arrow(ax, 10.5, 5.97, 10.5, 5.28)

    ax.annotate("shared weights", xy=(2.0, 5.0), xytext=(0.3, 5.0),
                fontsize=7.5, color="#1f77b4", va="center",
                arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=0.9))

    # ── Self-attention ────────────────────────────────────────────────────────
    box(ax, 2.5, 3.9, 2.0, 0.55, "Self-Attention\n(shared weights)", color="#1f77b4")
    box(ax, 10.5, 3.9, 2.0, 0.55, "Self-Attention\n(shared weights)", color="#1f77b4")

    arrow(ax, 2.5, 4.72, 2.5, 4.18)
    arrow(ax, 10.5, 4.72, 10.5, 4.18)

    # ── Private features ──────────────────────────────────────────────────────
    box(ax, 2.5, 2.85, 2.0, 0.55,
        "z_src_private\n(pooled, d=64)", color="#17becf", fontsize=8.5)
    box(ax, 10.5, 2.85, 2.0, 0.55,
        "z_tgt_private\n(pooled, d=64)", color="#17becf", fontsize=8.5)

    arrow(ax, 2.5, 3.62, 2.5, 3.13)
    arrow(ax, 10.5, 3.62, 10.5, 3.13)

    # ── Cross-attention ───────────────────────────────────────────────────────
    box(ax, 6.5, 3.9, 2.8, 0.65,
        "Cross-Attention\nQ = Target,  K = V = Source\nz_shared  (B, W, d)",
        color="#ff7f0e", fontsize=8)

    arrow(ax, 3.5, 3.9, 5.1, 3.9, color="#ff7f0e")
    arrow(ax, 9.5, 3.9, 7.9, 3.9, color="#ff7f0e")
    lbl(ax, 4.3,  4.12, "K, V", fontsize=8, color="#ff7f0e")
    lbl(ax, 8.7,  4.12, "Q",    fontsize=8, color="#ff7f0e")

    # ── GRL discriminator ────────────────────────────────────────────────────
    box(ax, 6.5, 2.1, 2.4, 0.55,
        "GRL Domain Discriminator\n(source=0 vs target=1)",
        color="#8c564b", fontsize=8)

    arrow(ax, 3.5, 2.85, 5.25, 2.28, color="#8c564b")
    arrow(ax, 9.5, 2.85, 7.75, 2.28, color="#8c564b")
    lbl(ax, 4.15, 2.72, "z_src", fontsize=8, color="#8c564b")
    lbl(ax, 8.85, 2.72, "z_tgt", fontsize=8, color="#8c564b")
    lbl(ax, 6.5,  1.68, "L_d  =  BCE  +  GRL gradient reversal",
        fontsize=8, color="#8c564b")

    # ── LSTM predictor ────────────────────────────────────────────────────────
    box(ax, 6.5, 2.95, 2.2, 0.55,
        "LSTM Predictor\n(2 layers, 40 units)",
        color="#e377c2", fontsize=8.5)

    arrow(ax, 6.5, 3.57, 6.5, 3.23)

    # ── Output ────────────────────────────────────────────────────────────────
    box(ax, 6.5, 1.2, 1.8, 0.5,
        "Prediction  y_hat\n(horizon = 1 step)",
        color="#bcbd22", fontsize=8.5, text_color="black")

    arrow(ax, 6.5, 2.67, 6.5, 1.45)

    lbl(ax, 6.5, 0.72,
        "L_y  =  MSE(y_hat,  y_target)",
        fontsize=8.5, color="#888800")

    # MMD loss annotation
    lbl(ax, 11.6, 2.1,
        "L_f  =  -MMD\n(private vs shared)",
        fontsize=7.5, color="#17becf")
    arrow(ax, 10.5, 2.57, 11.2, 2.3, color="#17becf", lw=0.9)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(6.5, 6.82,
            "CWPDDA Architecture  —  Cross-domain Workload Prediction via Domain Adversarial Adaptation",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color="#222222")

    plt.tight_layout()
    save("architecture.png")


architecture_diagram()

print("\nAll figures saved to ./figures/")
