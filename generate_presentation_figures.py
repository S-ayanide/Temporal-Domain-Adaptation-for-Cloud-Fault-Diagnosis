"""
generate_presentation_figures.py
=================================
Generates all comparison figures needed for the dissertation presentation.

Google (source) → Alibaba (target), LSTM as baseline.

Outputs go to: results/presentation_figures/
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.switch_backend("Agg")

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "presentation_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_BASELINE  = "#4C72B0"   # blue  — LSTM baseline
C_ARIMA     = "#808080"   # grey  — ARIMA (non-transfer reference)
C_METHOD    = "#DD4444"   # red   — transfer method
C_MCTL      = "#2CA02C"   # green — MCTL (different scale, keep distinct)
C_HIGHLIGHT = "#FF7F0E"   # orange — MC-CWPDDA (QRS 2026)

LABEL_FONT  = 11
TITLE_FONT  = 13
TICK_FONT   = 10

# ─────────────────────────────────────────────────────────────────────────────
# Raw result numbers  (Google → Alibaba, raw CPU scale unless noted)
# ─────────────────────────────────────────────────────────────────────────────

# N-BEATS  (raw CPU scale, Google source 2019 → Alibaba 2018 target)
nbeats = {
    "ARIMA":   {"MAE": 7.294, "MAPE": 19.65, "RMSE": 10.51},
    "LSTM":    {"MAE": 6.704, "MAPE": 18.47, "RMSE":  9.21},
    "N-BEATS": {"MAE": 6.711, "MAPE": 19.35, "RMSE":  9.50},
}

# DeepJDOT  (raw CPU scale)
deepjdot = {
    "ARIMA":    {"MAE": 19.45, "MAPE": 119.38, "RMSE": 29.95},
    "LSTM":     {"MAE": 17.69, "MAPE": 119.50, "RMSE": 23.39},
    "DeepJDOT": {"MAE": 17.99, "MAPE": 118.76, "RMSE": 24.29},
}

# CWPDDA  (raw CPU scale)
cwpdda = {
    "ARIMA":  {"MAE": 19.35, "MAPE": 148.34, "RMSE": 29.35},
    "LSTM":   {"MAE": 16.79, "MAPE": 134.53, "RMSE": 22.16},
    "CWPDDA": {"MAE": 16.38, "MAPE": 126.07, "RMSE": 22.04},
}

# MC-CWPDDA  (raw CPU scale)
mc_cwpdda = {
    "ARIMA":     {"MAE": 7.40,  "MAPE": 19.86, "RMSE": 10.65},
    "LSTM":      {"MAE": 6.882, "MAPE": 18.31, "RMSE":  9.37},
    "MC-CWPDDA": {"MAE": 6.867, "MAPE": 17.87, "RMSE":  9.37},
}

# MCTL  (normalised 0-1 scale — CANNOT be combined with raw-scale methods)
mctl_norm = {
    "ARIMA":     {"MAE": 0.2097, "MSE": 0.0974},
    "LSTM":      {"MAE": 0.1788, "MSE": 0.0548},
    "GRU":       {"MAE": 0.1737, "MSE": 0.0549},
    "CNN-LSTM":  {"MAE": 0.1728, "MSE": 0.0547},
    "Autoformer":{"MAE": 0.1992, "MSE": 0.0636},
    "BHT-ARIMA": {"MAE": 0.1754, "MSE": 0.0571},
    "TS2Vec":    {"MAE": 0.1761, "MSE": 0.0558},
    "WANN":      {"MAE": 0.1792, "MSE": 0.0598},
    "MCTL":      {"MAE": 0.1729, "MSE": 0.0536},
}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Raw-scale methods: N-BEATS, DeepJDOT, CWPDDA, MC-CWPDDA
#             vs LSTM baseline  (MAE + MAPE side by side)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_raw_scale_comparison():
    """
    Grouped bar chart: for each method group show ARIMA / LSTM / Method.
    Two panels: MAE (left) and MAPE % (right).
    Transfer pair: Google 2019 (source) → Alibaba 2018 (target).
    LSTM is the baseline — highlighted with a dashed reference line per group.
    """
    groups = [
        ("N-BEATS\n(zero-shot)",        nbeats,    "N-BEATS",    C_METHOD),
        ("DeepJDOT\n(zero-shot DA)",    deepjdot,  "DeepJDOT",   C_METHOD),
        ("CWPDDA\n(adversarial)",       cwpdda,    "CWPDDA",     C_METHOD),
        ("MC-CWPDDA\n(adversarial +\ncontrastive)", mc_cwpdda, "MC-CWPDDA", C_HIGHLIGHT),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Google 2019 → Alibaba 2018  |  Transfer Methods vs LSTM Baseline\n"
        "(lower is better — LSTM = in-domain baseline trained on Alibaba)",
        fontsize=TITLE_FONT, fontweight="bold", y=1.02
    )

    metrics = [
        ("MAE",    "MAE  (CPU utilisation units)",       axes[0]),
        ("MAPE",   "MAPE  (%)",                          axes[1]),
    ]

    bar_w   = 0.22
    offsets = [-bar_w, 0, bar_w]
    labels  = ["ARIMA\n(no transfer)", "LSTM\n(baseline)", "Transfer\nMethod"]
    colors  = [C_ARIMA, C_BASELINE, None]   # method colour filled in per group

    for metric_key, ylabel, ax in metrics:
        x_positions = np.arange(len(groups))

        for gi, (group_label, data, method_name, method_color) in enumerate(groups):
            bar_colors = [C_ARIMA, C_BASELINE, method_color]
            vals = [
                data["ARIMA"][metric_key],
                data["LSTM"][metric_key],
                data[method_name][metric_key],
            ]
            for bi, (offset, val, bc, bl) in enumerate(zip(offsets, vals, bar_colors, labels)):
                bar = ax.bar(
                    gi + offset, val,
                    width=bar_w * 0.9,
                    color=bc,
                    alpha=0.88,
                    edgecolor="white",
                    linewidth=0.6,
                    label=bl if gi == 0 else "_nolegend_",
                    zorder=3,
                )
                # value label on top of bar
                ax.text(
                    gi + offset, val + 0.01 * ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else val + 0.5,
                    f"{val:.2f}" if metric_key == "MAE" else f"{val:.1f}%",
                    ha="center", va="bottom",
                    fontsize=8, color="black",
                )

            # dashed LSTM reference line across this group's bars
            lstm_val = data["LSTM"][metric_key]
            ax.hlines(
                lstm_val,
                gi - bar_w * 1.6, gi + bar_w * 1.6,
                colors=C_BASELINE, linewidths=1.2, linestyles="--", alpha=0.6, zorder=4
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([g[0] for g in groups], fontsize=TICK_FONT)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONT)
        ax.set_xlabel("Transfer Method  (Google 2019 → Alibaba 2018)", fontsize=LABEL_FONT)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        if gi == 0:  # legend only once
            handles = [
                mpatches.Patch(color=C_ARIMA,     label="ARIMA  (no transfer, no adaptation)"),
                mpatches.Patch(color=C_BASELINE,  label="LSTM  (in-domain baseline on Alibaba)"),
                mpatches.Patch(color=C_METHOD,    label="Transfer method"),
                mpatches.Patch(color=C_HIGHLIGHT, label="MC-CWPDDA  (accepted QRS 2026)"),
            ]
            ax.legend(handles=handles, fontsize=9, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig1_raw_scale_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — MCTL comparison (normalised scale, many baselines)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_mctl_normalised():
    """
    Horizontal bar chart of all 9 methods on normalised MAE and MSE.
    LSTM bar highlighted as baseline. MCTL bar highlighted as best.
    NOTE on slide: values are normalised to 0–1 scale (CPU fraction),
    not the raw 0–100 CPU units used in N-BEATS / CWPDDA / MC-CWPDDA charts.
    """
    methods = list(mctl_norm.keys())
    mae_vals = [mctl_norm[m]["MAE"] for m in methods]
    mse_vals = [mctl_norm[m]["MSE"] for m in methods]

    bar_colors_mae = []
    for m in methods:
        if m == "LSTM":
            bar_colors_mae.append(C_BASELINE)
        elif m == "MCTL":
            bar_colors_mae.append(C_MCTL)
        elif m == "ARIMA":
            bar_colors_mae.append(C_ARIMA)
        else:
            bar_colors_mae.append("#AAAAAA")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "MCTL vs All Baselines  |  Google 2019 → Alibaba 2018\n"
        "Note: values on normalised 0–1 CPU fraction scale (not raw CPU units)",
        fontsize=TITLE_FONT, fontweight="bold"
    )

    y = np.arange(len(methods))
    bar_h = 0.6

    for ax, vals, xlabel, title in [
        (axes[0], mae_vals, "MAE  (normalised CPU fraction, lower is better)", "Mean Absolute Error"),
        (axes[1], mse_vals, "MSE  (normalised CPU fraction², lower is better)", "Mean Squared Error"),
    ]:
        bars = ax.barh(y, vals, height=bar_h, color=bar_colors_mae,
                       edgecolor="white", linewidth=0.5, zorder=3)

        # value labels
        for bar, val in zip(bars, vals):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left", fontsize=8.5)

        # LSTM reference line
        lstm_val = mctl_norm["LSTM"][list(mctl_norm["LSTM"].keys())[0 if xlabel.startswith("MAE") else 1]]
        ax.axvline(lstm_val, color=C_BASELINE, linestyle="--",
                   linewidth=1.5, alpha=0.8, label="LSTM baseline", zorder=4)

        ax.set_yticks(y)
        ax.set_yticklabels(methods, fontsize=TICK_FONT)
        ax.set_xlabel(xlabel, fontsize=LABEL_FONT)
        ax.set_title(title, fontsize=LABEL_FONT, fontweight="bold")
        ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [
        mpatches.Patch(color=C_ARIMA,    label="ARIMA  (classical baseline)"),
        mpatches.Patch(color=C_BASELINE, label="LSTM   (in-domain baseline — reference)"),
        mpatches.Patch(color="#AAAAAA",  label="Other neural baselines"),
        mpatches.Patch(color=C_MCTL,     label="MCTL   (contrastive transfer — best)"),
    ]
    fig.legend(handles=handles, fontsize=9, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.06), framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig2_mctl_normalised.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Relative improvement over LSTM baseline (all methods, same axis)
# ─────────────────────────────────────────────────────────────────────────────
def fig3_relative_improvement():
    """
    Shows % change in MAE relative to LSTM baseline for all methods.
    Negative = better than LSTM. Positive = worse than LSTM.
    This is the ONLY chart where all methods can share one axis,
    because we normalise out the scale difference between MCTL and the rest.
    Tr-Predictor is excluded (no JSON results available).
    """
    # Compute % change vs LSTM baseline: (method - lstm) / lstm * 100
    method_results = {
        "N-BEATS\n(zero-shot)":              (nbeats["N-BEATS"]["MAE"],    nbeats["LSTM"]["MAE"]),
        "DeepJDOT\n(zero-shot DA)":          (deepjdot["DeepJDOT"]["MAE"], deepjdot["LSTM"]["MAE"]),
        "CWPDDA\n(adversarial)":             (cwpdda["CWPDDA"]["MAE"],     cwpdda["LSTM"]["MAE"]),
        "MCTL\n(contrastive)*":              (mctl_norm["MCTL"]["MAE"],    mctl_norm["LSTM"]["MAE"]),
        "MC-CWPDDA\n(adv. + contrastive)":   (mc_cwpdda["MC-CWPDDA"]["MAE"], mc_cwpdda["LSTM"]["MAE"]),
    }

    labels = list(method_results.keys())
    pct_changes = [
        (m_val - lstm_val) / lstm_val * 100
        for m_val, lstm_val in method_results.values()
    ]

    bar_colors = [C_METHOD, C_METHOD, C_METHOD, C_MCTL, C_HIGHLIGHT]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.suptitle(
        "All Transfer Methods vs LSTM Baseline  |  Google 2019 → Alibaba 2018\n"
        "% change in MAE relative to LSTM  (negative = better than baseline)",
        fontsize=TITLE_FONT, fontweight="bold"
    )

    x = np.arange(len(labels))
    bars = ax.bar(x, pct_changes, color=bar_colors,
                  edgecolor="white", linewidth=0.6, alpha=0.88,
                  width=0.55, zorder=3)

    # Zero line = LSTM baseline
    ax.axhline(0, color=C_BASELINE, linewidth=2.0, linestyle="-",
               label="LSTM baseline (0% = same as LSTM)", zorder=4)

    # Value labels
    for bar, val in zip(bars, pct_changes):
        va = "bottom" if val >= 0 else "top"
        offset = 0.3 if val >= 0 else -0.3
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                f"{val:+.1f}%", ha="center", va=va,
                fontsize=9.5, fontweight="bold", color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_FONT)
    ax.set_ylabel("% change in MAE vs LSTM baseline\n(negative = better than LSTM)", fontsize=LABEL_FONT)
    ax.set_xlabel("Transfer Method  (Google 2019 → Alibaba 2018)", fontsize=LABEL_FONT)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotation for MCTL scale note
    ax.annotate(
        "* MCTL evaluated on normalised\n  0–1 scale (relative comparison valid)",
        xy=(3, pct_changes[3]),
        xytext=(3.3, pct_changes[3] - 4),
        fontsize=8, color="#555555",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
    )

    handles = [
        mpatches.Patch(color=C_BASELINE,  label="LSTM  (in-domain baseline, 0% line)"),
        mpatches.Patch(color=C_METHOD,    label="Transfer methods"),
        mpatches.Patch(color=C_MCTL,      label="MCTL  (contrastive)"),
        mpatches.Patch(color=C_HIGHLIGHT, label="MC-CWPDDA  (accepted QRS 2026)"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig3_relative_improvement.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Database / features slide visual
# ─────────────────────────────────────────────────────────────────────────────
def fig4_dataset_features():
    """
    Table-style figure showing the three datasets (Google, Alibaba, Azure)
    with their features, labels, and availability — for the datasets slide.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    columns = ["", "Google\nGC2011 & GC2019", "Alibaba\nAC2018 & AC2020", "Azure\nV2 2019"]
    rows = [
        ["Role",          "Source domain",              "Target domain",               "Third provider\n(attempted)"],
        ["Workload type", "Container workloads\n(Borg scheduler)", "E-commerce containers\n+ batch jobs", "General VM workloads"],
        ["CPU feature",   "avg_cpu (0–100%)\nfrom instance usage JSON", "cpu_util_percent (0–100%)\nfrom machine_usage CSV", "avg_cpu (0–100%)\nfrom vm_cpu_readings CSV"],
        ["Memory",        "memory fraction\n(derived from usage dict)", "mem_util_percent\n+ mem_gps (bandwidth)", "NOT AVAILABLE\n→ set to constant 0.5"],
        ["Network / Disk","Proxy features\n(engineered — not native)", "net_in, net_out\ndisk_io_percent  (native)", "NOT AVAILABLE\n→ not in dataset"],
        ["Failure labels","From percentile thresholds\non CPU/memory/disk/net", "From percentile thresholds\non CPU/memory/disk/net", "NOT IN DATASET\n→ approximated from\nVM lifetime (proxy)"],
        ["# Features",    "6  (3 native, 3 proxy)",     "6  (all native)",             "1 native + 5 missing/proxy"],
        ["Label quality", "Consistent definition\nacross source",  "Consistent definition\nacross target", "Different definition —\nnot comparable to G/A labels"],
    ]

    col_colors = [
        ["#F0F0F0"] * len(columns),   # header placeholder
    ]
    row_colors = []
    cell_colors = []

    for i, row in enumerate(rows):
        bg = "#FFFFFF" if i % 2 == 0 else "#F7F7F7"
        cell_colors.append([bg] * len(columns))

    # Override specific cells
    azure_col = 3
    problem_rows = [3, 4, 5, 6, 7]  # rows where Azure has issues
    for ri in problem_rows:
        cell_colors[ri][azure_col] = "#FFE8E8"

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Style header
    for j in range(len(columns)):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=10)

    # Style row labels (first column)
    for i in range(1, len(rows) + 1):
        table[i, 0].set_facecolor("#E8EEF4")
        table[i, 0].set_text_props(fontweight="bold", fontsize=9)

    ax.set_title(
        "Dataset Comparison  |  Features, Labels, and Availability",
        fontsize=TITLE_FONT, fontweight="bold", pad=15
    )

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig4_dataset_features.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — MC-CWPDDA real results (replacing the simulated chart)
# ─────────────────────────────────────────────────────────────────────────────
def fig5_mc_cwpdda_real():
    """
    Bar chart with ARIMA / LSTM / CWPDDA / MC-CWPDDA side by side.
    Uses only real experimental results — no estimates.
    Two panels: MAE (left) and MAPE % (right).
    """
    methods_ordered = ["ARIMA\n(no transfer)", "LSTM\n(baseline)", "CWPDDA\n(adversarial)", "MC-CWPDDA\n(adv. + contrastive)"]
    mae_vals  = [mc_cwpdda["ARIMA"]["MAE"],  mc_cwpdda["LSTM"]["MAE"],  cwpdda["CWPDDA"]["MAE"],  mc_cwpdda["MC-CWPDDA"]["MAE"]]
    mape_vals = [mc_cwpdda["ARIMA"]["MAPE"], mc_cwpdda["LSTM"]["MAPE"], cwpdda["CWPDDA"]["MAPE"], mc_cwpdda["MC-CWPDDA"]["MAPE"]]
    bar_colors = [C_ARIMA, C_BASELINE, C_METHOD, C_HIGHLIGHT]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "MC-CWPDDA vs CWPDDA vs Baselines  |  Google 2019 → Alibaba 2018\n"
        "MC-CWPDDA accepted at QRS 2026 — novel combination of adversarial + contrastive adaptation",
        fontsize=TITLE_FONT, fontweight="bold"
    )

    for ax, vals, ylabel, fmt in [
        (axes[0], mae_vals,  "MAE  (CPU utilisation units, lower is better)", ".2f"),
        (axes[1], mape_vals, "MAPE  (%, lower is better)",                    ".1f%"),
    ]:
        x = np.arange(len(methods_ordered))
        bars = ax.bar(x, vals, color=bar_colors,
                      edgecolor="white", linewidth=0.6,
                      alpha=0.88, width=0.55, zorder=3)

        # LSTM reference line
        lstm_val = mc_cwpdda["LSTM"]["MAE"] if ylabel.startswith("MAE") else mc_cwpdda["LSTM"]["MAPE"]
        ax.axhline(lstm_val, color=C_BASELINE, linewidth=1.5,
                   linestyle="--", alpha=0.7, label="LSTM baseline", zorder=4)

        for bar, val in zip(bars, vals):
            label = f"{val:{fmt[:-1]}}" if not fmt.endswith("%") else f"{val:.1f}%"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.01 * max(vals),
                    label, ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(methods_ordered, fontsize=TICK_FONT)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONT)
        ax.set_xlabel("Method  (Google 2019 → Alibaba 2018)", fontsize=LABEL_FONT)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [
        mpatches.Patch(color=C_ARIMA,     label="ARIMA  (classical, no transfer)"),
        mpatches.Patch(color=C_BASELINE,  label="LSTM   (in-domain baseline on Alibaba)"),
        mpatches.Patch(color=C_METHOD,    label="CWPDDA  (adversarial alignment only)"),
        mpatches.Patch(color=C_HIGHLIGHT, label="MC-CWPDDA  (adversarial + contrastive — QRS 2026)"),
    ]
    fig.legend(handles=handles, fontsize=9, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.06), framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig5_mc_cwpdda_real.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating presentation figures...")
    fig1_raw_scale_comparison()
    fig2_mctl_normalised()
    fig3_relative_improvement()
    fig4_dataset_features()
    fig5_mc_cwpdda_real()
    print(f"\nAll figures saved to: {OUT_DIR}")
    print("\nSummary of what each figure is for:")
    print("  fig1 → Slide 4–8 overview: N-BEATS / DeepJDOT / CWPDDA / MC-CWPDDA vs LSTM")
    print("  fig2 → Slide 7 (MCTL): all 9 baselines on normalised scale")
    print("  fig3 → Slide 9 (core finding): all methods on one axis via % improvement")
    print("  fig4 → Slide 2 (databases): feature/label comparison table")
    print("  fig5 → Slide 9 (MC-CWPDDA): real results only, no simulated bars")
    print("\nNOTE: Tr-Predictor has no JSON results — use report/figures/tr_results.png directly")
    print("NOTE: Alibaba→Google results do not exist — would need re-running experiments")
