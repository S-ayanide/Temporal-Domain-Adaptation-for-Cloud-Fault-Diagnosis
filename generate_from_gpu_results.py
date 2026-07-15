"""
generate_from_gpu_results.py
============================
Reads the all_results.json files produced by run_gpu.sh and generates
the four presentation figures:

  figD_regression_comparison.png   — MAE / MAPE / RMSE, both directions
  figE_classification_comparison.png — Acc / Precision / Recall / F1 / MCC / G-Mean
  figF_heatmap_g2a.png             — heatmap: methods × all metrics (G→A)
  figG_heatmap_a2g.png             — heatmap: methods × all metrics (A→G)

Run after scp-ing results from the GPU server:
    scp -r user@gpu-server:~/dissertation/results/g2a results/
    scp -r user@gpu-server:~/dissertation/results/a2g results/
    python generate_from_gpu_results.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = os.path.dirname(__file__)
G2A  = os.path.join(BASE, "results", "g2a", "all_results.json")
A2G  = os.path.join(BASE, "results", "a2g", "all_results.json")
OUT  = os.path.join(BASE, "results", "presentation_figures")
os.makedirs(OUT, exist_ok=True)

# ── Colour per method ──────────────────────────────────────────────────────────
COLORS = {
    "ARIMA":     "#888888",
    "LSTM":      "#4878CF",
    "GRU":       "#77BEDB",
    "CNN-LSTM":  "#F7A35C",
    "N-BEATS":   "#6ACC65",
    "DeepJDOT":  "#D65F5F",
    "CWPDDA":    "#B47CC7",
    "MC-CWPDDA": "#FF7F0E",
    "MCTL":      "#2CA02C",
    "Autoformer":"#90ED7D",
    "BHT-ARIMA": "#E4D354",
    "TS2Vec":    "#8085E9",
    "WANN":      "#F15C80",
}

LABEL_FONT = 11
TITLE_FONT = 12
TICK_FONT  = 9

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.grid":       True,
    "grid.linestyle":  "--",
    "grid.alpha":      0.35,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def _load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nResult file not found: {path}\n"
            "Run run_gpu.sh on the server first, then scp results back."
        )
    return json.load(open(path))


def _get(res, method, section, key, fallback=float("nan")):
    entry = res.get(method, {})
    if "error" in entry:
        return fallback
    return entry.get(section, {}).get(key, fallback)


# ── Figure D — Regression comparison (both directions, grouped by metric) ─────
def figD_regression(g2a, a2g):
    methods = [m for m in COLORS if m in g2a or m in a2g]
    metrics = [("MAE", "MAE"), ("MAPE_%", "MAPE (%)"), ("RMSE", "RMSE")]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(
        "Regression Metrics — All Transfer Methods\n"
        "(lower is better — dashed line = LSTM baseline)",
        fontsize=TITLE_FONT, fontweight="bold"
    )

    def _panel(ax, results, direction_label):
        avail = [m for m in methods if m in results and "error" not in results[m]]
        n_methods = len(avail)
        n_metrics = len(metrics)
        bar_w = 0.7 / n_methods
        x_groups = np.arange(n_metrics)

        for mi, (algo) in enumerate(avail):
            offset = (mi - n_methods / 2 + 0.5) * bar_w
            vals = [_get(results, algo, "regression", k) for k, _ in metrics]
            ax.bar(x_groups + offset, vals,
                   width=bar_w * 0.88,
                   color=COLORS.get(algo, "#AAAAAA"),
                   alpha=0.88, edgecolor="white", linewidth=0.5,
                   label=algo, zorder=3)

        # LSTM reference line per metric
        for gi, (k, _) in enumerate(metrics):
            lstm_val = _get(results, "LSTM", "regression", k)
            if not np.isnan(lstm_val):
                ax.hlines(lstm_val, gi - 0.42, gi + 0.42,
                          colors=COLORS["LSTM"], linewidths=1.8,
                          linestyles="--", alpha=0.75, zorder=4)

        ax.set_xticks(x_groups)
        ax.set_xticklabels([lab for _, lab in metrics], fontsize=TICK_FONT)
        ax.set_ylabel("Error (lower = better)", fontsize=LABEL_FONT)
        ax.set_xlabel("Metric", fontsize=LABEL_FONT)
        ax.set_title(f"({('a' if 'Google' in direction_label else 'b')})  {direction_label}",
                     fontsize=TITLE_FONT, fontweight="bold", pad=8)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.9)

    _panel(axes[0], g2a, "Google → Alibaba")
    _panel(axes[1], a2g, "Alibaba → Google")

    plt.tight_layout()
    out = os.path.join(OUT, "figD_regression_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure E — Classification metrics (both directions) ───────────────────────
def figE_classification(g2a, a2g):
    methods = [m for m in COLORS if m in g2a or m in a2g]
    clf_metrics = ["Accuracy", "Precision", "Recall", "F1", "MCC", "G-Mean"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(
        "Classification Metrics — All Transfer Methods\n"
        "(threshold = 70th percentile of target CPU — top 30% = high-load class)",
        fontsize=TITLE_FONT, fontweight="bold"
    )

    def _panel(ax, results, direction_label):
        avail = [m for m in methods if m in results and "error" not in results[m]
                 and "classification" in results[m]]
        n_methods = len(avail)
        n_metrics = len(clf_metrics)
        bar_w = 0.7 / n_methods
        x_groups = np.arange(n_metrics)

        for mi, algo in enumerate(avail):
            offset = (mi - n_methods / 2 + 0.5) * bar_w
            vals = [_get(results, algo, "classification", k) for k in clf_metrics]
            ax.bar(x_groups + offset, vals,
                   width=bar_w * 0.88,
                   color=COLORS.get(algo, "#AAAAAA"),
                   alpha=0.88, edgecolor="white", linewidth=0.5,
                   label=algo, zorder=3)

        # LSTM reference lines
        for gi, k in enumerate(clf_metrics):
            lstm_val = _get(results, "LSTM", "classification", k)
            if not np.isnan(lstm_val):
                ax.hlines(lstm_val, gi - 0.42, gi + 0.42,
                          colors=COLORS["LSTM"], linewidths=1.8,
                          linestyles="--", alpha=0.75, zorder=4)

        ax.set_xticks(x_groups)
        ax.set_xticklabels(clf_metrics, fontsize=TICK_FONT, rotation=15, ha="right")
        ax.set_ylabel("Metric value (higher = better)", fontsize=LABEL_FONT)
        ax.set_xlabel("Metric", fontsize=LABEL_FONT)
        ax.set_title(f"({'a' if 'Google' in direction_label else 'b'})  {direction_label}",
                     fontsize=TITLE_FONT, fontweight="bold", pad=8)
        ax.set_ylim(0, 1.08)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7.5, loc="lower right", ncol=2, framealpha=0.9)

    _panel(axes[0], g2a, "Google → Alibaba")
    _panel(axes[1], a2g, "Alibaba → Google")

    plt.tight_layout()
    out = os.path.join(OUT, "figE_classification_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figures F & G — Heatmaps (methods × all metrics) ─────────────────────────
def _heatmap(results, direction_label, filename):
    methods = [m for m in COLORS
               if m in results and "error" not in results[m]
               and "classification" in results[m]]

    # All metrics in one heatmap: regression + classification
    reg_keys = [("MAE", "MAE"), ("MAPE_%", "MAPE%"), ("RMSE", "RMSE")]
    clf_keys = [("Accuracy","Acc"), ("Precision","Prec"), ("Recall","Rec"),
                ("F1","F1"), ("MCC","MCC"), ("G-Mean","G-Mean")]

    col_labels = [lab for _, lab in reg_keys] + [lab for _, lab in clf_keys]
    col_sections = (["regression"]*3) + (["classification"]*6)
    col_raw_keys = [k for k, _ in reg_keys] + [k for k, _ in clf_keys]
    # For regression: lower=better → invert; for classification: higher=better → keep
    col_invert   = [True]*3 + [False]*6

    mat_vals = np.array([
        [_get(results, m, sec, k) for sec, k in zip(col_sections, col_raw_keys)]
        for m in methods
    ])

    # Normalise each column for colour (good = dark)
    mat_norm = np.zeros_like(mat_vals)
    for j in range(mat_vals.shape[1]):
        col = mat_vals[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            continue
        lo, hi = valid.min(), valid.max()
        normed = (col - lo) / (hi - lo + 1e-12)
        mat_norm[:, j] = 1 - normed if col_invert[j] else normed

    fig, ax = plt.subplots(figsize=(13, max(4, len(methods) * 0.55 + 1.5)))
    im = ax.imshow(mat_norm, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9.5, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=9.5)

    # Vertical divider between regression and classification blocks
    ax.axvline(2.5, color="white", linewidth=3)
    ax.text(1.0, -1.05, "Regression", ha="center", va="top",
            transform=ax.transData, fontsize=9, color="#444444", style="italic")
    ax.text(5.5, -1.05, "Classification", ha="center", va="top",
            transform=ax.transData, fontsize=9, color="#444444", style="italic")

    # Annotate cells
    for i in range(len(methods)):
        for j in range(len(col_labels)):
            v    = mat_vals[i, j]
            rank = mat_norm[i, j]
            fc   = "white" if rank > 0.55 else "black"
            txt  = f"{v:.2f}" if not np.isnan(v) else "—"
            if col_labels[j] in ("MAPE%",):
                txt = f"{v:.1f}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.5, color=fc, fontweight="bold")

    # Row separators
    for i in range(len(methods) - 1):
        ax.axhline(i + 0.5, color="white", linewidth=0.6)

    ax.grid(False)
    ax.set_xlabel("Metrics", fontsize=LABEL_FONT)
    ax.set_title(
        f"Algorithm Heatmap — {direction_label}\n"
        "Darker = better  (regression: lower error; classification: higher score)",
        fontsize=TITLE_FONT, fontweight="bold", pad=10
    )

    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Relative performance", fontsize=9)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["worst", "", "best"], fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading GPU results...")
    try:
        g2a = _load(G2A)
        a2g = _load(A2G)
    except FileNotFoundError as e:
        print(e)
        raise SystemExit(1)

    figD_regression(g2a, a2g)
    figE_classification(g2a, a2g)
    _heatmap(g2a, "Google → Alibaba", "figF_heatmap_g2a.png")
    _heatmap(a2g, "Alibaba → Google", "figG_heatmap_a2g.png")

    print(f"\nAll figures saved to: {OUT}")
    print("  figD → Regression:      MAE / MAPE / RMSE (both directions)")
    print("  figE → Classification:  Acc / Prec / Recall / F1 / MCC / G-Mean")
    print("  figF → Heatmap G→A:     all metrics in one view")
    print("  figG → Heatmap A→G:     all metrics in one view")
