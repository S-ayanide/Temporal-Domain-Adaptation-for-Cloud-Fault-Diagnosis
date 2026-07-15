"""
generate_domain_figures.py
===========================
Three publication-style figures for the dissertation:

  figA — Domain shift:  Source (Google, blue) vs Target (Alibaba, red) CPU time series
  figB — Heatmap:       Algorithms × metrics, coloured by relative rank (like reference fig)
  figC — Grouped bars:  Algorithms on x-axis, one bar per metric, two panels

Google 2019 (source) → Alibaba 2018 (target).
Output: results/presentation_figures/
"""

import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
RES  = os.path.join(BASE, "results")
OUT  = os.path.join(RES, "presentation_figures")
os.makedirs(OUT, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.alpha":         0.35,
    "grid.color":         "#BBBBBB",
})

C_SRC      = "#7EA6E0"   # periwinkle blue — source (Google)
C_TGT      = "#C0392B"   # crimson red     — target (Alibaba)

ALGO_COLOR = {
    "ARIMA":     "#888888",
    "LSTM":      "#4878CF",
    "N-BEATS":   "#6ACC65",
    "DeepJDOT":  "#D65F5F",
    "CWPDDA":    "#B47CC7",
    "MC-CWPDDA": "#C4AD66",
    "GRU":       "#77BEDB",
    "CNN-LSTM":  "#F7A35C",
    "Autoformer":"#90ED7D",
    "BHT-ARIMA": "#E4D354",
    "TS2Vec":    "#8085E9",
    "WANN":      "#F15C80",
    "MCTL":      "#2CA02C",
}

# ── load results ──────────────────────────────────────────────────────────────
def _j(f): return json.load(open(os.path.join(RES, f)))

nb  = _j("nbeats_results.json")
cw  = _j("cwpdda_results.json")
mc  = _j("mc_cwpdda_results.json")
mt  = _j("mctl_results.json")

# DeepJDOT (from known results)
dj = {
    "ARIMA":    {"MAE": 19.45, "MAPE_%": 119.38, "RMSE": 29.95},
    "LSTM":     {"MAE": 17.69, "MAPE_%": 119.50, "RMSE": 23.39},
    "DeepJDOT": {"MAE": 17.99, "MAPE_%": 118.76, "RMSE": 24.29},
}

# ── time-series helpers ───────────────────────────────────────────────────────
def _find_breaks(windows, n=2000, thr=0.99):
    breaks = []
    for i in range(1, min(n, len(windows))):
        p, q = windows[i-1, 5:], windows[i, :19]
        if np.std(p) < 1e-9 or np.std(q) < 1e-9:
            breaks.append(i); continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = float(np.corrcoef(p, q)[0, 1])
        if c < thr or np.isnan(c):
            breaks.append(i)
    return breaks

def _recon(windows, s, e):
    series = list(windows[s])
    for i in range(s + 1, e):
        series.extend(windows[i, 19:].tolist())
    return np.array(series, dtype=float)


# ═════════════════════════════════════════════════════════════════════════════
# Figure A — Domain shift time series
# ═════════════════════════════════════════════════════════════════════════════
def figA_domain_shift():
    data = np.load(os.path.join(RES, "preprocessed.npz"), allow_pickle=True)
    src_w = data["src_X"]
    tgt_w = data["tgt_train_X"]

    src_br = _find_breaks(src_w, 2000)
    tgt_br = _find_breaks(tgt_w, 500)

    # Collect long source series (≥150 points)
    src_series, prev = [], 0
    for b in src_br:
        if b - prev > 20:
            s = _recon(src_w, prev, b)
            if len(s) >= 150:
                src_series.append(s)
            if len(src_series) == 4:
                break
        prev = b

    # Collect target series (≥80 points)
    tgt_series, prev = [], 0
    for b in tgt_br:
        if b - prev >= 8:
            s = _recon(tgt_w, prev, b)
            if len(s) >= 80:
                tgt_series.append(s)
            if len(tgt_series) == 4:
                break
        prev = b

    fig, ax = plt.subplots(figsize=(12, 3.8))

    # Source lines — multiple overlapping, alpha layered
    alphas_src = [0.90, 0.72, 0.56, 0.42]
    for idx, s in enumerate(src_series):
        t = np.arange(len(s))
        ax.plot(t, s, color=C_SRC, linewidth=1.1,
                alpha=alphas_src[idx % len(alphas_src)],
                label="Source domain (Google)" if idx == 0 else "_nolegend_")

    # Target lines — shorter, concentrated at t=0 (overlapping with source start)
    alphas_tgt = [0.90, 0.75, 0.60, 0.48]
    for idx, s in enumerate(tgt_series):
        t = np.arange(len(s))
        ax.plot(t, s, color=C_TGT, linewidth=1.3,
                alpha=alphas_tgt[idx % len(alphas_tgt)],
                label="Target domain (Alibaba)" if idx == 0 else "_nolegend_")

    ax.set_xlabel("Time", fontsize=14, fontweight="bold")
    ax.set_ylabel("CPU resource\n(normalised)", fontsize=11)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    leg = ax.legend(fontsize=10.5, framealpha=0.95, loc="upper right",
                    edgecolor="#CCCCCC")

    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    plt.tight_layout()
    out = os.path.join(OUT, "figA_domain_shift.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure B — Heatmap:  algorithms (rows) × metrics (columns)
# Two panels: (a) raw-scale transfer methods  (b) MCTL normalised baselines
# ═════════════════════════════════════════════════════════════════════════════
def figB_heatmap():
    # ── panel (a) data ────────────────────────────────────────────────────────
    raw_algos   = ["ARIMA", "LSTM", "N-BEATS", "DeepJDOT", "CWPDDA", "MC-CWPDDA"]
    raw_metrics = ["MAE", "MAPE (%)", "RMSE"]

    def _raw(method, metric):
        if metric == "MAE":
            lut = {"ARIMA": nb["ARIMA"]["MAE"],   "LSTM": nb["LSTM"]["MAE"],
                   "N-BEATS": nb["N-BEATS"]["MAE"], "DeepJDOT": dj["DeepJDOT"]["MAE"],
                   "CWPDDA": cw["CWPDDA"]["MAE"],  "MC-CWPDDA": mc["MC-CWPDDA"]["MAE"]}
        elif metric == "MAPE (%)":
            lut = {"ARIMA": nb["ARIMA"]["MAPE_%"],   "LSTM": nb["LSTM"]["MAPE_%"],
                   "N-BEATS": nb["N-BEATS"]["MAPE_%"], "DeepJDOT": dj["DeepJDOT"]["MAPE_%"],
                   "CWPDDA": cw["CWPDDA"]["MAPE_%"],  "MC-CWPDDA": mc["MC-CWPDDA"]["MAPE_%"]}
        else:  # RMSE
            lut = {"ARIMA": nb["ARIMA"]["RMSE"],   "LSTM": nb["LSTM"]["RMSE"],
                   "N-BEATS": nb["N-BEATS"]["RMSE"], "DeepJDOT": dj["DeepJDOT"]["RMSE"],
                   "CWPDDA": cw["CWPDDA"]["RMSE"],  "MC-CWPDDA": mc["MC-CWPDDA"]["RMSE"]}
        return lut[method]

    mat_raw = np.array([[_raw(a, m) for m in raw_metrics] for a in raw_algos])

    # ── panel (b) data ────────────────────────────────────────────────────────
    mctl_algos   = ["ARIMA", "LSTM", "GRU", "CNN-LSTM", "Autoformer",
                    "BHT-ARIMA", "TS2Vec", "WANN", "MCTL"]
    mctl_metrics = ["MAE", "MSE"]
    mat_mctl = np.array([[mt[a][m] for m in mctl_metrics] for a in mctl_algos])

    # ── colour: lower error → darker  (invert per column) ────────────────────
    def _norm_inv(mat):
        out = np.zeros_like(mat)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            lo, hi = col.min(), col.max()
            out[:, j] = 1.0 - (col - lo) / (hi - lo + 1e-12)
        return out

    norm_raw  = _norm_inv(mat_raw)
    norm_mctl = _norm_inv(mat_mctl)

    # ── draw ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2),
                             gridspec_kw={"width_ratios": [1.0, 0.75]})

    def _panel(ax, mat_raw, mat_norm, algos, metrics, title, fmt_fn):
        im = ax.imshow(mat_norm, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

        # Axis ticks
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels(metrics, fontsize=10, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(algos)))
        ax.set_yticklabels(algos, fontsize=10)
        ax.set_xlabel("Metrics", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

        # Cell annotations (actual values)
        for i in range(len(algos)):
            for j in range(len(metrics)):
                val  = mat_raw[i, j]
                rank = mat_norm[i, j]
                fc   = "white" if rank > 0.55 else "black"
                ax.text(j, i, fmt_fn(val, j),
                        ha="center", va="center",
                        fontsize=9, color=fc, fontweight="bold")

        # Colour bar
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Performance\nMetric", fontsize=9)
        cb.set_ticks([0, 0.5, 1])
        cb.set_ticklabels(["worst", "", "best"], fontsize=8)

        # Separator lines between rows (light)
        for i in range(len(algos) - 1):
            ax.axhline(i + 0.5, color="white", linewidth=0.6)
        for j in range(len(metrics) - 1):
            ax.axvline(j + 0.5, color="white", linewidth=0.6)

        ax.grid(False)

    def _fmt_raw(v, col_idx):
        # MAE and RMSE: 2 dp; MAPE: 1 dp with %
        if col_idx == 1:   # MAPE
            return f"{v:.1f}"
        return f"{v:.2f}"

    def _fmt_mctl(v, col_idx):
        return f"{v:.4f}"

    _panel(axes[0], mat_raw,  norm_raw,  raw_algos,  raw_metrics,
           "(a)  Transfer Methods — raw CPU-unit scale", _fmt_raw)
    _panel(axes[1], mat_mctl, norm_mctl, mctl_algos, mctl_metrics,
           "(b)  MCTL Baselines — normalised 0–1 scale", _fmt_mctl)

    fig.suptitle(
        "Algorithm Performance Heatmap  |  Google 2019 → Alibaba 2018\n"
        "Darker = lower error = better",
        fontsize=13, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    out = os.path.join(OUT, "figB_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure C — Grouped bar chart:  metrics on x-axis, one bar per algorithm
# Two panels: (a) raw-scale methods  (b) MCTL normalised baselines
# ═════════════════════════════════════════════════════════════════════════════
def figC_grouped_bars():
    """
    Metrics are the x-axis groups; algorithms are the coloured bars within each group.
    Raw values on y-axis. LSTM dashed reference line per metric group.
    Two panels: (a) raw-scale transfer methods  (b) MCTL normalised baselines.
    """

    # ── panel (a): raw-scale ──────────────────────────────────────────────────
    raw_algos   = ["ARIMA", "LSTM", "N-BEATS", "DeepJDOT", "CWPDDA", "MC-CWPDDA"]
    raw_metrics = ["MAE", "MAPE (%)", "RMSE"]

    def _rv(algo, metric):
        if metric == "MAE":
            lut = {"ARIMA": nb["ARIMA"]["MAE"],    "LSTM": nb["LSTM"]["MAE"],
                   "N-BEATS": nb["N-BEATS"]["MAE"], "DeepJDOT": dj["DeepJDOT"]["MAE"],
                   "CWPDDA": cw["CWPDDA"]["MAE"],  "MC-CWPDDA": mc["MC-CWPDDA"]["MAE"]}
        elif metric == "MAPE (%)":
            lut = {"ARIMA": nb["ARIMA"]["MAPE_%"],    "LSTM": nb["LSTM"]["MAPE_%"],
                   "N-BEATS": nb["N-BEATS"]["MAPE_%"], "DeepJDOT": dj["DeepJDOT"]["MAPE_%"],
                   "CWPDDA": cw["CWPDDA"]["MAPE_%"],  "MC-CWPDDA": mc["MC-CWPDDA"]["MAPE_%"]}
        else:
            lut = {"ARIMA": nb["ARIMA"]["RMSE"],    "LSTM": nb["LSTM"]["RMSE"],
                   "N-BEATS": nb["N-BEATS"]["RMSE"], "DeepJDOT": dj["DeepJDOT"]["RMSE"],
                   "CWPDDA": cw["CWPDDA"]["RMSE"],  "MC-CWPDDA": mc["MC-CWPDDA"]["RMSE"]}
        return lut[algo]

    # ── panel (b): MCTL ───────────────────────────────────────────────────────
    mctl_algos   = ["ARIMA", "LSTM", "GRU", "CNN-LSTM", "Autoformer",
                    "BHT-ARIMA", "TS2Vec", "WANN", "MCTL"]
    mctl_metrics = ["MAE", "MSE"]

    def _mv(algo, metric):
        return mt[algo][metric]

    # ── draw helper ───────────────────────────────────────────────────────────
    def _panel(ax, algos, metrics, val_fn, title, ylabel):
        n_algos   = len(algos)
        n_metrics = len(metrics)
        bar_w     = 0.75 / n_algos
        x_groups  = np.arange(n_metrics)

        for ai, algo in enumerate(algos):
            offset = (ai - n_algos / 2 + 0.5) * bar_w
            vals   = [val_fn(algo, m) for m in metrics]
            ax.bar(x_groups + offset, vals,
                   width=bar_w * 0.88,
                   color=ALGO_COLOR[algo],
                   alpha=0.88,
                   edgecolor="white",
                   linewidth=0.5,
                   label=algo,
                   zorder=3)

        # LSTM dashed reference line per metric group
        for mi, m in enumerate(metrics):
            lstm_val = val_fn("LSTM", m)
            ax.hlines(lstm_val,
                      mi - 0.42, mi + 0.42,
                      colors=ALGO_COLOR["LSTM"], linewidths=1.8,
                      linestyles="--", alpha=0.75, zorder=4)

        ax.set_xticks(x_groups)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel("Metric", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8.5, loc="upper right", ncol=2,
                  framealpha=0.9, labelspacing=0.3)

    fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))
    fig.suptitle(
        "Evaluation Metric Comparison Across Algorithms  |  Google 2019 → Alibaba 2018\n"
        "(lower is better — dashed line = LSTM in-domain baseline)",
        fontsize=13, fontweight="bold"
    )

    _panel(axes[0], raw_algos, raw_metrics, _rv,
           "(a)  Transfer Methods  (raw CPU-unit scale)",
           "Error  (CPU utilisation units)")

    _panel(axes[1], mctl_algos, mctl_metrics, _mv,
           "(b)  MCTL Baselines  (normalised 0–1 scale)",
           "Error  (CPU fraction, 0–1 normalised)")

    plt.tight_layout()
    out = os.path.join(OUT, "figC_grouped_bars.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating domain-shift and comparison figures...")
    figA_domain_shift()
    figB_heatmap()
    figC_grouped_bars()
    print(f"\nAll saved to: {OUT}")
    print("  figA → Source (Google, blue) vs Target (Alibaba, red) time series")
    print("  figB → Heatmap: algorithms × metrics, darker = better")
    print("  figC → Grouped bars: algorithms × metrics (performance score 0–1)")
