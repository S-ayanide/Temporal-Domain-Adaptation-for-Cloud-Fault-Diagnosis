"""
Step 2: Label Scarcity Experiment (Figure 2)
Compares TA-DATL vs DATL at varying proportions of labeled target data.
TA-DATL should show stronger robustness at low-label regimes due to
more stable temporal features and better-calibrated pseudo-labels.
"""

import argparse
import json, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from models import DATL, TA_DATL, DANN
from trainer import train_adversarial, train_ta_datl


def _proc_dir():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", type=str, default=None)
    a, _ = ap.parse_known_args()
    if not a.processed_dir:
        return (BASE_DIR / "data" / "processed").resolve()
    p = Path(a.processed_dir)
    return p.resolve() if p.is_absolute() else (BASE_DIR / p).resolve()


PROC_DIR = _proc_dir()
FIG_DIR  = BASE_DIR/"results"/"figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR  = BASE_DIR/"results"/"tables";  TAB_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR  = BASE_DIR/"logs";              LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_DIR/"02_label_scarcity.log"),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

EPOCHS  = 150
LR      = 1e-3
RATIOS  = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]


def load():
    with open(PROC_DIR/"meta.json") as f: meta = json.load(f)
    src_t = np.load(PROC_DIR/"source_temporal.npz")
    tgt_t = np.load(PROC_DIR/"target_temporal.npz")
    X_st = src_t["X"].astype(np.float32); y_st = src_t["y"].astype(np.int64)
    X_tt = tgt_t["X"].astype(np.float32); y_tt = tgt_t["y"].astype(np.int64)
    N_s,W,F = X_st.shape
    sc = StandardScaler()
    X_st = sc.fit_transform(X_st.reshape(-1,F)).reshape(N_s,W,F).astype(np.float32)
    X_tt = sc.transform(X_tt.reshape(-1,F)).reshape(len(X_tt),W,F).astype(np.float32)

    src_f = pd.read_parquet(PROC_DIR/"source_flat.parquet")
    tgt_f = pd.read_parquet(PROC_DIR/"target_flat.parquet")
    fc = [c for c in src_f.columns if c not in {"label","labeled"}]
    X_sf = StandardScaler().fit(src_f[fc]).transform(src_f[fc]).astype(np.float32)
    y_sf = src_f["label"].values.astype(np.int64)
    X_tf = StandardScaler().fit(src_f[fc]).transform(tgt_f[fc]).astype(np.float32)
    y_tf = tgt_f["label"].values.astype(np.int64)
    return X_sf,y_sf,X_tf,y_tf, X_st,y_st,X_tt,y_tt, int(meta["n_classes"]), F


def run():
    logger.info("="*65)
    logger.info("  Experiment: Label Scarcity (Figure 2)")
    logger.info("="*65)
    X_sf,y_sf,X_tf,y_tf, X_st,y_st,X_tt,y_tt, n_cls, in_t = load()
    in_f = X_sf.shape[1]
    rng  = np.random.default_rng(42)
    records = {"TA-DATL (Ours)":[], "DATL":[], "DANN":[]}

    for ratio in RATIOS:
        n_lb = max(int(len(X_tt)*ratio), n_cls*2)
        idx  = rng.choice(len(X_tt), n_lb, replace=False)
        lb_t = np.zeros(len(X_tt), dtype=bool); lb_t[idx] = True
        lb_f = lb_t.copy()
        logger.info(f"\nRatio={ratio:.0%}  labeled={lb_t.sum()}/{len(lb_t)}")

        # TA-DATL
        ta = TA_DATL(in_t, n_cls, hidden_dim=120, n_groups=4)
        m  = train_ta_datl(ta, X_st, y_st, X_tt, y_tt, lb_t, n_cls, EPOCHS, LR, 128)
        records["TA-DATL (Ours)"].append({"ratio":ratio, **m})
        logger.info(f"  TA-DATL  acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")

        # DATL (flat)
        datl = DATL(in_f, n_cls)
        m    = train_adversarial(datl,"DATL",X_sf,y_sf,X_tf,y_tf,lb_f,n_cls,EPOCHS,LR,256)
        records["DATL"].append({"ratio":ratio, **m})
        logger.info(f"  DATL     acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")

        # DANN (flat)
        dann = DANN(in_f, n_cls)
        m    = train_adversarial(dann,"DANN",X_sf,y_sf,X_tf,y_tf,lb_f,n_cls,EPOCHS,LR,256)
        records["DANN"].append({"ratio":ratio, **m})
        logger.info(f"  DANN     acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")

    with open(TAB_DIR/"label_scarcity.json","w") as f: json.dump(records,f,indent=2)

    # Plot
    colors  = {"TA-DATL (Ours)":"#F44336","DATL":"#2196F3","DANN":"#9E9E9E"}
    markers = {"TA-DATL (Ours)":"*","DATL":"o","DANN":"s"}
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    for ax, metric, title in zip(axes,["accuracy","f1","auc"],["Accuracy","F1-Score","AUC"]):
        for name, recs in records.items():
            xs = [r["ratio"] for r in recs]; ys = [r[metric] for r in recs]
            ax.plot(xs, ys, marker=markers[name], color=colors[name],
                    label=name, linewidth=2, markersize=7)
        ax.set_xlabel("Labeled target proportion"); ax.set_ylabel(title)
        ax.set_title(f"{title} vs Label Availability", fontweight="bold")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0%}"))
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle("TA-DATL vs Baselines under Label Scarcity",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR/"figure2_label_scarcity.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"\nFigure saved → {FIG_DIR}/figure2_label_scarcity.png")
    return records


if __name__ == "__main__":
    run()
