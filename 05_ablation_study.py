"""
Step 5: Ablation Study (novel — not in the replicated paper)
Systematically removes each TA-DATL component to quantify its contribution.

Variants:
  TA-DATL (Full)        — full model
  w/o Temporal Encoder  — replaced with flat MLP (= DATL)
  w/o Temporal MMD      — λ1 = 0
  w/o Calibrated PL     — temperature = 1.0 (standard softmax)
  w/o Adversarial Loss  — λ2 = 0

The result shows which components drive the improvement over DATL.
"""

import json, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from models import TA_DATL, DATL
from trainer import train_ta_datl, train_adversarial

PROC_DIR = BASE_DIR/"data"/"processed"
FIG_DIR  = BASE_DIR/"results"/"figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR  = BASE_DIR/"results"/"tables";  TAB_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR  = BASE_DIR/"logs";              LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_DIR/"05_ablation.log"),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

EPOCHS = 150; LR = 1e-3


def load():
    with open(PROC_DIR/"meta.json") as f: meta = json.load(f)
    src_t = np.load(PROC_DIR/"source_temporal.npz")
    tgt_t = np.load(PROC_DIR/"target_temporal.npz")
    X_st = src_t["X"].astype(np.float32); y_st = src_t["y"].astype(np.int64)
    X_tt = tgt_t["X"].astype(np.float32); y_tt = tgt_t["y"].astype(np.int64)
    lb_t = tgt_t["labeled"].astype(bool)
    N_s,W,F = X_st.shape
    sc = StandardScaler()
    X_st = sc.fit_transform(X_st.reshape(-1,F)).reshape(N_s,W,F).astype(np.float32)
    X_tt = sc.transform(X_tt.reshape(-1,F)).reshape(len(X_tt),W,F).astype(np.float32)
    src_f = pd.read_parquet(PROC_DIR/"source_flat.parquet")
    tgt_f = pd.read_parquet(PROC_DIR/"target_flat.parquet")
    fc  = [c for c in src_f.columns if c not in {"label","labeled"}]
    sc2 = StandardScaler()
    X_sf = sc2.fit_transform(src_f[fc]).astype(np.float32); y_sf = src_f["label"].values.astype(np.int64)
    X_tf = sc2.transform(tgt_f[fc]).astype(np.float32);     y_tf = tgt_f["label"].values.astype(np.int64)
    lb_f = tgt_f["labeled"].values.astype(bool)
    return (X_sf,y_sf,X_tf,y_tf,lb_f, X_st,y_st,X_tt,y_tt,lb_t,
            int(meta["n_classes"]), X_sf.shape[1], F)


def run():
    logger.info("="*65)
    logger.info("  Ablation Study")
    logger.info("="*65)
    (X_sf,y_sf,X_tf,y_tf,lb_f, X_st,y_st,X_tt,y_tt,lb_t, n_cls,in_f,in_t) = load()

    variants = [
        ("TA-DATL (Full)",
         lambda: TA_DATL(in_t,n_cls,hidden_dim=120,n_groups=4,
                         lambda1=0.1,lambda2=0.1,temperature=1.5),
         "temporal"),
        ("w/o Temporal Encoder (= DATL)",
         lambda: DATL(in_f,n_cls),
         "flat"),
        ("w/o Temporal MMD (λ₁=0)",
         lambda: TA_DATL(in_t,n_cls,hidden_dim=120,n_groups=4,
                         lambda1=0.0,lambda2=0.1,temperature=1.5),
         "temporal"),
        ("w/o Calibrated PL (T=1)",
         lambda: TA_DATL(in_t,n_cls,hidden_dim=120,n_groups=4,
                         lambda1=0.1,lambda2=0.1,temperature=1.0),
         "temporal"),
        ("w/o Adversarial Loss (λ₂=0)",
         lambda: TA_DATL(in_t,n_cls,hidden_dim=120,n_groups=4,
                         lambda1=0.1,lambda2=0.0,temperature=1.5),
         "temporal"),
    ]

    records = {}
    for name, build, mode in variants:
        logger.info(f"\n[{name}]")
        model = build()
        if mode == "flat":
            m = train_adversarial(model,"DATL",X_sf,y_sf,X_tf,y_tf,lb_f,n_cls,EPOCHS,LR,256)
        else:
            m = train_ta_datl(model,X_st,y_st,X_tt,y_tt,lb_t,n_cls,EPOCHS,LR,128)
        records[name] = m
        logger.info(f"  acc={m['accuracy']*100:.2f}  f1={m['f1']*100:.2f}  auc={m['auc']*100:.2f}")

    with open(TAB_DIR/"ablation.json","w") as f: json.dump(records,f,indent=2)

    # Print ablation table
    full = records["TA-DATL (Full)"]
    rows = [f"\n{'Variant':<40}  {'Acc':>6}  {'F1':>6}  {'AUC':>6}  {'ΔF1':>6}"]
    rows.append("-"*68)
    for name, m in records.items():
        delta = (m["f1"] - full["f1"]) * 100
        d_str = f"{delta:+.2f}" if name != "TA-DATL (Full)" else "  —  "
        rows.append(f"  {name:<38}  {m['accuracy']*100:>5.2f}  "
                    f"{m['f1']*100:>5.2f}  {m['auc']*100:>5.2f}  {d_str:>6}")
    rows.append("-"*68)
    table = "\n".join(rows)
    logger.info(table)
    with open(TAB_DIR/"ablation.txt","w") as f: f.write(table+"\n")

    # Plot
    labels   = list(records.keys())
    f1_vals  = [records[k]["f1"]*100 for k in labels]
    acc_vals = [records[k]["accuracy"]*100 for k in labels]
    colors   = ["#F44336"] + ["#90CAF9"]*4
    x        = np.arange(len(labels))
    fig, ax  = plt.subplots(figsize=(11,5))
    bars = ax.bar(x, f1_vals, color=colors, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("F1-Score (%)"); ax.set_title("Ablation Study — TA-DATL Component Contributions",
                                                  fontweight="bold")
    ax.set_ylim(max(0,min(f1_vals)-5), min(100,max(f1_vals)+8))
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(full["f1"]*100, color="#F44336", linestyle="--", alpha=0.5, label="Full model")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR/"figure5_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"\nFigure → {FIG_DIR}/figure5_ablation.png")
    return records


if __name__ == "__main__":
    run()
