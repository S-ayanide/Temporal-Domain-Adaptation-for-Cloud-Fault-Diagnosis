"""
Step 4: Heterogeneous Nodes Experiment (Figure 4)
Evaluates per node type: cpu_heavy, mem_heavy, io_heavy, mixed.
TA-DATL should generalise better across node types because its
temporal encoder captures type-specific resource dynamics.
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
from trainer import train_adversarial, train_ta_datl, evaluate


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
    handlers=[logging.FileHandler(LOG_DIR/"04_heterogeneous.log"),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

NODE_TYPES  = ["cpu_heavy","mem_heavy","io_heavy","mixed"]
NODE_LABELS = {"cpu_heavy":"CPU-Intensive","mem_heavy":"Memory-Intensive",
               "io_heavy":"I/O-Bound","mixed":"Mixed-Load"}
EPOCHS = 150; LR = 1e-3


def load():
    with open(PROC_DIR/"meta.json") as f: meta = json.load(f)
    src_t = np.load(PROC_DIR/"source_temporal.npz")
    tgt_t = np.load(PROC_DIR/"target_temporal.npz")
    X_st = src_t["X"].astype(np.float32); y_st = src_t["y"].astype(np.int64)
    X_tt = tgt_t["X"].astype(np.float32); y_tt = tgt_t["y"].astype(np.int64)
    lb_t = tgt_t["labeled"].astype(bool)
    nt_t = tgt_t["node_type"]
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
    return (X_sf,y_sf,X_tf,y_tf,lb_f, X_st,y_st,X_tt,y_tt,lb_t,nt_t,
            int(meta["n_classes"]), X_sf.shape[1], F)


def run():
    logger.info("="*65)
    logger.info("  Experiment: Heterogeneous Nodes (Figure 4)")
    logger.info("="*65)
    (X_sf,y_sf,X_tf,y_tf,lb_f, X_st,y_st,X_tt,y_tt,lb_t,nt_t, n_cls,in_f,in_t) = load()

    # Train both models on the full target domain
    logger.info("\nTraining TA-DATL on full domain ...")
    ta = TA_DATL(in_t, n_cls, hidden_dim=120, n_groups=4)
    train_ta_datl(ta, X_st,y_st,X_tt,y_tt,lb_t, n_cls,EPOCHS,LR,128)

    logger.info("Training DATL on full domain ...")
    datl = DATL(in_f, n_cls)
    train_adversarial(datl,"DATL",X_sf,y_sf,X_tf,y_tf,lb_f, n_cls,EPOCHS,LR,256)

    logger.info("Training DANN on full domain ...")
    dann = DANN(in_f, n_cls)
    train_adversarial(dann,"DANN",X_sf,y_sf,X_tf,y_tf,lb_f, n_cls,EPOCHS,LR,256)

    # Evaluate per node type
    records = {"TA-DATL (Ours)":[], "DATL":[], "DANN":[]}
    for nt in NODE_TYPES:
        mask_t = nt_t == nt
        mask_f = mask_t   # same windows, same order
        if mask_t.sum() < 10:
            logger.warning(f"  {nt}: only {mask_t.sum()} samples — skip"); continue
        y_node = y_tt[mask_t]
        if len(np.unique(y_node)) < 2:
            logger.warning(f"  {nt}: single class — skip"); continue

        for name, model, X_node in [
            ("TA-DATL (Ours)", ta,   X_tt[mask_t]),
            ("DATL",           datl, X_tf[mask_f]),
            ("DANN",           dann, X_tf[mask_f]),
        ]:
            m = evaluate(model, X_node, y_node, n_cls)
            records[name].append({"node_type":nt, **m})
            logger.info(f"  {NODE_LABELS[nt]:<22} [{name}]  "
                        f"acc={m['accuracy']:.4f} f1={m['f1']:.4f}")

    with open(TAB_DIR/"heterogeneous_nodes.json","w") as f: json.dump(records,f,indent=2)

    # Plot
    node_labels_present = [NODE_LABELS[r["node_type"]] for r in records.get("TA-DATL (Ours)",[])
                           if r["node_type"] in NODE_LABELS]
    if not node_labels_present:
        logger.warning("No node-type data to plot."); return records

    colors  = {"TA-DATL (Ours)":"#F44336","DATL":"#9C27B0","DANN":"#9E9E9E"}
    bar_w   = 0.28
    x       = np.arange(len(node_labels_present))
    fig, axes = plt.subplots(1,3,figsize=(16,5))
    for ax, metric, title in zip(axes,["accuracy","f1","auc"],["Accuracy","F1-Score","AUC"]):
        for i,(name,recs) in enumerate(records.items()):
            vals = [r[metric] for r in recs]
            ax.bar(x + (i-1)*bar_w, vals, bar_w, label=name,
                   color=colors[name], alpha=0.85, edgecolor="white")
        ax.set_xlabel("Node Type"); ax.set_ylabel(title)
        ax.set_title(f"{title} across Node Types",fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(node_labels_present,rotation=15,ha="right")
        ax.set_ylim(0.5,1.05); ax.legend(fontsize=9); ax.grid(True,axis="y",alpha=0.3)
    plt.suptitle("TA-DATL on Heterogeneous Nodes",fontsize=14,fontweight="bold",y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR/"figure4_heterogeneous_nodes.png",dpi=150,bbox_inches="tight")
    plt.close()
    logger.info(f"\nFigure → {FIG_DIR}/figure4_heterogeneous_nodes.png")
    return records


if __name__ == "__main__":
    run()
