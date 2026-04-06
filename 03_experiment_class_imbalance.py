"""
Step 3: Class Imbalance Experiment (Figure 3)  — with checkpointing
"""

import argparse
import json, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

BASE_DIR  = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from models import DATL, TA_DATL, DANN, CDAN, FixBi, ToAlign
from trainer import train_adversarial, train_fixbi, train_ta_datl


def _proc_dir():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", type=str, default=None)
    a, _ = ap.parse_known_args()
    if not a.processed_dir:
        return (BASE_DIR / "data" / "processed").resolve()
    p = Path(a.processed_dir)
    return p.resolve() if p.is_absolute() else (BASE_DIR / p).resolve()


PROC_DIR = _proc_dir()
FIG_DIR   = BASE_DIR/"results"/"figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR   = BASE_DIR/"results"/"tables";  TAB_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR   = BASE_DIR/"logs";              LOG_DIR.mkdir(exist_ok=True)
CKPT_FILE = TAB_DIR/"class_imbalance_checkpoint.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_DIR/"03_class_imbalance.log"),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

EPOCHS = 150; LR = 1e-3
IMBALANCE_RATIOS = [1, 2, 5, 10, 20]


def load_checkpoint():
    if CKPT_FILE.exists():
        with open(CKPT_FILE) as f: ckpt = json.load(f)
        done = sum(len(v) for v in ckpt.values())
        logger.info(f"  Resumed: {done} combinations done.")
        return ckpt
    return {}

def save_checkpoint(r):
    with open(CKPT_FILE,"w") as f: json.dump(r,f,indent=2)

def is_done(r, name, ir):
    return any(x["imbalance_ratio"]==ir for x in r.get(name,[]))


def apply_imbalance(X, y, ir, rng):
    classes, counts = np.unique(y, return_counts=True)
    n_maj = counts.max(); n_min = max(1, n_maj//ir)
    maj   = classes[counts.argmax()]
    keep  = []
    for c in classes:
        idx = np.where(y==c)[0]
        keep.append(idx if c==maj else rng.choice(idx,min(len(idx),n_min),replace=False))
    k = np.concatenate(keep); rng.shuffle(k)
    return X[k], y[k]


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
    logger.info("  Experiment: Class Imbalance (Figure 3)  [checkpointed]")
    logger.info("="*65)
    (X_sf,y_sf,X_tf,y_tf,lb_f, X_st,y_st,X_tt,y_tt,lb_t, n_cls, in_f, in_t) = load()
    rng = np.random.default_rng(42)

    model_configs = {
        "DANN":          ("flat", lambda: DANN(in_f,n_cls)),
        "CDAN":          ("flat", lambda: CDAN(in_f,n_cls)),
        "FixBi":         ("flat", lambda: FixBi(in_f,n_cls)),
        "ToAlign":       ("flat", lambda: ToAlign(in_f,n_cls)),
        "DATL":          ("flat", lambda: DATL(in_f,n_cls)),
        "TA-DATL (Ours)":("temporal", lambda: TA_DATL(in_t,n_cls,hidden_dim=120,n_groups=4)),
    }

    records = load_checkpoint()
    for nm in model_configs: records.setdefault(nm,[])
    total = len(IMBALANCE_RATIOS)*len(model_configs)
    done  = sum(len(v) for v in records.values())
    logger.info(f"  Progress: {done}/{total}\n")

    for ir in IMBALANCE_RATIOS:
        # Apply imbalance to both flat and temporal source data
        X_sf_im, y_sf_im = apply_imbalance(X_sf, y_sf, ir, rng)
        X_st_im, y_st_im = apply_imbalance(X_st, y_st, ir, rng)
        logger.info(f"IR={ir}:1  flat src={X_sf_im.shape[0]:,}  "
                    f"temporal src={X_st_im.shape[0]:,}")

        for name, (mode, build) in model_configs.items():
            if is_done(records, name, ir):
                logger.info(f"  [{name}] IR={ir} — done, skip"); continue

            model = build()
            if mode == "temporal":
                m = train_ta_datl(model,X_st_im,y_st_im,X_tt,y_tt,lb_t,n_cls,EPOCHS,LR,128)
            elif name == "FixBi":
                m = train_fixbi(model,X_sf_im,y_sf_im,X_tf,y_tf,lb_f,n_cls,EPOCHS,LR,256)
            else:
                m = train_adversarial(model,name,X_sf_im,y_sf_im,X_tf,y_tf,lb_f,n_cls,EPOCHS,LR,256)

            records[name].append({"imbalance_ratio":ir, **m})
            save_checkpoint(records)
            logger.info(f"  {name:<18} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")

    with open(TAB_DIR/"class_imbalance.json","w") as f: json.dump(records,f,indent=2)

    # Plot
    colors  = {"DANN":"#9E9E9E","CDAN":"#2196F3","FixBi":"#4CAF50",
               "ToAlign":"#FF9800","DATL":"#9C27B0","TA-DATL (Ours)":"#F44336"}
    markers = {"DANN":"o","CDAN":"s","FixBi":"^","ToAlign":"D","DATL":"P","TA-DATL (Ours)":"*"}
    fig, axes = plt.subplots(1,3,figsize=(16,5))
    for ax, metric, title in zip(axes,["accuracy","f1","auc"],["Accuracy","F1-Score","AUC"]):
        for name, recs in records.items():
            xs=[r["imbalance_ratio"] for r in recs]; ys=[r[metric] for r in recs]
            ax.plot(xs,ys,marker=markers[name],color=colors[name],
                    label=name,linewidth=2,markersize=6)
        ax.set_xlabel("Imbalance Ratio (IR)"); ax.set_ylabel(title)
        ax.set_title(f"{title} vs Class Imbalance",fontweight="bold")
        ax.set_xscale("log"); ax.set_xticks(IMBALANCE_RATIOS)
        ax.set_xticklabels([f"{ir}:1" for ir in IMBALANCE_RATIOS])
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
    plt.suptitle("Performance under Class Imbalance",fontsize=14,fontweight="bold",y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR/"figure3_class_imbalance.png",dpi=150,bbox_inches="tight")
    plt.close()
    logger.info(f"\nFigure → {FIG_DIR}/figure3_class_imbalance.png")
    return records


if __name__ == "__main__":
    run()
