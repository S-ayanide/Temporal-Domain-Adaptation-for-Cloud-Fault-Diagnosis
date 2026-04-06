"""
Step 1: Train All Models → Table 1

Trains 6 methods and prints a comparison table.
Flat models (DANN, CDAN, FixBi, ToAlign, DATL) use mean-pooled windows.
TA-DATL uses the full temporal window tensor.
"""

import argparse
import json, logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from models import DANN, CDAN, FixBi, ToAlign, DATL, TA_DATL
from trainer import train_adversarial, train_fixbi, train_ta_datl


def _parse_args():
    ap = argparse.ArgumentParser(description="Train all models (Table 1)")
    ap.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Directory with meta.json / npz / parquet (default: data/processed). "
        "Use data/processed_google_alibaba after 00_prepare_data_google_alibaba.py.",
    )
    return ap.parse_args()


_args = _parse_args()
PROC_DIR = (
    Path(_args.processed_dir)
    if _args.processed_dir and Path(_args.processed_dir).is_absolute()
    else (BASE_DIR / _args.processed_dir if _args.processed_dir else BASE_DIR / "data" / "processed")
)
PROC_DIR = PROC_DIR.resolve()
CKPT_DIR = BASE_DIR / "checkpoints"
RES_DIR  = BASE_DIR / "results" / "tables"
LOG_DIR  = BASE_DIR / "logs"
for d in [CKPT_DIR, RES_DIR, LOG_DIR]: d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_DIR/"01_train.log"),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logger.info("Using processed data directory: %s", PROC_DIR)

HIDDEN   = 128
DROPOUT  = 0.3
EPOCHS   = 200
LR       = 1e-3
BATCH_F  = 256   # flat models
BATCH_T  = 128   # temporal model (windows are larger)


def load():
    with open(PROC_DIR/"meta.json") as f: meta = json.load(f)

    # Temporal data (for TA-DATL)
    src_t = np.load(PROC_DIR/"source_temporal.npz")
    tgt_t = np.load(PROC_DIR/"target_temporal.npz")
    X_src_t = src_t["X"].astype(np.float32)    # (N, W, F)
    y_src_t = src_t["y"].astype(np.int64)
    X_tgt_t = tgt_t["X"].astype(np.float32)
    y_tgt_t = tgt_t["y"].astype(np.int64)
    tgt_lb  = tgt_t["labeled"].astype(bool)

    # Normalise each feature channel across time (fit on source, apply to target)
    N_s, W, F = X_src_t.shape
    sc = StandardScaler()
    X_src_t = sc.fit_transform(X_src_t.reshape(-1, F)).reshape(N_s, W, F).astype(np.float32)
    N_t      = X_tgt_t.shape[0]
    X_tgt_t  = sc.transform(X_tgt_t.reshape(-1, F)).reshape(N_t, W, F).astype(np.float32)

    # Flat data (mean-pooled, for baselines)
    src_f = pd.read_parquet(PROC_DIR/"source_flat.parquet")
    tgt_f = pd.read_parquet(PROC_DIR/"target_flat.parquet")
    feat_cols = [c for c in src_f.columns if c not in {"label","labeled"}]
    X_src_f = src_f[feat_cols].values.astype(np.float32)
    y_src_f = src_f["label"].values.astype(np.int64)
    X_tgt_f = tgt_f[feat_cols].values.astype(np.float32)
    y_tgt_f = tgt_f["label"].values.astype(np.int64)
    tgt_lb_f = tgt_f["labeled"].values.astype(bool)
    sc_f = StandardScaler()
    X_src_f = sc_f.fit_transform(X_src_f).astype(np.float32)
    X_tgt_f = sc_f.transform(X_tgt_f).astype(np.float32)

    n_classes = int(meta["n_classes"])
    input_f   = X_src_f.shape[1]
    input_t   = F
    logger.info(f"Flat  — src:{X_src_f.shape} tgt:{X_tgt_f.shape}")
    logger.info(f"Temporal — src:{X_src_t.shape} tgt:{X_tgt_t.shape}")
    return (X_src_f, y_src_f, X_tgt_f, y_tgt_f, tgt_lb_f,
            X_src_t, y_src_t, X_tgt_t, y_tgt_t, tgt_lb,
            n_classes, input_f, input_t)


def run():
    logger.info("="*65)
    logger.info("  Training all models (Table 1)")
    logger.info("="*65)
    (X_sf, y_sf, X_tf, y_tf, lb_f,
     X_st, y_st, X_tt, y_tt, lb_t,
     n_cls, in_f, in_t) = load()

    results = {}
    t0 = time.time()

    def _flat(name, model):
        logger.info(f"\n[{name}]")
        if name == "FixBi":
            m = train_fixbi(model, X_sf, y_sf, X_tf, y_tf, lb_f,
                            n_cls, EPOCHS, LR, BATCH_F, CKPT_DIR/f"{name.lower()}.pt")
        else:
            m = train_adversarial(model, name, X_sf, y_sf, X_tf, y_tf, lb_f,
                                  n_cls, EPOCHS, LR, BATCH_F, CKPT_DIR/f"{name.lower()}.pt")
        results[name] = m
        logger.info(f"  {name:<12} acc={m['accuracy']*100:.1f}  "
                    f"f1={m['f1']*100:.1f}  auc={m['auc']*100:.1f}")

    _flat("DANN",    DANN(in_f,    n_cls, HIDDEN, DROPOUT))
    _flat("CDAN",    CDAN(in_f,    n_cls, HIDDEN, DROPOUT))
    _flat("FixBi",   FixBi(in_f,   n_cls, HIDDEN, DROPOUT))
    _flat("ToAlign", ToAlign(in_f, n_cls, HIDDEN, DROPOUT))
    _flat("DATL",    DATL(in_f,    n_cls, HIDDEN, DROPOUT))

    # TA-DATL (temporal)
    logger.info(f"\n[TA-DATL (Ours)]")
    ta = TA_DATL(in_t, n_cls, hidden_dim=120, n_groups=4,
                 dropout=DROPOUT, lambda1=0.1, lambda2=0.1,
                 pseudo_threshold=0.85, temperature=1.5)
    m = train_ta_datl(ta, X_st, y_st, X_tt, y_tt, lb_t,
                      n_cls, EPOCHS, LR, BATCH_T,
                      save_path=CKPT_DIR/"ta_datl.pt")
    results["TA-DATL (Ours)"] = m
    logger.info(f"  TA-DATL      acc={m['accuracy']*100:.1f}  "
                f"f1={m['f1']*100:.1f}  auc={m['auc']*100:.1f}")

    logger.info(f"\nTotal time: {(time.time()-t0)/60:.1f} min")

    # ── Print table ──────────────────────────────────────────────────────────
    ref = {"DANN":   (84.2,81.0,87.2), "CDAN":   (85.7,82.3,88.5),
           "FixBi":  (86.3,83.1,89.2), "ToAlign":(87.1,83.7,89.9),
           "DATL":   (89.6,85.4,91.3)}

    sep  = "-"*60
    rows = [f"\n{'Method':<18}  {'Accuracy':>9}  {'F1-Score':>8}  {'AUC':>8}", sep]
    for name, m in results.items():
        rows.append(f"  {name:<16}  {m['accuracy']*100:>8.1f}  "
                    f"{m['f1']*100:>7.1f}  {m['auc']*100:>7.1f}")
    rows += [sep, "\n  Replicated paper baselines (reference):"]
    for name, (a,f,u) in ref.items():
        rows.append(f"  {name:<16}  {a:>8.1f}  {f:>7.1f}  {u:>7.1f}")
    rows.append(sep)
    table = "\n".join(rows)
    logger.info(table)

    with open(RES_DIR/"table1.json","w") as f: json.dump(results,f,indent=2)
    with open(RES_DIR/"table1.txt","w")  as f: f.write(table+"\n")
    logger.info(f"\nSaved to {RES_DIR}/table1.*")
    return results


if __name__ == "__main__":
    run()
