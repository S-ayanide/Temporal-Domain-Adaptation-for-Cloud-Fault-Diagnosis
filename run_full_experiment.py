"""
run_full_experiment.py
======================
Trains and evaluates ALL seven methods for ONE transfer direction.
Produces regression metrics (MAE, MAPE, RMSE) AND classification metrics
(Accuracy, Precision, Recall, F1, MCC, G-Mean) for every method.

Usage (on GPU server):
    # Google → Alibaba
    python run_full_experiment.py \
        --direction google_to_alibaba \
        --google  data/raw/google \
        --alibaba data/raw/alibaba \
        --device  cuda \
        --out     results/g2a

    # Alibaba → Google  (swap source/target)
    python run_full_experiment.py \
        --direction alibaba_to_google \
        --google  data/raw/google \
        --alibaba data/raw/alibaba \
        --device  cuda \
        --out     results/a2g

    # Load cached preprocessed data (skip slow reload)
    python run_full_experiment.py \
        --direction google_to_alibaba \
        --load-cache results/g2a/preprocessed.npz \
        --device cuda --out results/g2a

    # Skip DeepJDOT (POT library needed) and GluonTS (DeepAR/DRP/MQF2)
    python run_full_experiment.py ... --skip-deepjdot --skip-gluonts

Classification threshold:
    By default the threshold is set automatically to the 70th percentile of
    y_true on the normalised 0-1 scale — top 30% CPU = "high load" event.
    Override with --clf-threshold 0.8 etc.

Output (per direction):
    results/<dir>/
        all_results.json          — all methods × all metrics
        regression_results.json   — regression metrics only (MAE/MAPE/RMSE/MSE)
        classification_results.json — Acc/Prec/Recall/F1/MCC/G-Mean
        preprocessed.npz + .json  — cached preprocessed arrays
        checkpoints/              — saved model weights (.pt files)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--direction", required=True,
                   choices=["google_to_alibaba", "alibaba_to_google"],
                   help="Transfer direction")
    p.add_argument("--google",  default="data/raw/google",  help="Google raw data dir")
    p.add_argument("--alibaba", default="data/raw/alibaba", help="Alibaba raw data dir")
    p.add_argument("--out",     default="results/experiment", help="Output directory")
    p.add_argument("--device",  default="cuda", help="cuda | cpu")
    p.add_argument("--seed",    type=int, default=42)

    # Data size
    p.add_argument("--max-source", type=int, default=3000, help="Max source series to load")
    p.add_argument("--max-target", type=int, default=3000, help="Max target series to load")

    # Cache
    p.add_argument("--load-cache", default=None, metavar="PATH.npz",
                   help="Skip preprocessing — load from saved .npz")
    p.add_argument("--save-cache", default=None, metavar="PATH.npz",
                   help="Save preprocessed arrays to .npz after preprocessing")

    # Classification threshold (on 0-1 normalised scale)
    p.add_argument("--clf-threshold", type=float, default=None,
                   help="Binarisation threshold (default: 70th percentile of y_true)")

    # Training epochs
    p.add_argument("--lstm-epochs",     type=int, default=150)
    p.add_argument("--cwpdda-epochs",   type=int, default=100)
    p.add_argument("--nbeats-epochs",   type=int, default=100)
    p.add_argument("--mctl-s1-epochs",  type=int, default=50)
    p.add_argument("--mctl-s2a-epochs", type=int, default=50)
    p.add_argument("--mctl-s2b-epochs", type=int, default=50)
    p.add_argument("--mc-s1-epochs",    type=int, default=30)
    p.add_argument("--mc-s2-epochs",    type=int, default=50)
    p.add_argument("--mc-s3-epochs",    type=int, default=100)
    p.add_argument("--deepjdot-epochs", type=int, default=50)

    # Training hyperparams
    p.add_argument("--batch-size",  type=int,   default=256,
                   help="Batch size (256 is efficient on 40GB GPU)")
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--patience",    type=int,   default=20)
    p.add_argument("--d-model",     type=int,   default=64)
    p.add_argument("--lstm-hidden", type=int,   default=40)
    p.add_argument("--lstm-layers", type=int,   default=2)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--proj-dim",    type=int,   default=64)

    # ARIMA subsample (fitting is O(n²))
    p.add_argument("--arima-subsample", type=int, default=1000,
                   help="Max test windows for ARIMA (fitting is slow)")

    # Skip flags
    p.add_argument("--skip-gluonts",  action="store_true",
                   help="Skip DeepAR/DRP/MQF2 (require gluonts)")
    p.add_argument("--skip-deepjdot", action="store_true",
                   help="Skip DeepJDOT (requires POT library)")
    p.add_argument("--quick",         action="store_true",
                   help="Smoke test: few epochs, small data")

    # Checkpoint every N epochs
    p.add_argument("--ckpt-every", type=int, default=10)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────

def _elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    return f"{s//3600:02d}h{(s%3600)//60:02d}m{s%60:02d}s"


def _banner(msg: str):
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}", flush=True)


def _save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(f"  Saved: {path}", flush=True)


# ─── Metric helpers ───────────────────────────────────────────────────────────

def _reg_and_clf(y_true: np.ndarray, y_pred: np.ndarray,
                 scale: str, threshold: float | None) -> dict:
    """Return {regression: {...}, classification: {...}}."""
    from evaluate import cwpdda_metrics, mctl_metrics, classification_metrics
    reg = cwpdda_metrics(y_true, y_pred) if scale == "cwpdda" else mctl_metrics(y_true, y_pred)
    clf = classification_metrics(y_true, y_pred, threshold=threshold)
    return {"regression": reg, "classification": clf}


# ─── Baseline inference (ARIMA, LSTM, etc.) ───────────────────────────────────

def _baseline_predict(model, X_test: np.ndarray) -> np.ndarray:
    return model.predict(X_test)


def _nn_predict(model, X_test: np.ndarray, device: str,
                batch_size: int = 4096) -> np.ndarray:
    import torch
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = torch.from_numpy(X_test[i:i+batch_size]).float().to(device)
            out = model.predict(xb) if hasattr(model, "predict") \
                  else model(xb)
            parts.append(out.cpu().numpy() if hasattr(out, "cpu") else out)
    return np.concatenate(parts, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)
    t_total = time.time()

    # ── Quick mode ────────────────────────────────────────────────────────────
    if args.quick:
        args.max_source = 300
        args.max_target = 300
        args.lstm_epochs     = 5
        args.cwpdda_epochs   = 5
        args.nbeats_epochs   = 5
        args.mctl_s1_epochs  = 3
        args.mctl_s2a_epochs = 3
        args.mctl_s2b_epochs = 3
        args.mc_s1_epochs    = 3
        args.mc_s2_epochs    = 3
        args.mc_s3_epochs    = 5
        args.deepjdot_epochs = 5
        args.skip_gluonts    = True
        args.arima_subsample = 200
        print("[quick mode] reduced epochs and data for smoke test")

    # ── GPU check ─────────────────────────────────────────────────────────────
    import torch
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[WARNING] CUDA not available — falling back to CPU")
            args.device = "cpu"
        else:
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            print(f"\n[GPU] {args.device} — {name}  ({mem} GB)\n")

    out_dir  = Path(args.out)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    direction_label = args.direction.replace("_", " → ").replace("google", "Google").replace("alibaba", "Alibaba")
    print(f"\n[Direction] {direction_label}", flush=True)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    _banner("Step 1/4 — Load & Preprocess")

    from preprocess import build_source_target, save_preprocess_cache, load_preprocess_cache

    if args.load_cache:
        data = load_preprocess_cache(args.load_cache)
        print(f"  Loaded from cache: {args.load_cache}")
    else:
        from data_loader import load_google, load_alibaba

        print(f"  Loading Google  (max {args.max_source} series)...")
        google_series  = load_google(args.google,  max_series=args.max_source)
        print(f"  Loading Alibaba (max {args.max_target} series)...")
        alibaba_series = load_alibaba(args.alibaba, max_series=args.max_target, no_chunk=False)

        # Swap source/target for Alibaba → Google direction
        if args.direction == "google_to_alibaba":
            src_series, tgt_series = google_series, alibaba_series
        else:
            src_series, tgt_series = alibaba_series, google_series

        print(f"  Preprocessing (source={len(src_series)} series, target={len(tgt_series)} series)...")
        data = build_source_target(
            src_series, tgt_series,
            use_dtw=False, seed=args.seed,
        )

        cache_path = args.save_cache or str(out_dir / "preprocessed.npz")
        save_preprocess_cache(cache_path, data)
        print(f"  Saved preprocessed cache → {cache_path}")

    print(f"\n  src windows:       {data['src_X'].shape[0]:>10,}")
    print(f"  tgt train windows: {data['tgt_train_X'].shape[0]:>10,}")
    print(f"  tgt test windows:  {data['tgt_test_X'].shape[0]:>10,}")

    X_tr = data["tgt_train_X"];  y_tr = data["tgt_train_y"]
    X_te = data["tgt_test_X"];   y_te = data["tgt_test_y"]
    X_src = data["src_X"];       y_src = data["src_y"]
    W = X_tr.shape[1]
    horizon = y_tr.shape[1]

    # Classification threshold: computed once from test ground truth (0-1 scale)
    clf_thr = args.clf_threshold
    if clf_thr is None:
        clf_thr = float(np.percentile(y_te, 70))
        print(f"\n  [clf] Auto threshold = {clf_thr:.4f}  "
              f"(70th pct of y_test — top 30% = high-load class)")
    else:
        print(f"\n  [clf] Fixed threshold = {clf_thr:.4f}")

    all_results: dict = {}   # method → {regression: {...}, classification: {...}}

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: ARIMA baseline
    # ──────────────────────────────────────────────────────────────────────────
    _banner("Step 2/4a — ARIMA baseline")
    t0 = time.time()
    from baselines import ARIMABaseline
    arima = ARIMABaseline()
    arima.fit(X_tr, y_tr)
    n_arima = min(args.arima_subsample, len(X_te))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(X_te), n_arima, replace=False)
    pred_arima = _baseline_predict(arima, X_te[idx])
    all_results["ARIMA"] = _reg_and_clf(y_te[idx], pred_arima, "cwpdda", clf_thr)
    print(f"  ARIMA done  ({_elapsed(t0)},  n={n_arima})", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2b: LSTM baseline
    # ──────────────────────────────────────────────────────────────────────────
    _banner("Step 2/4b — LSTM baseline")
    t0 = time.time()
    from baselines import LSTMBaseline
    lstm_kw = dict(window_size=W, horizon=horizon, epochs=args.lstm_epochs, device=args.device)
    lstm = LSTMBaseline(**lstm_kw)
    lstm.fit(X_tr, y_tr)
    pred_lstm = _baseline_predict(lstm, X_te)
    all_results["LSTM"] = _reg_and_clf(y_te, pred_lstm, "cwpdda", clf_thr)
    print(f"  LSTM done  ({_elapsed(t0)})", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2c: GRU baseline
    # ──────────────────────────────────────────────────────────────────────────
    _banner("Step 2/4c — GRU baseline")
    t0 = time.time()
    from baselines import GRUBaseline
    gru = GRUBaseline(window_size=W, horizon=horizon, epochs=args.lstm_epochs, device=args.device)
    gru.fit(X_tr, y_tr)
    pred_gru = _baseline_predict(gru, X_te)
    all_results["GRU"] = _reg_and_clf(y_te, pred_gru, "cwpdda", clf_thr)
    print(f"  GRU done  ({_elapsed(t0)})", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2d: CNN-LSTM baseline
    # ──────────────────────────────────────────────────────────────────────────
    _banner("Step 2/4d — CNN-LSTM baseline")
    t0 = time.time()
    from baselines import CNNLSTMBaseline
    cnnlstm = CNNLSTMBaseline(window_size=W, horizon=horizon, epochs=50, device=args.device)
    cnnlstm.fit(X_tr, y_tr)
    pred_cnnlstm = _baseline_predict(cnnlstm, X_te)
    all_results["CNN-LSTM"] = _reg_and_clf(y_te, pred_cnnlstm, "cwpdda", clf_thr)
    print(f"  CNN-LSTM done  ({_elapsed(t0)})", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: N-BEATS (zero-shot: trained on source only)
    # ──────────────────────────────────────────────────────────────────────────
    _banner("Step 3/4 — N-BEATS (zero-shot transfer)")
    t0 = time.time()
    from nbeats import NBeats
    from train import train_nbeats
    nbeats_model = NBeats(window_size=W, horizon=horizon, n_blocks=8,
                          n_layers=4, hidden_size=256)
    train_nbeats(nbeats_model, data,
                 device=args.device,
                 epochs=args.nbeats_epochs,
                 batch_size=args.batch_size,
                 lr=args.lr,
                 patience=args.patience,
                 save_dir=str(ckpt_dir),
                 verbose=True,
                 checkpoint_every=args.ckpt_every)
    pred_nbeats = nbeats_model.predict_numpy_batched(X_te, args.device)
    all_results["N-BEATS"] = _reg_and_clf(y_te, pred_nbeats, "cwpdda", clf_thr)
    torch.save(nbeats_model.state_dict(), ckpt_dir / "nbeats_final.pt")
    print(f"  N-BEATS done  ({_elapsed(t0)})", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: DeepJDOT
    # ──────────────────────────────────────────────────────────────────────────
    if not args.skip_deepjdot:
        _banner("Step 4/4 — DeepJDOT")
        t0 = time.time()
        try:
            sys.path.insert(0, str(Path(__file__).parent / "deepjdot"))
            from deepjdot.model import DeepJDOT
            from deepjdot.train import train_deepjdot
            dj = DeepJDOT(window_size=W, horizon=horizon)
            train_deepjdot(dj, data,
                           device=args.device,
                           epochs=args.deepjdot_epochs,
                           batch_size=args.batch_size,
                           lr=args.lr,
                           save_dir=str(ckpt_dir),
                           verbose=True)
            pred_dj = dj.predict_numpy_batched(X_te, args.device)
            all_results["DeepJDOT"] = _reg_and_clf(y_te, pred_dj, "cwpdda", clf_thr)
            torch.save(dj.state_dict(), ckpt_dir / "deepjdot_final.pt")
            print(f"  DeepJDOT done  ({_elapsed(t0)})", flush=True)
        except Exception as e:
            print(f"  [SKIP] DeepJDOT failed: {e}", flush=True)
            all_results["DeepJDOT"] = {"error": str(e)}
    else:
        print("  [SKIP] DeepJDOT (--skip-deepjdot)", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5: CWPDDA
    # ──────────────────────────────────────────────────────────────────────────
    _banner("Step 5/4 — CWPDDA (adversarial domain adaptation)")
    t0 = time.time()
    from cwpdda import CWPDDA
    from train import train_cwpdda
    cwpdda_model = CWPDDA(window_size=W, d_model=args.d_model,
                          lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers,
                          dropout=args.dropout, horizon=horizon)
    train_cwpdda(cwpdda_model, data,
                 device=args.device,
                 epochs=args.cwpdda_epochs,
                 batch_size=args.batch_size,
                 lr=args.lr,
                 patience=args.patience,
                 save_dir=str(ckpt_dir),
                 verbose=True,
                 checkpoint_every=args.ckpt_every)
    pred_cwpdda = cwpdda_model.predict_numpy_batched(X_te, args.device)
    all_results["CWPDDA"] = _reg_and_clf(y_te, pred_cwpdda, "cwpdda", clf_thr)
    torch.save(cwpdda_model.state_dict(), ckpt_dir / "cwpdda_final.pt")
    print(f"  CWPDDA done  ({_elapsed(t0)})", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 6: MC-CWPDDA
    # ──────────────────────────────────────────────────────────────────────────
    _banner("Step 6/4 — MC-CWPDDA (adversarial + contrastive)")
    t0 = time.time()
    from mc_cwpdda import MCCWPDDA
    from train import train_mc_cwpdda
    mc_model = MCCWPDDA(window_size=W, d_model=args.d_model,
                        lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers,
                        dropout=args.dropout, horizon=horizon, proj_dim=args.proj_dim)
    train_mc_cwpdda(mc_model, data,
                    device=args.device,
                    stage1_epochs=args.mc_s1_epochs,
                    stage2_epochs=args.mc_s2_epochs,
                    stage3_epochs=args.mc_s3_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    patience=args.patience,
                    save_dir=str(ckpt_dir),
                    verbose=True,
                    checkpoint_every=args.ckpt_every)
    pred_mc = mc_model.predict_numpy_batched(X_te, args.device)
    all_results["MC-CWPDDA"] = _reg_and_clf(y_te, pred_mc, "cwpdda", clf_thr)
    torch.save(mc_model.state_dict(), ckpt_dir / "mc_cwpdda_final.pt")
    print(f"  MC-CWPDDA done  ({_elapsed(t0)})", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 7: MCTL (+ full set of neural baselines)
    # ──────────────────────────────────────────────────────────────────────────
    _banner("Step 7/4 — MCTL + baselines (normalised scale)")
    t0 = time.time()
    from mctl import MCTL
    from train import train_mctl
    mctl_model = MCTL(window_size=W, hidden_dim=128, horizon=horizon)
    train_mctl(mctl_model, data,
               device=args.device,
               stage1_epochs=args.mctl_s1_epochs,
               stage2a_epochs=args.mctl_s2a_epochs,
               stage2b_epochs=args.mctl_s2b_epochs,
               batch_size=args.batch_size,
               lr=args.lr,
               save_dir=str(ckpt_dir),
               verbose=True)
    pred_mctl = _nn_predict(mctl_model, X_te, args.device)
    all_results["MCTL"] = _reg_and_clf(y_te, pred_mctl, "mctl", clf_thr)
    torch.save(mctl_model.state_dict(), ckpt_dir / "mctl_final.pt")

    # MCTL neural baselines (same 50-epoch setup, normalised scale)
    mctl_baselines = [
        ("Autoformer", "AutoformerBaseline"),
        ("BHT-ARIMA",  "BHTARIMABaseline"),
        ("TS2Vec",     "TS2VecBaseline"),
    ]
    from baselines import AutoformerBaseline, BHTARIMABaseline, TS2VecBaseline, WANNBaseline
    for name, cls_name in mctl_baselines:
        try:
            t1 = time.time()
            cls = eval(cls_name)
            m = cls(window_size=W, horizon=horizon, epochs=50, device=args.device)
            m.fit(X_tr, y_tr)
            pred = _baseline_predict(m, X_te)
            all_results[name] = _reg_and_clf(y_te, pred, "mctl", clf_thr)
            print(f"  {name} done  ({_elapsed(t1)})", flush=True)
        except Exception as e:
            print(f"  [SKIP] {name}: {e}", flush=True)
            all_results[name] = {"error": str(e)}

    # WANN (needs source data)
    try:
        t1 = time.time()
        wann = WANNBaseline(window_size=W, horizon=horizon, epochs=50, device=args.device)
        wann.fit(X_src, y_src, X_tr, y_tr)
        pred_wann = _baseline_predict(wann, X_te)
        all_results["WANN"] = _reg_and_clf(y_te, pred_wann, "mctl", clf_thr)
        print(f"  WANN done  ({_elapsed(t1)})", flush=True)
    except Exception as e:
        print(f"  [SKIP] WANN: {e}", flush=True)
        all_results["WANN"] = {"error": str(e)}

    print(f"  MCTL block done  ({_elapsed(t0)})", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Save results
    # ──────────────────────────────────────────────────────────────────────────
    _banner("Saving results")

    # Full results (regression + classification nested)
    _save_json(out_dir / "all_results.json", all_results)

    # Flat regression-only table
    reg_only = {}
    for method, res in all_results.items():
        if "regression" in res:
            reg_only[method] = res["regression"]
        elif "error" not in res:
            reg_only[method] = res
    _save_json(out_dir / "regression_results.json", reg_only)

    # Flat classification-only table
    clf_only = {}
    for method, res in all_results.items():
        if "classification" in res:
            clf_only[method] = res["classification"]
    _save_json(out_dir / "classification_results.json", clf_only)

    # Summary to stdout
    print(f"\n{'='*70}")
    print(f"  RESULTS — {direction_label}")
    print(f"  Classification threshold: {clf_thr:.4f}  (top 30% CPU = high-load)")
    print(f"{'='*70}")
    print(f"\n  {'Method':<14}  {'MAE':>8}  {'MAPE%':>7}  {'RMSE':>8}  "
          f"  {'Acc':>6}  {'F1':>6}  {'MCC':>6}  {'G-Mean':>7}")
    print(f"  {'-'*80}")
    for method, res in all_results.items():
        if "error" in res:
            print(f"  {method:<14}  [FAILED: {res['error'][:40]}]")
            continue
        r = res.get("regression", {})
        c = res.get("classification", {})
        mae   = r.get("MAE",    r.get("MAE",     float("nan")))
        mape  = r.get("MAPE_%", r.get("MAPE",    float("nan")))
        rmse  = r.get("RMSE",   float("nan"))
        acc   = c.get("Accuracy",  float("nan"))
        f1    = c.get("F1",        float("nan"))
        mcc   = c.get("MCC",       float("nan"))
        gmean = c.get("G-Mean",    float("nan"))
        print(f"  {method:<14}  {mae:8.4f}  {mape:7.2f}  {rmse:8.4f}  "
              f"  {acc:6.4f}  {f1:6.4f}  {mcc:6.4f}  {gmean:7.4f}")

    print(f"\n  Total runtime: {_elapsed(t_total)}")
    print(f"  Results saved to: {out_dir.resolve()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
