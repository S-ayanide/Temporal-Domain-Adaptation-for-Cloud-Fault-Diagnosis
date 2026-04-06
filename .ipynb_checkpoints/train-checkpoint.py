"""
train.py
========
Training loops for CWPDDA and MCTL.

CWPDDA training (Section 4.1 of paper):
  - Joint optimisation: Ly + Lf + Ld
  - 70/20/10 split, lr=1e-3, dropout=0.1, α=10, β=0.75

MCTL training (Section 3 of Zuo et al.):
  - Stage 1: source encoder pretraining on Google data
  - Stage 2a: contrastive KL transfer
  - Stage 2b: regression head fine-tuning
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _loader(X, y, bs, shuffle=True):
    return DataLoader(
        TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float()),
        batch_size=bs, shuffle=shuffle, drop_last=False,
    )


def _val_mse(model_predict_fn, X, y, device):
    model_predict_fn.__self__.eval() if hasattr(model_predict_fn, '__self__') else None
    with torch.no_grad():
        xb = torch.from_numpy(X).float().to(device)
        pred = model_predict_fn(xb).cpu().numpy()
    return float(np.mean((pred.squeeze() - y.squeeze()) ** 2))


# ─── CWPDDA training ──────────────────────────────────────────────────────────

def train_cwpdda(
    model,
    data: dict,
    device: str = "cpu",
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 15,
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Joint training of CWPDDA.

    data keys expected:
        src_X, src_y        — Google source windows
        tgt_train_X/y       — Alibaba train
        tgt_val_X/y         — Alibaba val
        tgt_test_X/y        — Alibaba test
    """
    from models.cwpdda import grl_lambda

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    X_src, y_src = data["src_X"], data["src_y"]
    X_tr,  y_tr  = data["tgt_train_X"], data["tgt_train_y"]
    X_val, y_val = data["tgt_val_X"],   data["tgt_val_y"]

    n = min(len(X_src), len(X_tr))
    dl_s = DataLoader(TensorDataset(torch.from_numpy(X_src[:n]).float(),
                                     torch.from_numpy(y_src[:n]).float()),
                      batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False, num_workers=0)
    dl_t = DataLoader(TensorDataset(torch.from_numpy(X_tr[:n]).float(),
                                     torch.from_numpy(y_tr[:n]).float()),
                      batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False, num_workers=0)

    total_steps = epochs * min(len(dl_s), len(dl_t))
    step = 0

    best_val, best_state, no_improve = float("inf"), None, 0
    history = []

    if verbose:
        import torch as _t
        _g = (f"  GPU: {_t.cuda.get_device_name(0)}"
              if device.startswith("cuda") and _t.cuda.is_available() else "")
        print(f"\n[CWPDDA] Training — {epochs} epochs | device={device}{_g}")
        if device.startswith("cuda"):
            _t.cuda.empty_cache()
            free = _t.cuda.mem_get_info(0)[0] / 1024**3
            total = _t.cuda.mem_get_info(0)[1] / 1024**3
            print(f"        GPU memory: {free:.1f} GiB free / {total:.1f} GiB total")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for (xs, ys), (xt, yt) in zip(dl_s, dl_t):
            xs, ys = xs.to(device), ys.to(device)
            xt, yt = xt.to(device), yt.to(device)

            opt.zero_grad()
            loss, info = model.compute_loss(xs, ys, xt, yt, step, total_steps)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            step += 1

        epoch_loss /= max(len(dl_s), 1)

        # Validation
        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(X_val).float().to(device)
            xs_dummy = xv  # inference: use target as dummy source
            zv, _, _ = model.extractor(xs_dummy, xv)
            pred_val = model.predictor(zv).cpu().numpy()
        val_mse = float(np.mean((pred_val.squeeze() - y_val.squeeze()) ** 2))
        sched.step(val_mse)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_mse": val_mse})

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch:3d}/{epochs}  loss={epoch_loss:.5f}  val_mse={val_mse:.5f}")

        if no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch}  best_val_mse={best_val:.5f}")
            break

    if best_state:
        model.load_state_dict(best_state)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(save_dir) / "cwpdda.pt")

    return {"history": history, "best_val_mse": best_val}


# ─── MCTL training (three stages) ────────────────────────────────────────────

def train_mctl(
    model,
    data: dict,
    device: str = "cpu",
    stage1_epochs: int = 50,
    stage2a_epochs: int = 50,
    stage2b_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Three-stage MCTL training."""
    model = model.to(device)

    X_src, y_src = data["src_X"], data["src_y"]
    X_tr,  y_tr  = data["tgt_train_X"], data["tgt_train_y"]
    X_val, y_val = data["tgt_val_X"],   data["tgt_val_y"]

    # ── Stage 1: source encoder pretraining ──────────────────────────────────
    if verbose:
        print(f"\n[MCTL Stage 1] Source encoder pretraining — {stage1_epochs} epochs")

    model.unfreeze_source()
    src_head = nn.Linear(model.hidden_dim, y_src.shape[1]).to(device)
    opt1 = torch.optim.Adam(
        list(model.source_encoder.parameters()) + list(src_head.parameters()), lr=lr
    )
    dl_src = _loader(X_src, y_src, batch_size)
    for epoch in range(1, stage1_epochs + 1):
        model.train(); src_head.train()
        for xb, yb in dl_src:
            xb, yb = xb.to(device), yb.to(device)
            opt1.zero_grad()
            F.mse_loss(src_head(model.source_encoder(xb)), yb).backward()
            opt1.step()
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch}/{stage1_epochs}")

    # ── Stage 2a: contrastive transfer ───────────────────────────────────────
    if verbose:
        print(f"\n[MCTL Stage 2a] Contrastive transfer — {stage2a_epochs} epochs")

    model.freeze_source()
    opt2a = torch.optim.Adam(model.target_encoder.parameters(), lr=lr * 0.5)
    n = min(len(X_src), len(X_tr))
    dl_s2 = DataLoader(TensorDataset(torch.from_numpy(X_src[:n]).float()),
                       batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False, num_workers=0)
    dl_t2 = DataLoader(TensorDataset(torch.from_numpy(X_tr[:n]).float()),
                       batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False, num_workers=0)

    for epoch in range(1, stage2a_epochs + 1):
        model.train()
        ep_loss = 0.0
        for (xs,), (xt,) in zip(dl_s2, dl_t2):
            xs, xt = xs.to(device), xt.to(device)
            opt2a.zero_grad()
            loss = model.transfer_loss(xs, xt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.target_encoder.parameters(), 1.0)
            opt2a.step()
            ep_loss += loss.item()
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch}/{stage2a_epochs}  KL={ep_loss/max(len(dl_s2),1):.5f}")

    # ── Stage 2b: fine-tune regression head ──────────────────────────────────
    if verbose:
        print(f"\n[MCTL Stage 2b] Regression head fine-tune — {stage2b_epochs} epochs")

    opt2b = torch.optim.Adam(
        list(model.target_encoder.parameters()) +
        list(model.regression_head.parameters()),
        lr=lr * 0.1,
    )
    dl_tr = _loader(X_tr, y_tr, batch_size)
    dl_va = _loader(X_val, y_val, batch_size, shuffle=False)
    best_val, best_state, no_improve = float("inf"), None, 0

    for epoch in range(1, stage2b_epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt2b.zero_grad()
            F.mse_loss(model.predict(xb), yb).backward()
            opt2b.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                val_loss += F.mse_loss(model.predict(xb.to(device)), yb.to(device)).item()
        val_loss /= max(len(dl_va), 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch}/{stage2b_epochs}  val_mse={val_loss:.5f}")
        if no_improve >= patience:
            if verbose: print(f"  Early stop at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(save_dir) / "mctl.pt")

    return {"best_val_mse": best_val}