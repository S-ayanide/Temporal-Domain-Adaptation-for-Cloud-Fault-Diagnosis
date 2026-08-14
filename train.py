"""
train.py
========
Training loops for CWPDDA, MCTL, MC-CWPDDA, and N-BEATS.

CWPDDA training (Section 4.1 of paper):
  - Joint optimisation: Ly + Lf + Ld
  - 70/20/10 split, lr=1e-3, dropout=0.1, α=10, β=0.75

MCTL training (Section 3 of Zuo et al.):
  - Stage 1: source encoder pretraining on Google data
  - Stage 2a: contrastive KL transfer
  - Stage 2b: regression head fine-tuning

N-BEATS training (Oreshkin et al., AAAI 2021):
  - Source-only training on Google data (zero-shot: no Alibaba at train time)
  - MaxAbs per-window scaling handles cross-domain amplitude differences
  - Evaluated directly on Alibaba without fine-tuning
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


def _cuda_device_index(device: str) -> int:
    if not device.startswith("cuda"):
        return 0
    if ":" in device:
        return int(device.split(":", 1)[1])
    return 0


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
    checkpoint_every: int = 10,
    resume_from: Optional[str] = None,
) -> dict:
    """
    Joint training of CWPDDA.

    data keys expected:
        src_X, src_y        — Google source windows
        tgt_train_X/y       — Alibaba train
        tgt_val_X/y         — Alibaba val
        tgt_test_X/y        — Alibaba test

    checkpoint_every: save a recovery checkpoint every N epochs (survives server timeouts)
    resume_from:      path to a checkpoint file written by this function to resume training
    """
    from cwpdda import grl_lambda

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.7)

    X_src = data["src_X"]  # y_src unused: compute_loss only predicts on target
    X_tr,  y_tr  = data["tgt_train_X"], data["tgt_train_y"]
    X_val, y_val = data["tgt_val_X"],   data["tgt_val_y"]

    model.register_source_ref(X_src)
    if device.startswith("cuda") and model._src_ref is not None:
        model._src_ref = model._src_ref.to(device)

    dl_t = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(),
                                     torch.from_numpy(y_tr).float()),
                      batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False, num_workers=0)

    total_steps = epochs * len(dl_t)  # GRL schedule based on target batches
    step = 0

    best_val, best_state, no_improve = float("inf"), None, 0
    history = []
    start_epoch = 1

    # ── Resume from checkpoint if requested ───────────────────────────────────
    if resume_from and Path(resume_from).is_file():
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_epoch = ckpt["epoch"] + 1
        step        = ckpt["step"]
        best_val    = ckpt["best_val"]
        history     = ckpt.get("history", [])
        if verbose:
            print(f"\n[CWPDDA] Resuming from epoch {ckpt['epoch']} "
                  f"(best_val_mse={best_val:.5f})")

    if verbose:
        import torch as _t
        _di = _cuda_device_index(device)
        _g = (f"  GPU: {_t.cuda.get_device_name(_di)}"
              if device.startswith("cuda") and _t.cuda.is_available() else "")
        print(f"\n[CWPDDA] Training — epochs {start_epoch}–{epochs} | device={device}{_g}")
        if device.startswith("cuda"):
            _t.cuda.empty_cache()
            free = _t.cuda.mem_get_info(_di)[0] / 1024**3
            total_mem = _t.cuda.mem_get_info(_di)[1] / 1024**3
            print(f"        GPU memory: {free:.1f} GiB free / {total_mem:.1f} GiB total")

    ckpt_dir = Path(save_dir) if save_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_Ly = 0.0; epoch_Lf = 0.0; epoch_Ld = 0.0

        for (xt, yt) in dl_t:
            xt, yt = xt.to(device), yt.to(device)
            with torch.no_grad():
                xs = model._match_source(xt)  # (B, W) nearest source window

            opt.zero_grad()
            loss, info = model.compute_loss(xs, yt, xt, yt, step, total_steps)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            epoch_Ly   += info["Ly"]
            epoch_Lf   += info["Lf"]
            epoch_Ld   += info["Ld"]
            step += 1

        n_batches = max(len(dl_t), 1)
        epoch_loss /= n_batches
        epoch_Ly   /= n_batches
        epoch_Lf   /= n_batches
        epoch_Ld   /= n_batches

        # Validation — source ref already registered before training loop
        model.eval()
        val_bs = min(4096, max(batch_size * 16, 512))
        if len(X_val) == 0:
            val_mse = float("inf")
        else:
            pred_val = model.predict_numpy_batched(X_val, device, batch_size=val_bs)
            val_mse = float(np.mean((pred_val.squeeze() - y_val.squeeze()) ** 2))
        sched.step(val_mse)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_mse": val_mse})

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            # Save best model immediately whenever it improves
            if ckpt_dir:
                torch.save(best_state, ckpt_dir / "cwpdda_best.pt")
        else:
            no_improve += 1

        if verbose and epoch % 5 == 0:
            print(f"  epoch {epoch:3d}/{epochs}  "
                  f"loss={epoch_loss:.5f}  "
                  f"Ly={epoch_Ly:.5f}  Lf={epoch_Lf:.4f}  Ld={epoch_Ld:.4f}  "
                  f"val_mse={val_mse:.5f}")

        # Periodic recovery checkpoint — survives server timeouts
        if ckpt_dir and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            recovery = {
                "epoch":    epoch,
                "step":     step,
                "best_val": best_val,
                "history":  history,
                "model":    {k: v.clone() for k, v in model.state_dict().items()},
                "opt":      opt.state_dict(),
                "sched":    sched.state_dict(),
            }
            torch.save(recovery, ckpt_dir / "cwpdda_resume.pt")
            if verbose:
                print(f"  [ckpt] Saved recovery checkpoint at epoch {epoch}", flush=True)

        if no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch}  best_val_mse={best_val:.5f}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # Register source reference for correct cross-attention at inference time
    model.register_source_ref(X_src)

    if ckpt_dir:
        torch.save(model.state_dict(), ckpt_dir / "cwpdda.pt")

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


# ─── MC-CWPDDA training (three stages) ───────────────────────────────────────

def train_mc_cwpdda(
    model,
    data: dict,
    device: str = "cpu",
    stage1_epochs: int = 30,
    stage2_epochs: int = 50,
    stage3_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 15,
    save_dir: Optional[str] = None,
    verbose: bool = True,
    checkpoint_every: int = 10,
    resume_from: Optional[str] = None,
) -> dict:
    """
    Three-stage curriculum: (1) pretrain source branch, (2) freeze source + align
    target contrastively, (3) unfreeze all + joint fine-tune with early stopping.
    Supports checkpointing to survive server timeouts.
    """
    model = model.to(device)
    X_src, y_src = data["src_X"],       data["src_y"]
    X_tr,  y_tr  = data["tgt_train_X"], data["tgt_train_y"]
    X_val, y_val = data["tgt_val_X"],   data["tgt_val_y"]

    if verbose:
        print(f"\n[MC-CWPDDA] device={device}")

    ckpt_dir = Path(save_dir) if save_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Source pre-training ─────────────────────────────────────────
    if verbose:
        print(f"\n[MC-CWPDDA Stage 1] Source pre-training — {stage1_epochs} epochs")

    model.unfreeze_all()
    opt1 = torch.optim.Adam(
        list(model.extractor.proj_src.parameters()) +
        list(model.extractor.self_attn_src.parameters()) +
        list(model.predictor.parameters()),
        lr=lr,
    )

    dl_src = _loader(X_src, y_src, batch_size)
    for epoch in range(1, stage1_epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in dl_src:
            xb, yb = xb.to(device), yb.to(device)
            opt1.zero_grad()
            z_shared, _, _ = model.extractor(xb, xb)
            loss = F.mse_loss(model.predictor(z_shared), yb)
            loss.backward()
            opt1.step()
            ep_loss += loss.item()
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d}/{stage1_epochs}  mse={ep_loss/max(len(dl_src),1):.5f}")

    # ── Stage 2: Contrastive alignment ───────────────────────────────────────
    if verbose:
        print(f"\n[MC-CWPDDA Stage 2] Contrastive alignment — {stage2_epochs} epochs")

    model.freeze_source_branch()
    opt2 = torch.optim.Adam(
        list(model.extractor.proj_tgt.parameters()) +
        list(model.extractor.self_attn_tgt.parameters()) +
        list(model.extractor.cross_attn.parameters()) +
        list(model.contrastive_head.parameters()),
        lr=lr * 0.5,
    )

    n = min(len(X_src), len(X_tr))
    dl_s2 = DataLoader(
        TensorDataset(torch.from_numpy(X_src[:n]).float()),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
    )
    dl_t2 = DataLoader(
        TensorDataset(torch.from_numpy(X_tr[:n]).float()),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
    )

    for epoch in range(1, stage2_epochs + 1):
        model.train()
        ep_loss = 0.0
        for (xs,), (xt,) in zip(dl_s2, dl_t2):
            xs, xt = xs.to(device), xt.to(device)
            opt2.zero_grad()
            loss, _ = model.contrastive_alignment_loss(xs, xt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            ep_loss += loss.item()
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d}/{stage2_epochs}  "
                  f"Lc+Lkl={ep_loss/max(len(dl_s2),1):.5f}")

    # ── Stage 3: Joint fine-tuning ───────────────────────────────────────────
    if verbose:
        print(f"\n[MC-CWPDDA Stage 3] Joint fine-tuning — up to {stage3_epochs} epochs")

    model.unfreeze_all()
    opt3   = torch.optim.Adam(model.parameters(), lr=lr * 0.3)
    sched3 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt3, patience=5, factor=0.5)

    dl_s3 = DataLoader(
        TensorDataset(torch.from_numpy(X_src[:n]).float(),
                      torch.from_numpy(y_src[:n]).float()),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
    )
    dl_t3 = DataLoader(
        TensorDataset(torch.from_numpy(X_tr[:n]).float(),
                      torch.from_numpy(y_tr[:n]).float()),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
    )

    total_steps = stage3_epochs * min(len(dl_s3), len(dl_t3))
    step = 0
    best_val, best_state, no_improve = float("inf"), None, 0
    history: list[dict] = []
    start_epoch = 1

    # Resume from checkpoint if requested (Stage 3 only)
    if resume_from and Path(resume_from).is_file():
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt3.load_state_dict(ckpt["opt"])
        sched3.load_state_dict(ckpt["sched"])
        start_epoch = ckpt["epoch"] + 1
        step        = ckpt["step"]
        best_val    = ckpt["best_val"]
        history     = ckpt.get("history", [])
        if verbose:
            print(f"  Resumed Stage 3 from epoch {ckpt['epoch']} "
                  f"(best_val_mse={best_val:.5f})")

    val_bs = min(4096, max(batch_size * 16, 512))

    for epoch in range(start_epoch, stage3_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (xs, ys), (xt, yt) in zip(dl_s3, dl_t3):
            xs, ys = xs.to(device), ys.to(device)
            xt, yt = xt.to(device), yt.to(device)
            opt3.zero_grad()
            loss, _ = model.compute_loss(xs, ys, xt, yt, step, total_steps)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt3.step()
            epoch_loss += loss.item()
            step += 1
        epoch_loss /= max(len(dl_s3), 1)

        # Validation — ensure source ref is set for correct cross-attention
        model.eval()
        if model._src_ref is None:
            model.register_source_ref(X_src)
        if len(X_val) == 0:
            val_mse = float("inf")
        else:
            pred_val = model.predict_numpy_batched(X_val, device, batch_size=val_bs)
            val_mse  = float(np.mean((pred_val.squeeze() - y_val.squeeze()) ** 2))
        sched3.step(val_mse)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_mse": val_mse})

        if val_mse < best_val:
            best_val   = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            if ckpt_dir:
                torch.save(best_state, ckpt_dir / "mc_cwpdda_best.pt")
        else:
            no_improve += 1

        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch:3d}/{stage3_epochs}  "
                  f"loss={epoch_loss:.5f}  val_mse={val_mse:.5f}")

        # Recovery checkpoint (survives server timeouts)
        if ckpt_dir and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            torch.save({
                "epoch":    epoch,
                "step":     step,
                "best_val": best_val,
                "history":  history,
                "model":    {k: v.clone() for k, v in model.state_dict().items()},
                "opt":      opt3.state_dict(),
                "sched":    sched3.state_dict(),
            }, ckpt_dir / "mc_cwpdda_resume.pt")
            if verbose:
                print(f"  [ckpt] Saved recovery checkpoint at Stage-3 epoch {epoch}",
                      flush=True)

        if no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch}  best_val_mse={best_val:.5f}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # Register source reference so cross-attention works correctly at inference
    model.register_source_ref(X_src)

    if ckpt_dir:
        torch.save(model.state_dict(), ckpt_dir / "mc_cwpdda.pt")

    return {"history": history, "best_val_mse": best_val}


# ─── N-BEATS training (zero-shot: source domain only) ────────────────────────

def train_nbeats(
    model,
    data: dict,
    device: str = "cpu",
    epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 1e-3,
    patience: int = 20,
    save_dir: Optional[str] = None,
    verbose: bool = True,
    checkpoint_every: int = 10,
    resume_from: Optional[str] = None,
) -> dict:
    """
    Zero-shot N-BEATS training — Oreshkin et al., AAAI 2021.

    Uses source (Google) data only. No Alibaba (target) data is seen during
    training. At inference the trained model is applied directly to Alibaba
    windows via predict_numpy_batched().

    Loss: MSE on forecast horizon.  The paper ensemble-trains with MASE/MAPE/
    sMAPE; for simplicity we use MSE here (the difference is minor for the
    cloud workload regression task).

    checkpoint_every / resume_from: same recovery mechanism as train_cwpdda.
    """
    model = model.to(device)
    X_src, y_src = data["src_X"], data["src_y"]
    # Use target val split to monitor transfer quality during training
    X_val, y_val = data["tgt_val_X"], data["tgt_val_y"]

    dl_src = DataLoader(
        TensorDataset(torch.from_numpy(X_src).float(), torch.from_numpy(y_src).float()),
        batch_size=batch_size, shuffle=True, drop_last=False,
        pin_memory=False, num_workers=0,
    )

    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

    best_val, best_state, no_improve = float("inf"), None, 0
    history: list[dict] = []
    start_epoch = 1

    ckpt_dir = Path(save_dir) if save_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if requested.
    # Supports two formats:
    #   1. Recovery dict (nbeats_resume.pt): has "model", "opt", "sched", "epoch" keys
    #   2. Flat state dict (nbeats_best.pt): saved by torch.save(model.state_dict(), ...)
    if resume_from and Path(resume_from).is_file():
        ckpt = torch.load(resume_from, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            # Full recovery checkpoint
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            sched.load_state_dict(ckpt["sched"])
            start_epoch = ckpt["epoch"] + 1
            best_val    = ckpt["best_val"]
            history     = ckpt.get("history", [])
            if verbose:
                print(f"\n[N-BEATS] Resuming from recovery checkpoint epoch {ckpt['epoch']} "
                      f"(best_val_mse={best_val:.5f})")
        else:
            # Flat state dict (e.g. nbeats_best.pt) — load weights only, start fresh
            model.load_state_dict(ckpt)
            if verbose:
                print(f"\n[N-BEATS] Loaded weights from {resume_from} (flat state dict). "
                      f"Optimizer/scheduler reset; epoch counter starts at 1.")

    if verbose:
        print(f"\n[N-BEATS] Zero-shot training on {len(X_src):,} source windows "
              f"— device={device}")
        print(f"          n_blocks={model.n_blocks}  "
              f"shared_weights={model.shared_weights}  "
              f"epochs={epochs}  batch_size={batch_size}  lr={lr}")

    val_bs = min(4096, max(batch_size * 4, 512))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for xb, yb in dl_src:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()

        epoch_loss /= max(len(dl_src), 1)

        # Validate on target domain (Alibaba val) — measures zero-shot quality
        model.eval()
        if len(X_val) == 0:
            val_mse = float("inf")
        else:
            pred_val = model.predict_numpy_batched(X_val, device, batch_size=val_bs)
            val_mse  = float(np.mean((pred_val.squeeze() - y_val.squeeze()) ** 2))
        sched.step(val_mse)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_mse": val_mse})

        if val_mse < best_val:
            best_val   = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            if ckpt_dir:
                torch.save(best_state, ckpt_dir / "nbeats_best.pt")
        else:
            no_improve += 1

        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch:3d}/{epochs}  "
                  f"src_loss={epoch_loss:.5f}  val_mse(tgt)={val_mse:.5f}")

        # Recovery checkpoint
        if ckpt_dir and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            torch.save({
                "epoch":    epoch,
                "best_val": best_val,
                "history":  history,
                "model":    {k: v.clone() for k, v in model.state_dict().items()},
                "opt":      opt.state_dict(),
                "sched":    sched.state_dict(),
            }, ckpt_dir / "nbeats_resume.pt")
            if verbose:
                print(f"  [ckpt] Saved recovery checkpoint at epoch {epoch}", flush=True)

        if no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch}  best_val_mse={best_val:.5f}")
            break

    if best_state:
        model.load_state_dict(best_state)

    if ckpt_dir:
        torch.save(model.state_dict(), ckpt_dir / "nbeats.pt")

    return {"history": history, "best_val_mse": best_val}