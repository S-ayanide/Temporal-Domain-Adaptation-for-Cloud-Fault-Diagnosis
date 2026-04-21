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

        # Validation — register source ref so cross-attn works correctly
        model.eval()
        if not hasattr(model, '_src_ref') or model._src_ref is None:
            model.register_source_ref(X_src)
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

        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch:3d}/{epochs}  loss={epoch_loss:.5f}  val_mse={val_mse:.5f}")

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
    Three-stage MC-CWPDDA training curriculum.

    Stage 1 — Source pre-training:
        Train proj_src + self_attn_src + a temporary linear head on Google source
        data with MSE.  Initialises source representations with workload-predictive
        structure before any domain alignment starts.

    Stage 2 — Contrastive alignment:
        Freeze the source branch (proj_src, self_attn_src).  Train target branch,
        cross-attention, and contrastive head Gc with Lc + λ4·Lkl on paired
        (source, target) batches.

    Stage 3 — Joint fine-tuning:
        Unfreeze everything; optimise the full loss
        L = Ly + λ1·Lf + λ2·Ld + λ3·Lc + λ4·Lkl.
        Early stopping on target validation MSE.

    checkpoint_every / resume_from work the same as train_cwpdda — survives
    server timeouts by saving a full recovery checkpoint every N Stage-3 epochs.
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
    tmp_head = nn.Linear(model.extractor.out_dim, y_src.shape[1]).to(device)
    opt1 = torch.optim.Adam(
        list(model.extractor.proj_src.parameters()) +
        list(model.extractor.self_attn_src.parameters()) +
        list(tmp_head.parameters()),
        lr=lr,
    )

    dl_src = _loader(X_src, y_src, batch_size)
    for epoch in range(1, stage1_epochs + 1):
        model.train(); tmp_head.train()
        ep_loss = 0.0
        for xb, yb in dl_src:
            xb, yb = xb.to(device), yb.to(device)
            opt1.zero_grad()
            # Use source branch only: extractor(xb, xb) feeds xb as both branches
            z_shared, _, _ = model.extractor(xb, xb)
            loss = F.mse_loss(tmp_head(z_shared), yb)
            loss.backward()
            opt1.step()
            ep_loss += loss.item()
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d}/{stage1_epochs}  mse={ep_loss/max(len(dl_src),1):.5f}")

    del tmp_head  # discard temp head

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
    opt3   = torch.optim.Adam(model.parameters(), lr=lr * 0.1)
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

        # Validation
        model.eval()
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
    if ckpt_dir:
        torch.save(model.state_dict(), ckpt_dir / "mc_cwpdda.pt")

    return {"history": history, "best_val_mse": best_val}