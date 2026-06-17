"""
deepjdot/train.py
=================
Training loop for DeepJDOT (workload prediction adaptation).

Algorithm (per Algorithm 1 of Damodaran et al., ECCV 2018):

  For each minibatch of source (x_s, y_s) and target (x_t):

    Step 1 — Solve OT (network frozen):
      Compute cost matrix C_ij = α·‖z_i^s − z_j^t‖² + λ_t·MSE(y_i^s, ŷ_j^t)
      Solve: γ̂ = argmin_{γ ∈ Π(μ_s, μ_t)} ΣΣ γ_ij · C_ij
      (exact LP via ot.emd — no Sinkhorn regularisation, per paper)

    Step 2 — Update network (γ̂ frozen):
      Compute total loss = source MSE + γ̂-weighted (feature + label) terms
      Backprop and update g, f via Adam

The OT solve requires the POT library: pip install POT
Falls back to uniform coupling (equivalent to MMD feature alignment without
label propagation) if POT is not installed.
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


# ─── OT solver ────────────────────────────────────────────────────────────────

def _solve_ot(C: np.ndarray) -> np.ndarray:
    """
    Solve the exact OT problem with uniform marginals:
      min_{γ ∈ Π(1/m, 1/m)} ΣΣ γ_ij · C_ij

    Uses ot.emd (network simplex) from the POT library, matching the paper.
    Falls back to uniform γ = (1/m²) · 1_{m×m} if POT is unavailable
    (less informative — no label propagation effect, but training still runs).
    """
    m = C.shape[0]
    a = np.ones(m, dtype=np.float64) / m   # uniform source marginal
    b = np.ones(m, dtype=np.float64) / m   # uniform target marginal

    try:
        import ot
        # Clip negative values (numerical noise from GPU→CPU transfer)
        C_clean = np.clip(C, 0.0, None)
        gamma = ot.emd(a, b, C_clean)
        return gamma.astype(np.float32)
    except ImportError:
        # Uniform coupling — no OT, just MMD-like alignment
        return np.full((m, m), 1.0 / (m * m), dtype=np.float32)


# ─── Training loop ────────────────────────────────────────────────────────────

def train_deepjdot(
    model,
    data: dict,
    device: str = "cpu",
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 2e-4,
    patience: int = 20,
    save_dir: Optional[str] = None,
    verbose: bool = True,
    checkpoint_every: int = 10,
    resume_from: Optional[str] = None,
) -> dict:
    """
    Train DeepJDOT (Algorithm 1 of paper, adapted for regression).

    data keys:
        src_X, src_y        — Google source windows + labels
        tgt_train_X/y       — Alibaba train windows (labels used only for val MSE)
        tgt_val_X/y         — Alibaba val windows

    checkpoint_every / resume_from: same recovery mechanism as CWPDDA/N-BEATS.

    Batch size note: each iteration solves a (batch_size × batch_size) OT problem.
    Network simplex runs in O(m² log m); 256×256 takes ~0.01s on CPU. 512×512
    takes ~0.05s. Stay ≤ 512 unless running on a machine with fast LP solvers.
    """
    model = model.to(device)

    X_src, y_src = data["src_X"],       data["src_y"]
    X_tr,  y_tr  = data["tgt_train_X"], data["tgt_train_y"]
    X_val, y_val = data["tgt_val_X"],   data["tgt_val_y"]

    # Balance source and target — draw equal-sized minibatches from both
    n = min(len(X_src), len(X_tr))

    dl_src = DataLoader(
        TensorDataset(torch.from_numpy(X_src[:n]).float(),
                      torch.from_numpy(y_src[:n]).float()),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
    )
    dl_tgt = DataLoader(
        TensorDataset(torch.from_numpy(X_tr[:n]).float()),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)

    best_val, best_state, no_improve = float("inf"), None, 0
    history: list[dict] = []
    start_epoch = 1

    ckpt_dir = Path(save_dir) if save_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume support (same two-format detection as N-BEATS)
    if resume_from and Path(resume_from).is_file():
        ckpt = torch.load(resume_from, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            sched.load_state_dict(ckpt["sched"])
            start_epoch = ckpt["epoch"] + 1
            best_val    = ckpt["best_val"]
            history     = ckpt.get("history", [])
            if verbose:
                print(f"\n[DeepJDOT] Resuming from recovery checkpoint epoch "
                      f"{ckpt['epoch']} (best_val_mse={best_val:.5f})")
        else:
            model.load_state_dict(ckpt)
            if verbose:
                print(f"\n[DeepJDOT] Loaded weights from {resume_from} (flat state dict).")

    try:
        import ot as _ot
        ot_available = True
    except ImportError:
        ot_available = False

    if verbose:
        print(f"\n[DeepJDOT] Training — device={device}  "
              f"epochs={epochs}  batch={batch_size}  lr={lr}")
        print(f"           α={model.alpha}  λ_t={model.lambda_t}  "
              f"OT solver={'ot.emd (exact)' if ot_available else 'uniform (POT not installed)'}")
        print(f"           src={len(X_src):,}  tgt_train={len(X_tr):,}  "
              f"using {n:,} from each per epoch")

    val_bs = min(4096, max(batch_size * 8, 512))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_Ls = epoch_Lf = epoch_Ll = 0.0
        n_batches = 0

        for (x_s, y_s), (x_t,) in zip(dl_src, dl_tgt):
            x_s, y_s = x_s.to(device), y_s.to(device)
            x_t = x_t.to(device)

            # ── Step 1: Solve OT with frozen network ───────────────────────────
            with torch.no_grad():
                z_s      = model.encode(x_s)               # (m, d)
                z_t      = model.encode(x_t)               # (m, d)
                y_hat_t  = model.predictor(z_t)            # (m, H)
                C        = model.compute_cost_matrix(z_s, z_t, y_s, y_hat_t)
                C_np     = C.cpu().numpy().astype(np.float64)

            gamma_np = _solve_ot(C_np)
            gamma    = torch.from_numpy(gamma_np).float().to(device)

            # ── Step 2: Update network with γ fixed ────────────────────────────
            opt.zero_grad()
            loss, info = model.compute_loss(x_s, y_s, x_t, gamma)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            epoch_Ls   += info["L_src"]
            epoch_Lf   += info["L_feat"]
            epoch_Ll   += info["L_label"]
            n_batches  += 1

        nb = max(n_batches, 1)
        epoch_loss /= nb; epoch_Ls /= nb; epoch_Lf /= nb; epoch_Ll /= nb

        # Validate on Alibaba val split
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
                torch.save(best_state, ckpt_dir / "deepjdot_best.pt")
        else:
            no_improve += 1

        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d}/{epochs}  "
                  f"loss={epoch_loss:.5f}  "
                  f"Ls={epoch_Ls:.5f}  Lf={epoch_Lf:.4f}  Ll={epoch_Ll:.5f}  "
                  f"val_mse={val_mse:.5f}")

        if ckpt_dir and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            torch.save({
                "epoch":    epoch,
                "best_val": best_val,
                "history":  history,
                "model":    {k: v.clone() for k, v in model.state_dict().items()},
                "opt":      opt.state_dict(),
                "sched":    sched.state_dict(),
            }, ckpt_dir / "deepjdot_resume.pt")
            if verbose:
                print(f"  [ckpt] Saved recovery checkpoint at epoch {epoch}", flush=True)

        if no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch}  best_val_mse={best_val:.5f}")
            break

    if best_state:
        model.load_state_dict(best_state)

    if ckpt_dir:
        torch.save(model.state_dict(), ckpt_dir / "deepjdot.pt")

    return {"history": history, "best_val_mse": best_val}
