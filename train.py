"""
train.py
========
Two-stage MCTL training loop.

Stage 1: Train source encoder on Google data (standard MSE regression)
Stage 2: Transfer to target via contrastive KL loss, then fine-tune head on Alibaba

Also trains all baselines from the paper for direct comparison.
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

from models.mctl import MCTL
from models.tcn import TCNPredictor


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                 shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _val_loss(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model.predict(xb) if hasattr(model, "predict") else model(xb)
            total += F.mse_loss(pred, yb, reduction="sum").item()
            n += len(xb)
    model.train()
    return total / max(n, 1)


# ─── Stage 1: source encoder pretraining ──────────────────────────────────────

def train_source_encoder(
    model: MCTL,
    src_X: np.ndarray,
    src_y: np.ndarray,
    device: str = "cpu",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    verbose: bool = True,
) -> list:
    """
    Train source TCN encoder + a temporary regression head on Google data.
    After this stage the source encoder captures workload patterns from Google.
    """
    model = model.to(device)
    model.unfreeze_source()

    # Temporary head for source pretraining
    src_head = nn.Linear(model.hidden_dim, model.regression_head.out_features).to(device)
    optimizer = torch.optim.Adam(
        list(model.source_encoder.parameters()) + list(src_head.parameters()),
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )

    loader = _make_loader(src_X, src_y, batch_size)
    losses = []

    if verbose:
        print(f"\n[Stage 1] Source encoder pretraining — {epochs} epochs")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            z = model.source_encoder(xb)
            pred = src_head(z)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        scheduler.step(epoch_loss)
        losses.append(epoch_loss)
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d}/{epochs}  loss={epoch_loss:.6f}")

    return losses


# ─── Stage 2a: contrastive transfer ───────────────────────────────────────────

def train_transfer(
    model: MCTL,
    src_X: np.ndarray,
    tgt_train_X: np.ndarray,
    device: str = "cpu",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    verbose: bool = True,
) -> list:
    """
    Stage 2a: update target encoder via contrastive KL loss.
    Source encoder is frozen. Only target encoder parameters are updated.
    """
    model = model.to(device)
    model.freeze_source()

    optimizer = torch.optim.Adam(model.target_encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n = min(len(src_X), len(tgt_train_X))
    ds_s = TensorDataset(torch.from_numpy(src_X[:n]).float())
    ds_t = TensorDataset(torch.from_numpy(tgt_train_X[:n]).float())
    dl_s = DataLoader(ds_s, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_t = DataLoader(ds_t, batch_size=batch_size, shuffle=True, drop_last=True)

    losses = []
    if verbose:
        print(f"\n[Stage 2a] Contrastive transfer — {epochs} epochs")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (xs,), (xt,) in zip(dl_s, dl_t):
            xs, xt = xs.to(device), xt.to(device)
            optimizer.zero_grad()
            loss = model.compute_transfer_loss(xs, xt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.target_encoder.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        epoch_loss /= max(len(dl_s), 1)
        losses.append(epoch_loss)
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d}/{epochs}  KL={epoch_loss:.6f}")

    return losses


# ─── Stage 2b: fine-tune regression head ─────────────────────────────────────

def train_regression_head(
    model: MCTL,
    tgt_train_X: np.ndarray,
    tgt_train_y: np.ndarray,
    tgt_val_X: np.ndarray,
    tgt_val_y: np.ndarray,
    device: str = "cpu",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    verbose: bool = True,
) -> list:
    """
    Stage 2b: train regression head on target training data.
    Target encoder is kept trainable (joint fine-tuning).
    """
    model = model.to(device)

    optimizer = torch.optim.Adam(
        list(model.target_encoder.parameters())
        + list(model.regression_head.parameters()),
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )

    train_loader = _make_loader(tgt_train_X, tgt_train_y, batch_size)
    val_loader   = _make_loader(tgt_val_X,   tgt_val_y,   batch_size, shuffle=False)

    best_val, best_state, no_improve = float("inf"), None, 0
    losses = []

    if verbose:
        print(f"\n[Stage 2b] Regression head fine-tune — {epochs} epochs")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model.predict(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        val_loss = _val_loss(model, val_loader, device)
        scheduler.step(val_loss)
        losses.append(epoch_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d}/{epochs}  train_mse={epoch_loss:.6f}  val_mse={val_loss:.6f}")

        if no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    return losses


# ─── Full MCTL training pipeline ─────────────────────────────────────────────

def train_mctl(
    data: dict,
    device: str = "cpu",
    window_size: int = 24,
    hidden_dim: int = 128,
    n_layers: int = 3,
    kernel_size: int = 3,
    dropout: float = 0.2,
    horizon: int = 1,
    stage1_epochs: int = 50,
    stage2a_epochs: int = 50,
    stage2b_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> MCTL:
    """
    Full two-stage MCTL training.
    `data` is the dict returned by preprocess.build_source_target().
    """
    model = MCTL(
        window_size=window_size,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        kernel_size=kernel_size,
        dropout=dropout,
        horizon=horizon,
    )

    t0 = time.time()

    # Stage 1
    train_source_encoder(
        model, data["src_X"], data["src_y"],
        device=device, epochs=stage1_epochs,
        batch_size=batch_size, lr=lr, verbose=verbose,
    )

    # Stage 2a
    train_transfer(
        model, data["src_X"], data["tgt_train_X"],
        device=device, epochs=stage2a_epochs,
        batch_size=batch_size, lr=lr * 0.5, verbose=verbose,
    )

    # Stage 2b
    train_regression_head(
        model,
        data["tgt_train_X"], data["tgt_train_y"],
        data["tgt_val_X"],   data["tgt_val_y"],
        device=device, epochs=stage2b_epochs,
        batch_size=batch_size, lr=lr * 0.1, verbose=verbose,
    )

    if verbose:
        print(f"\n  Total training time: {time.time() - t0:.1f}s")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(save_dir) / "mctl.pt")
        print(f"  Saved to {save_dir}/mctl.pt")

    return model
