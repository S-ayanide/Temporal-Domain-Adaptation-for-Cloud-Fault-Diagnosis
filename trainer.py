"""
Unified training engine — handles both flat (DANN/CDAN/FixBi/ToAlign/DATL)
and temporal (TA-DATL) models through a common interface.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from models import grl_alpha

logger  = logging.getLogger(__name__)
DEVICE  = torch.device("cuda")


# ── Tensor helpers ────────────────────────────────────────────────────────────

def to_tensor(arr, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype).to(DEVICE)


def make_loader(X, y=None, batch_size=256, shuffle=True):
    Xt = to_tensor(X)
    ds = TensorDataset(Xt, to_tensor(y, torch.long)) if y is not None \
         else TensorDataset(Xt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=False)


def next_batch(it, loader):
    try:
        return next(it)
    except StopIteration:
        return next(iter(loader))


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X, y, n_classes):
    model.eval()
    with torch.no_grad():
        logits = model.predict(to_tensor(X))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(1)
    acc   = accuracy_score(y, preds)
    f1    = f1_score(y, preds, average="macro", zero_division=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            auc = roc_auc_score(y, probs, multi_class="ovr",
                                average="macro", labels=list(range(n_classes))) \
                  if n_classes > 2 else roc_auc_score(y, probs[:, 1])
        except ValueError:
            auc = float("nan")
    return {"accuracy": acc, "f1": f1, "auc": auc}


# ── TA-DATL trainer (temporal model) ─────────────────────────────────────────

def train_ta_datl(model, X_src, y_src, X_tgt, y_tgt, tgt_labeled,
                  n_classes, epochs=150, lr=1e-3, batch_size=128,
                  pseudo_freq=10, save_path=None):
    """
    X_src / X_tgt : (N, W, F)  temporal windows
    """
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_ul = X_tgt[~tgt_labeled]
    X_lb = X_tgt[tgt_labeled];  y_lb = y_tgt[tgt_labeled]
    if len(X_ul) == 0: X_ul = X_lb

    X_src_w, y_src_w = X_src.copy(), y_src.copy()
    best_f1, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        alpha = grl_alpha(epoch, epochs)

        # Pseudo-label refresh
        if epoch % pseudo_freq == 0 and epoch > 15:
            model.eval()
            _, pl_y, mask = model.get_pseudo_labels(to_tensor(X_ul))
            if mask.sum() > 0:
                X_src_w = np.concatenate([X_src, X_ul[mask.cpu().numpy()]])
                y_src_w = np.concatenate([y_src, pl_y.cpu().numpy()])
            model.train()

        src_loader = make_loader(X_src_w, y_src_w, batch_size)
        tgt_iter   = iter(make_loader(X_ul, batch_size=batch_size))

        for x_s, y_s in src_loader:
            x_t = next_batch(tgt_iter, make_loader(X_ul, batch_size=batch_size))[0]
            n   = min(x_s.size(0), x_t.size(0))
            x_s, y_s, x_t = x_s[:n], y_s[:n], x_t[:n]

            opt.zero_grad()
            cl, ds, dt, fs, ft = model(x_s, x_t, alpha)
            loss, _ = model.compute_loss(cl, y_s, ds, dt, fs, ft)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sch.step()
        m = evaluate(model, X_lb, y_lb, n_classes)
        if m["f1"] > best_f1:
            best_f1    = m["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 30 == 0 or epoch == epochs:
            logger.info(f"  [TA-DATL] epoch {epoch:3d}/{epochs} | "
                        f"acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
                        f"auc={m['auc']:.4f}  T={torch.exp(model.log_T).item():.3f}")

    if best_state: model.load_state_dict(best_state)
    if save_path:  torch.save(model.state_dict(), save_path)
    return evaluate(model, X_lb, y_lb, n_classes)


# ── Generic adversarial trainer (DANN / CDAN / ToAlign / DATL) ───────────────

def train_adversarial(model, name, X_src, y_src, X_tgt, y_tgt, tgt_labeled,
                      n_classes, epochs=150, lr=1e-3, batch_size=256,
                      save_path=None):
    """Works for flat (N,F) inputs."""
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_ul = X_tgt[~tgt_labeled];  X_lb = X_tgt[tgt_labeled];  y_lb = y_tgt[tgt_labeled]
    if len(X_ul) == 0: X_ul = X_lb

    # Pseudo-label support for DATL
    has_pseudo = hasattr(model, "get_pseudo_labels")
    X_sw, y_sw = X_src.copy(), y_src.copy()
    best_f1, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        alpha = grl_alpha(epoch, epochs)

        if has_pseudo and epoch % 10 == 0 and epoch > 15:
            model.eval()
            _, pl_y, mask = model.get_pseudo_labels(to_tensor(X_ul))
            if mask.sum() > 0:
                X_sw = np.vstack([X_src, X_ul[mask.cpu().numpy()]])
                y_sw = np.concatenate([y_src, pl_y.cpu().numpy()])
            model.train()

        src_loader = make_loader(X_sw, y_sw, batch_size)
        tgt_iter   = iter(make_loader(X_ul, batch_size=batch_size))

        for x_s, y_s in src_loader:
            x_t = next_batch(tgt_iter, make_loader(X_ul, batch_size=batch_size))[0]
            n   = min(x_s.size(0), x_t.size(0))
            x_s, y_s, x_t = x_s[:n], y_s[:n], x_t[:n]

            opt.zero_grad()
            out  = model(x_s, x_t, alpha)
            loss, _ = model.compute_loss(*out[:2], *out[2:4], y_s, *out[4:])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sch.step()
        m = evaluate(model, X_lb, y_lb, n_classes)
        if m["f1"] > best_f1:
            best_f1    = m["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 30 == 0 or epoch == epochs:
            logger.info(f"  [{name}] epoch {epoch:3d}/{epochs} | "
                        f"acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
                        f"auc={m['auc']:.4f}")

    if best_state: model.load_state_dict(best_state)
    if save_path:  torch.save(model.state_dict(), save_path)
    return evaluate(model, X_lb, y_lb, n_classes)


# ── FixBi trainer ─────────────────────────────────────────────────────────────

def train_fixbi(model, X_src, y_src, X_tgt, y_tgt, tgt_labeled,
                n_classes, epochs=150, lr=1e-3, batch_size=256, save_path=None):
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_ul = X_tgt[~tgt_labeled];  X_lb = X_tgt[tgt_labeled];  y_lb = y_tgt[tgt_labeled]
    if len(X_ul) == 0: X_ul = X_lb

    best_f1, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        src_loader = make_loader(X_src, y_src, batch_size)
        tgt_iter   = iter(make_loader(X_ul, batch_size=batch_size))

        for x_s, y_s in src_loader:
            x_t = next_batch(tgt_iter, make_loader(X_ul, batch_size=batch_size))[0]
            n   = min(x_s.size(0), x_t.size(0))
            x_s, y_s, x_t = x_s[:n], y_s[:n], x_t[:n]

            opt.zero_grad()
            ls_s, lt_s, ls_t, lt_t, _, _ = model(x_s, x_t)
            loss, _ = model.compute_loss(ls_s, lt_s, ls_t, lt_t, y_s)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sch.step()
        m = evaluate(model, X_lb, y_lb, n_classes)
        if m["f1"] > best_f1:
            best_f1    = m["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 30 == 0 or epoch == epochs:
            logger.info(f"  [FixBi] epoch {epoch:3d}/{epochs} | "
                        f"acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
                        f"auc={m['auc']:.4f}")

    if best_state: model.load_state_dict(best_state)
    if save_path:  torch.save(model.state_dict(), save_path)
    return evaluate(model, X_lb, y_lb, n_classes)
