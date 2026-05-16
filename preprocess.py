"""
preprocess.py
=============
Preprocessing for both CWPDDA and MCTL.

Key fixes vs original:
  1. CWPDDA does NOT filter by series length — it uses all machines.
     The paper says "containers meeting small sample condition" but the
     architecture works on any length. Filtering to MAX_TARGET_LEN=100
     was discarding 99% of Alibaba data and leaving only 734 short series.

  2. DTW is for MCTL only (source selection for few-shot transfer).
     For CWPDDA, skip DTW entirely — use all Google series as source.

  3. Source/target balance: cap both at MAX_WINDOWS_PER_DOMAIN so the
     training loop sees a balanced dataset. 162k source vs 2.9M target
     means the model trains mostly on target patterns without enough
     source signal for domain alignment.

  4. WINDOW_STEP increased to 5 (was 1) to reduce redundancy.
     Step=1 on a 4000-point series gives 3977 nearly-identical windows.
     Step=5 gives ~795 more diverse windows per machine.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from numpy.typing import NDArray


# Keys saved in preprocess bundle (.npz + .json sidecar) for run.py --load-cache
_PREPROCESS_ARRAY_KEYS = (
    "src_X",
    "src_y",
    "tgt_train_X",
    "tgt_train_y",
    "tgt_val_X",
    "tgt_val_y",
    "tgt_test_X",
    "tgt_test_y",
)


def save_preprocess_cache(cache_path: str | Path, data: dict) -> None:
    """
    Save arrays to ``*.npz`` and ``meta`` + ``dtw_pairs`` to a sibling ``*.json``.
    Same basename: ``foo.npz`` ↔ ``foo.json``.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(data[k]) for k in _PREPROCESS_ARRAY_KEYS}
    np.savez_compressed(cache_path, **arrays)
    side = cache_path.with_suffix(".json")
    payload = {
        "meta": data["meta"],
        "dtw_pairs": [list(p) for p in data["dtw_pairs"]],
    }
    side.write_text(json.dumps(payload, indent=2))


def load_preprocess_cache(cache_path: str | Path) -> dict:
    """Load dict compatible with :func:`build_source_target` output."""
    cache_path = Path(cache_path)
    side = cache_path.with_suffix(".json")
    if not cache_path.is_file():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    if not side.is_file():
        raise FileNotFoundError(
            f"Expected sidecar JSON next to cache: {side}\n"
            "(Save with --save-cache first; both .npz and .json are required.)"
        )
    payload = json.loads(side.read_text())
    bundle = np.load(cache_path, allow_pickle=False)
    try:
        out = {k: bundle[k] for k in _PREPROCESS_ARRAY_KEYS}
    finally:
        bundle.close()
    out["meta"] = payload["meta"]
    out["dtw_pairs"] = [tuple(p) for p in payload["dtw_pairs"]]
    return out


# ─── Configuration ────────────────────────────────────────────────────────────

WINDOW_SIZE   = 24      # input window length  (matches both papers)
PRED_HORIZON  = 1       # predict 1 step ahead
WINDOW_STEP   = 5       # stride — step=1 creates millions of near-duplicate windows

# CWPDDA uses all series regardless of length
MIN_SOURCE_LEN = 30     # drop trivially short Google series (< 30 points)

# Max windows per domain — keeps source/target roughly balanced
# and prevents 3M-window datasets that never converge in reasonable time
MAX_WINDOWS_PER_DOMAIN = 200_000

# MCTL-specific: length thresholds for few-shot setup
MCTL_MIN_SOURCE_LEN = 80   # Google: must be a "long" job
MCTL_MAX_TARGET_LEN = 100  # Alibaba: must be a "short" job (few-shot)

TRAIN_RATIO = 0.7   # CWPDDA paper uses 70/20/10
VAL_RATIO   = 0.2


# ─── Normalisation ─────────────────────────────────────────────────────────────

def normalise(series: NDArray) -> NDArray:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-6:
        return np.zeros_like(series, dtype=np.float32)
    return ((series - lo) / (hi - lo)).astype(np.float32)


def normalise_all(series_list: List[NDArray]) -> List[NDArray]:
    return [normalise(s) for s in series_list]


# ─── DTW (for MCTL source selection) ──────────────────────────────────────────

def dtw_distance(a: NDArray, b: NDArray, band: int = 10) -> float:
    n, m = len(a), len(b)
    cost = np.full((n, m), np.inf, dtype=np.float64)
    cost[0, 0] = abs(float(a[0]) - float(b[0]))
    for i in range(1, n):
        cost[i, 0] = cost[i-1, 0] + abs(float(a[i]) - float(b[0]))
    for j in range(1, m):
        cost[0, j] = cost[0, j-1] + abs(float(a[0]) - float(b[j]))
    for i in range(1, n):
        j_lo = max(1, i - band)
        j_hi = min(m, i + band + 1)
        for j in range(j_lo, j_hi):
            d = abs(float(a[i]) - float(b[j]))
            cost[i, j] = d + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
    return float(cost[n-1, m-1])


def _subsample(s: NDArray, n: int) -> NDArray:
    if len(s) <= n:
        return s
    return s[np.linspace(0, len(s)-1, n, dtype=int)]


def select_source_by_dtw(target: NDArray, sources: List[NDArray],
                          top_k: int = 1, subsample_len: int = 50) -> List[int]:
    t_sub = _subsample(target, subsample_len)
    dists = sorted(
        [(dtw_distance(t_sub, _subsample(src, subsample_len)), i)
         for i, src in enumerate(sources)]
    )
    return [i for _, i in dists[:top_k]]


# ─── Windowing ────────────────────────────────────────────────────────────────

def make_windows(series: NDArray, window_size: int = WINDOW_SIZE,
                 horizon: int = PRED_HORIZON,
                 step: int = WINDOW_STEP) -> Tuple[NDArray, NDArray]:
    total = window_size + horizon
    if len(series) < total:
        return (np.empty((0, window_size), dtype=np.float32),
                np.empty((0, horizon), dtype=np.float32))
    xs, ys = [], []
    for start in range(0, len(series) - total + 1, step):
        xs.append(series[start : start + window_size])
        ys.append(series[start + window_size : start + window_size + horizon])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def make_windows_all(series_list: List[NDArray], window_size: int = WINDOW_SIZE,
                     horizon: int = PRED_HORIZON,
                     step: int = WINDOW_STEP,
                     max_windows: Optional[int] = None) -> Tuple[NDArray, NDArray]:
    all_x, all_y = [], []
    for s in series_list:
        x, y = make_windows(s, window_size, horizon, step)
        if len(x):
            all_x.append(x)
            all_y.append(y)
    if not all_x:
        return (np.empty((0, window_size), dtype=np.float32),
                np.empty((0, horizon), dtype=np.float32))
    X = np.concatenate(all_x)
    Y = np.concatenate(all_y)
    if max_windows and len(X) > max_windows:
        idx = np.random.default_rng(42).choice(len(X), max_windows, replace=False)
        idx.sort()
        X, Y = X[idx], Y[idx]
    return X, Y


def temporal_split_series(series: NDArray, train_ratio: float = TRAIN_RATIO,
                           val_ratio: float = VAL_RATIO) -> Tuple[NDArray, NDArray, NDArray]:
    n = len(series)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return series[:n_train], series[n_train:n_train+n_val], series[n_train+n_val:]


# ─── CWPDDA pipeline ──────────────────────────────────────────────────────────

def build_source_target(
    google_series: List[NDArray],
    alibaba_series: List[NDArray],
    window_size: int = WINDOW_SIZE,
    horizon: int = PRED_HORIZON,
    use_dtw: bool = False,        # CWPDDA does not use DTW
    dtw_top_k: int = 1,
    seed: int = 42,
    max_windows: int = MAX_WINDOWS_PER_DOMAIN,
    window_step: int = WINDOW_STEP,
    max_target_len: int = 0,      # 0 = no filter; >0 = keep only short-series targets
) -> dict:
    """
    Preprocessing pipeline for CWPDDA (and optionally MCTL).

    CWPDDA setup (Wang et al., Euro-Par 2025):
      - Source = ALL Google series (no length filter — the paper uses
        all available Google containers, not just long ones)
      - Target = ALL Alibaba machines (no length filter)
      - No DTW — domain alignment is handled by the GRL during training
      - Windows capped at max_windows per domain for balance

      --max-target-len N (optional, recommended for CWPDDA):
        The paper uses "containers meeting small sample condition" — only
        short-lived Alibaba containers (few data points).  With all ~200k
        target windows, a plain LSTM has enough data and transfer learning
        doesn't help.  Setting max_target_len=200 replicates the few-shot
        regime where Google knowledge is genuinely needed.

    MCTL setup (Zuo et al., 2024) — set use_dtw=True:
      - Source = Google series >= MCTL_MIN_SOURCE_LEN
      - Target = Alibaba series <= MCTL_MAX_TARGET_LEN  (few-shot)
      - DTW selects the most similar source for each target
    """
    rng = np.random.default_rng(seed)

    print(f"\n[preprocess] Filtering series...")

    if use_dtw:
        # MCTL: length-based filtering
        src_raw = [s for s in google_series  if len(s) >= MCTL_MIN_SOURCE_LEN]
        tgt_raw = [s for s in alibaba_series if len(s) <= MCTL_MAX_TARGET_LEN]
        if not tgt_raw:
            print(f"  No Alibaba series <= {MCTL_MAX_TARGET_LEN} pts. Using all.")
            tgt_raw = alibaba_series[:]
        if not src_raw:
            src_raw = google_series[:]
    else:
        # CWPDDA: use everything (just drop trivially short series)
        src_raw = [s for s in google_series  if len(s) >= MIN_SOURCE_LEN]
        if max_target_len > 0:
            tgt_raw = [s for s in alibaba_series if len(s) <= max_target_len]
            if not tgt_raw:
                print(f"  [warn] No Alibaba series <= {max_target_len} pts; using all.")
                tgt_raw = alibaba_series[:]
            else:
                print(f"  Few-shot filter: kept {len(tgt_raw)} Alibaba series "
                      f"(of {len(alibaba_series)}) with <= {max_target_len} pts")
        else:
            tgt_raw = alibaba_series[:]
        if not src_raw:
            src_raw = google_series[:]

    print(f"  Source (Google): {len(src_raw)} series")
    print(f"  Target (Alibaba): {len(tgt_raw)} series")

    print(f"[preprocess] Normalising...")
    src_norm = normalise_all(src_raw)
    tgt_norm = normalise_all(tgt_raw)

    # --- DTW (MCTL only) ---
    dtw_pairs: List[Tuple[int, int]] = []
    if use_dtw and src_norm and tgt_norm:
        print(f"[preprocess] DTW source selection (top_k={dtw_top_k})...")
        selected: set = set()
        for t_idx, tgt_s in enumerate(tgt_norm):
            best = select_source_by_dtw(tgt_s, src_norm, top_k=dtw_top_k)
            for s_idx in best:
                dtw_pairs.append((t_idx, s_idx))
                selected.add(s_idx)
            if (t_idx + 1) % 100 == 0:
                print(f"  DTW: {t_idx+1}/{len(tgt_norm)}")
        src_norm = [src_norm[i] for i in sorted(selected)]
        print(f"  {len(src_norm)} unique sources selected")
    else:
        dtw_pairs = [(t, 0) for t in range(len(tgt_norm))]

    # --- Windowing ---
    print(f"[preprocess] Windowing  (W={window_size}, step={window_step}, "
          f"cap={max_windows:,} per domain)...")

    src_X, src_y = make_windows_all(src_norm, window_size, horizon,
                                     window_step, max_windows)

    # Target: temporal split per series, then window
    tgt_train_s, tgt_val_s, tgt_test_s = [], [], []
    for s in tgt_norm:
        tr, va, te = temporal_split_series(s, TRAIN_RATIO, VAL_RATIO)
        if len(tr) >= window_size + horizon: tgt_train_s.append(tr)
        if len(va) >= window_size + horizon: tgt_val_s.append(va)
        if len(te) >= window_size + horizon: tgt_test_s.append(te)

    tgt_train_X, tgt_train_y = make_windows_all(tgt_train_s, window_size, horizon,
                                                  window_step, max_windows)
    tgt_val_X,   tgt_val_y   = make_windows_all(tgt_val_s,   window_size, horizon,
                                                  window_step, max_windows // 3)
    tgt_test_X,  tgt_test_y  = make_windows_all(tgt_test_s,  window_size, horizon,
                                                  window_step, max_windows // 3)

    meta = {
        "n_src_series":      len(src_norm),
        "n_tgt_series":      len(tgt_norm),
        "src_windows":       int(len(src_X)),
        "tgt_train_windows": int(len(tgt_train_X)),
        "tgt_val_windows":   int(len(tgt_val_X)),
        "tgt_test_windows":  int(len(tgt_test_X)),
        "window_size":       window_size,
        "horizon":           horizon,
        "window_step":       window_step,
        "cache_spec": {
            "max_windows":    max_windows,
            "use_dtw":        use_dtw,
            "window_size":    window_size,
            "horizon":        horizon,
            "max_target_len": max_target_len,
        },
    }

    print(f"[preprocess] Done.")
    print(f"  src windows:        {meta['src_windows']:>10,}")
    print(f"  tgt train windows:  {meta['tgt_train_windows']:>10,}")
    print(f"  tgt val windows:    {meta['tgt_val_windows']:>10,}")
    print(f"  tgt test windows:   {meta['tgt_test_windows']:>10,}")

    ratio = meta['src_windows'] / max(meta['tgt_train_windows'], 1)
    if ratio < 0.1:
        print(f"\n  [warn] src/tgt ratio = {ratio:.3f} — source data is much smaller "
              f"than target. Consider --no-dtw or loading more Google shards.")

    return {
        "src_X": src_X,   "src_y": src_y,
        "tgt_train_X": tgt_train_X, "tgt_train_y": tgt_train_y,
        "tgt_val_X":   tgt_val_X,   "tgt_val_y":   tgt_val_y,
        "tgt_test_X":  tgt_test_X,  "tgt_test_y":  tgt_test_y,
        "dtw_pairs": dtw_pairs,
        "meta": meta,
    }