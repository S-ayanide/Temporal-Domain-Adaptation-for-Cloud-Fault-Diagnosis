"""
preprocess.py
=============
Follows the MCTL paper's preprocessing exactly:

1. Linear normalisation per series to [0, 1]
2. DTW-based source selection: for each target (short) series, find the
   most similar source (long) series from Google
3. Sliding window extraction with configurable window size and step
4. Split into train / val / test

Paper definitions (Section 3.1 + 3.2):
  - Source domain: Google series with len >= MIN_SOURCE_LEN  (long jobs)
  - Target domain: Alibaba series with len < MAX_TARGET_LEN  (short jobs)
  - "Short task sequences defined as batch jobs running < 8h"
    → at 5-min sampling = 96 points; we use that as the threshold
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from numpy.typing import NDArray


_ARRAY_KEYS = (
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
    Write preprocessed arrays to ``*.npz`` and ``meta`` + ``dtw_pairs`` to ``*.json``
    (same basename, ``.npz`` → ``.json``).
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(data[k]) for k in _ARRAY_KEYS}
    np.savez_compressed(cache_path, **arrays)
    side = cache_path.with_suffix(".json")
    payload = {
        "meta": data["meta"],
        "dtw_pairs": [list(p) for p in data["dtw_pairs"]],
    }
    side.write_text(json.dumps(payload, indent=2))


def load_preprocess_cache(cache_path: str | Path) -> dict:
    """Load dict produced by :func:`save_preprocess_cache`."""
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
        data = {k: bundle[k] for k in _ARRAY_KEYS}
    finally:
        bundle.close()
    data["meta"] = payload["meta"]
    data["dtw_pairs"] = [tuple(p) for p in payload["dtw_pairs"]]
    return data


# ─── Configuration (matches MCTL paper) ──────────────────────────────────────

WINDOW_SIZE   = 24      # W: input sequence length (past 24 steps)
PRED_HORIZON  = 1       # predict 1 step ahead (matches paper's evaluation setup)
WINDOW_STEP   = 1       # stride between windows

MIN_SOURCE_LEN = 80     # Google series must have >= 80 points to be a source
MAX_TARGET_LEN = 100    # Alibaba series with <= 100 points = "short task" (few-shot)
                        # Paper says <96, we use 100 to be slightly generous

DTW_N_CANDIDATES = 5    # top-k sources selected per target (paper uses DTW to pick 1)

TRAIN_RATIO = 0.6
VAL_RATIO   = 0.2
# TEST = remaining 0.2


# ─── Normalisation ────────────────────────────────────────────────────────────

def normalise(series: NDArray) -> NDArray:
    """Linear normalisation to [0, 1]. Paper eq: x_norm = (x - min) / (max - min)."""
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-6:
        return np.zeros_like(series, dtype=np.float32)
    return ((series - lo) / (hi - lo)).astype(np.float32)


def normalise_all(series_list: List[NDArray]) -> List[NDArray]:
    return [normalise(s) for s in series_list]


# ─── DTW (fast, sakoe-chiba band) ─────────────────────────────────────────────

def dtw_distance(a: NDArray, b: NDArray, band: int = 10) -> float:
    """
    Sakoe-Chiba banded DTW between two 1-D series.
    Faster than full DTW; band width follows common practice.
    """
    n, m = len(a), len(b)
    cost = np.full((n, m), np.inf, dtype=np.float64)
    cost[0, 0] = abs(float(a[0]) - float(b[0]))
    for i in range(1, n):
        cost[i, 0] = cost[i - 1, 0] + abs(float(a[i]) - float(b[0]))
    for j in range(1, m):
        cost[0, j] = cost[0, j - 1] + abs(float(a[0]) - float(b[j]))
    for i in range(1, n):
        j_lo = max(1, i - band)
        j_hi = min(m, i + band + 1)
        for j in range(j_lo, j_hi):
            d = abs(float(a[i]) - float(b[j]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n - 1, m - 1])


def select_source_by_dtw(
    target: NDArray,
    sources: List[NDArray],
    top_k: int = 1,
    subsample_len: int = 50,
) -> List[int]:
    """
    For a given target series, return the indices of the top_k most
    similar source series (lowest DTW distance).

    We subsample both to subsample_len for speed.
    """
    t_sub = _subsample(target, subsample_len)
    dists = []
    for idx, src in enumerate(sources):
        s_sub = _subsample(src, subsample_len)
        dists.append((dtw_distance(t_sub, s_sub), idx))
    dists.sort(key=lambda x: x[0])
    return [idx for _, idx in dists[:top_k]]


def _subsample(s: NDArray, n: int) -> NDArray:
    if len(s) <= n:
        return s
    idx = np.linspace(0, len(s) - 1, n, dtype=int)
    return s[idx]


# ─── Windowing ────────────────────────────────────────────────────────────────

def make_windows(
    series: NDArray,
    window_size: int = WINDOW_SIZE,
    horizon: int = PRED_HORIZON,
    step: int = WINDOW_STEP,
) -> Tuple[NDArray, NDArray]:
    """
    Sliding window over a 1-D series.

    Returns:
        X: (N, window_size)  — input windows
        y: (N, horizon)      — prediction targets
    """
    total = window_size + horizon
    if len(series) < total:
        return np.empty((0, window_size), dtype=np.float32), np.empty((0, horizon), dtype=np.float32)

    xs, ys = [], []
    for start in range(0, len(series) - total + 1, step):
        xs.append(series[start : start + window_size])
        ys.append(series[start + window_size : start + window_size + horizon])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def make_windows_all(
    series_list: List[NDArray],
    window_size: int = WINDOW_SIZE,
    horizon: int = PRED_HORIZON,
    step: int = WINDOW_STEP,
) -> Tuple[NDArray, NDArray]:
    """Apply make_windows to every series and concatenate."""
    all_x, all_y = [], []
    for s in series_list:
        x, y = make_windows(s, window_size, horizon, step)
        if len(x):
            all_x.append(x)
            all_y.append(y)
    if not all_x:
        return np.empty((0, window_size), dtype=np.float32), np.empty((0, horizon), dtype=np.float32)
    return np.concatenate(all_x), np.concatenate(all_y)


# ─── Train / val / test split ─────────────────────────────────────────────────

def split_series(
    series_list: List[NDArray],
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    seed: int = 42,
) -> Tuple[List[NDArray], List[NDArray], List[NDArray]]:
    """Split list of series into train / val / test by series (not by timestep)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(series_list))
    n = len(series_list)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = [series_list[i] for i in idx[:n_train]]
    val   = [series_list[i] for i in idx[n_train : n_train + n_val]]
    test  = [series_list[i] for i in idx[n_train + n_val :]]
    return train, val, test


def temporal_split_series(series: NDArray, train_ratio: float = TRAIN_RATIO,
                           val_ratio: float = VAL_RATIO) -> Tuple[NDArray, NDArray, NDArray]:
    """Split a single series temporally (no shuffle) for time-series correctness."""
    n = len(series)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return series[:n_train], series[n_train:n_train + n_val], series[n_train + n_val:]


# ─── Full pipeline ────────────────────────────────────────────────────────────

def build_source_target(
    google_series: List[NDArray],
    alibaba_series: List[NDArray],
    window_size: int = WINDOW_SIZE,
    horizon: int = PRED_HORIZON,
    use_dtw: bool = True,
    dtw_top_k: int = 1,
    seed: int = 42,
) -> dict:
    """
    Full preprocessing pipeline matching the MCTL paper.

    Steps:
        1. Filter: source = Google long series; target = Alibaba short series
        2. Normalise each series to [0, 1]
        3. DTW-select best source for each target (optional but recommended)
        4. Window all series
        5. Split target into train / val / test

    Returns a dict with keys:
        src_X, src_y         — all source windows (training only)
        tgt_train_X, tgt_train_y
        tgt_val_X,   tgt_val_y
        tgt_test_X,  tgt_test_y
        dtw_pairs            — list of (tgt_idx, src_idx) pairs
        meta                 — stats dict
    """
    print(f"\n[preprocess] Filtering series...")
    src_raw = [s for s in google_series  if len(s) >= MIN_SOURCE_LEN]
    tgt_raw = [s for s in alibaba_series if len(s) <= MAX_TARGET_LEN]

    # Fallback: if all Alibaba series are long, take all of them
    if not tgt_raw:
        print(f"  Warning: no Alibaba series <= {MAX_TARGET_LEN} points. Using all.")
        tgt_raw = alibaba_series[:]

    if not src_raw:
        print(f"  Warning: no Google series >= {MIN_SOURCE_LEN} points. Using all.")
        src_raw = google_series[:]

    print(f"  Source (Google long): {len(src_raw)} series")
    print(f"  Target (Alibaba short): {len(tgt_raw)} series")

    print(f"[preprocess] Normalising...")
    src_norm = normalise_all(src_raw)
    tgt_norm = normalise_all(tgt_raw)

    # --- DTW source selection ---
    dtw_pairs: List[Tuple[int, int]] = []
    selected_src_indices: set = set()

    if use_dtw and src_norm and tgt_norm:
        print(f"[preprocess] DTW source selection (top_k={dtw_top_k})...")
        for t_idx, tgt_s in enumerate(tgt_norm):
            best = select_source_by_dtw(tgt_s, src_norm, top_k=dtw_top_k)
            for s_idx in best:
                dtw_pairs.append((t_idx, s_idx))
                selected_src_indices.add(s_idx)
            if (t_idx + 1) % 100 == 0:
                print(f"  DTW: {t_idx + 1}/{len(tgt_norm)}")
        print(f"  {len(selected_src_indices)} unique source series selected by DTW")
        src_for_training = [src_norm[i] for i in sorted(selected_src_indices)]
    else:
        src_for_training = src_norm
        dtw_pairs = [(t, 0) for t in range(len(tgt_norm))]  # dummy

    # --- Windowing ---
    print(f"[preprocess] Windowing (W={window_size}, H={horizon})...")
    src_X, src_y = make_windows_all(src_for_training, window_size, horizon)

    # Split target temporally per series, then window
    tgt_train_s, tgt_val_s, tgt_test_s = [], [], []
    for s in tgt_norm:
        tr, va, te = temporal_split_series(s)
        if len(tr) >= window_size + horizon: tgt_train_s.append(tr)
        if len(va) >= window_size + horizon: tgt_val_s.append(va)
        if len(te) >= window_size + horizon: tgt_test_s.append(te)

    tgt_train_X, tgt_train_y = make_windows_all(tgt_train_s, window_size, horizon)
    tgt_val_X,   tgt_val_y   = make_windows_all(tgt_val_s,   window_size, horizon)
    tgt_test_X,  tgt_test_y  = make_windows_all(tgt_test_s,  window_size, horizon)

    meta = {
        "n_src_series": len(src_for_training),
        "n_tgt_series": len(tgt_norm),
        "src_windows": len(src_X),
        "tgt_train_windows": len(tgt_train_X),
        "tgt_val_windows": len(tgt_val_X),
        "tgt_test_windows": len(tgt_test_X),
        "window_size": window_size,
        "horizon": horizon,
    }

    print(f"[preprocess] Done.")
    print(f"  src windows:        {meta['src_windows']:,}")
    print(f"  tgt train windows:  {meta['tgt_train_windows']:,}")
    print(f"  tgt val windows:    {meta['tgt_val_windows']:,}")
    print(f"  tgt test windows:   {meta['tgt_test_windows']:,}")

    return {
        "src_X": src_X,   "src_y": src_y,
        "tgt_train_X": tgt_train_X, "tgt_train_y": tgt_train_y,
        "tgt_val_X":   tgt_val_X,   "tgt_val_y":   tgt_val_y,
        "tgt_test_X":  tgt_test_X,  "tgt_test_y":  tgt_test_y,
        "dtw_pairs": dtw_pairs,
        "meta": meta,
    }
