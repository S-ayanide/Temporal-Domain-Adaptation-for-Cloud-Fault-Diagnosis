"""
similarity.py — Source domain selection for Tr-Predictor (Liu et al., Entropy 2022).

Two metrics are combined to rank candidate source domains:
  1. TWED  (Time Warp Edit Distance)   — smaller = more similar
  2. TE    (Transfer Entropy)           — larger = more information transfer

Reference:
  Liu et al. "Tr-Predictor: An Ensemble Transfer Learning Model for
  Small-Sample Cloud Workload Prediction", Entropy 2022, 24, 742.
  https://doi.org/10.3390/e24060742
"""

import numpy as np
from scipy.stats import rankdata


# ---------------------------------------------------------------------------
# 1. TWED — Time Warp Edit Distance
# ---------------------------------------------------------------------------

def twed(ts1: np.ndarray, ts2: np.ndarray,
         lam: float = 0.5, nu: float = 0.001) -> float:
    """
    Compute TWED between two univariate time series.

    TWED is a metric on time series that allows elastic matching
    while penalising time-stamp differences (stiffness nu) and
    deletions (penalty lambda).

    Parameters
    ----------
    ts1, ts2 : 1-D float arrays
        The two time series (need not be the same length).
    lam : float
        Deletion cost (λ in the paper). Default 0.5.
    nu : float
        Stiffness / time penalty (ν in the paper). Default 0.001.

    Returns
    -------
    float
        TWED distance (lower = more similar).

    Notes
    -----
    Implementation follows the DP recurrence in Eq. (1) of the paper.
    Time stamps are assumed to be integer indices 0, 1, …, n-1.
    """
    n = len(ts1)
    m = len(ts2)

    # DP table — uses (n+1) × (m+1) with Inf padding
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0

    # Init first column: delete all of ts1[0..i-1]
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + ts1[i - 1] + lam

    # Init first row: delete all of ts2[0..j-1]
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + ts2[j - 1] + lam

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Cost of matching ts1[i-1] with ts2[j-1]
            # Add stiffness penalty proportional to time index difference
            c_match = (abs(ts1[i - 1] - ts2[j - 1])
                       + nu * abs((i - 1) - (j - 1)))

            # Previous match
            if i > 1 and j > 1:
                c_prev = abs(ts1[i - 2] - ts2[j - 2]) + nu * abs((i - 2) - (j - 2))
            else:
                c_prev = 0.0

            # Three transitions
            # 1. Match
            cost_match = dp[i - 1, j - 1] + c_match + c_prev

            # 2. Delete from ts1 (move down)
            cost_del1 = dp[i - 1, j] + (
                abs(ts1[i - 1] - ts1[i - 2]) + nu if i > 1 else ts1[i - 1]
            ) + lam

            # 3. Delete from ts2 (move right)
            cost_del2 = dp[i, j - 1] + (
                abs(ts2[j - 1] - ts2[j - 2]) + nu if j > 1 else ts2[j - 1]
            ) + lam

            dp[i, j] = min(cost_match, cost_del1, cost_del2)

    return float(dp[n, m])


def twed_similarity(target: np.ndarray, sources: list,
                    lam: float = 0.5, nu: float = 0.001) -> np.ndarray:
    """
    Compute TWED between target and each source series.

    Parameters
    ----------
    target  : 1-D array
    sources : list of 1-D arrays
    lam, nu : TWED hyper-parameters

    Returns
    -------
    np.ndarray, shape (n_sources,)
        TWED distance per source (lower = more similar).
    """
    return np.array([twed(target, s, lam=lam, nu=nu) for s in sources])


# ---------------------------------------------------------------------------
# 2. Transfer Entropy via Copula Entropy (nonparametric)
# ---------------------------------------------------------------------------

def _rank_normalise(x: np.ndarray) -> np.ndarray:
    """Map x to (0,1) via empirical CDF (rank-based)."""
    n = len(x)
    return rankdata(x) / (n + 1)   # avoids exact 0 or 1


def _copula_entropy_2d(u: np.ndarray, v: np.ndarray,
                        n_bins: int = None) -> float:
    """
    Estimate copula entropy H_C(U,V) = H(U,V) - H(U) - H(V)
    using rank-normalised histogram density (KNN or histogram).

    Since H(U) = H(V) = 0 for uniform marginals, we have
        H_C ≈ H(U, V)   (all in nats)

    We use a 2-D histogram for speed.
    """
    n = len(u)
    if n_bins is None:
        n_bins = max(5, int(np.sqrt(n / 5)))

    hist, _, _ = np.histogram2d(u, v, bins=n_bins, range=[[0, 1], [0, 1]])
    # Normalise to probability
    p = hist / hist.sum()
    # Shannon entropy (ignore zeros)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def transfer_entropy(source: np.ndarray, target: np.ndarray,
                     lag: int = 1, n_bins: int = None) -> float:
    """
    Estimate Transfer Entropy TE(source → target) via copula entropy.

    TE(X→Y) = I(Y_{t+1}; X_t | Y_t)
             ≈ H(Y_{t+1}, Y_t) + H(X_t, Y_t) − H(Y_t) − H(Y_{t+1}, X_t, Y_t)

    We approximate this with rank-normalised histograms (copula approach).

    Parameters
    ----------
    source, target : 1-D arrays of the same length.
    lag   : number of time steps for conditioning. Default 1.
    n_bins: histogram bins per dimension (auto if None).

    Returns
    -------
    float
        Estimated TE in nats (higher = more directional influence).
    """
    # Truncate to the shorter series so arrays stay aligned
    n = min(len(source), len(target))
    if n < lag + 2:
        return 0.0

    src = source[:n]
    tgt = target[:n]

    # Align
    X  = src[: n - lag]    # X_t
    Yt = tgt[: n - lag]    # Y_t
    Yf = tgt[lag:]          # Y_{t+lag}

    # Rank-normalise
    X_r  = _rank_normalise(X)
    Yt_r = _rank_normalise(Yt)
    Yf_r = _rank_normalise(Yf)

    n_b = n_bins or max(5, int((len(X) / 5) ** (1 / 3)))

    # H(Y_t+1, Y_t)
    h_yfy = _copula_entropy_2d(Yf_r, Yt_r, n_b)
    # H(X_t, Y_t)
    h_xy  = _copula_entropy_2d(X_r, Yt_r, n_b)
    # H(Y_t)
    h_y   = float(-np.mean(np.log(np.clip(_rank_normalise(Yt_r), 1e-10, 1))))

    # H(Y_t+1, X_t, Y_t) — 3-D histogram
    m = len(X_r)
    n_b3 = max(3, int((m / 10) ** (1 / 3)))
    hist3, _ = np.histogramdd(
        np.stack([Yf_r, X_r, Yt_r], axis=1),
        bins=n_b3,
        range=[[0, 1], [0, 1], [0, 1]],
    )
    p3 = hist3 / hist3.sum()
    mask3 = p3 > 0
    h_yfxy = float(-np.sum(p3[mask3] * np.log(p3[mask3])))

    te = h_yfy + h_xy - h_y - h_yfxy
    return max(float(te), 0.0)   # TE ≥ 0 in theory; clamp numerical noise


def transfer_entropy_all(target: np.ndarray, sources: list,
                          lag: int = 1, n_bins: int = None) -> np.ndarray:
    """
    Compute TE from each source to target.

    Returns
    -------
    np.ndarray, shape (n_sources,)
        TE per source (higher = more relevant).
    """
    return np.array([transfer_entropy(s, target, lag=lag, n_bins=n_bins)
                     for s in sources])


# ---------------------------------------------------------------------------
# 3. Combined ranking — select top-k sources
# ---------------------------------------------------------------------------

def select_sources(target: np.ndarray, sources: list,
                   source_names: list = None,
                   top_k: int = None,
                   lam: float = 0.5, nu: float = 0.001,
                   lag: int = 1) -> list:
    """
    Rank candidate source domains and return the top-k most suitable.

    Combined score (paper §3.2):
        rank_twed  — lower TWED → lower rank number → better
        rank_te    — higher TE  → lower rank number → better
        score = rank_twed + rank_te   (lower total = better source)

    Parameters
    ----------
    target       : 1-D float array — target domain representative series
    sources      : list of 1-D float arrays — one per candidate source
    source_names : optional list of str labels
    top_k        : how many sources to return (default = all, sorted)
    lam, nu      : TWED hyper-parameters
    lag          : TE lag (time steps)

    Returns
    -------
    list of (index, name, score) tuples, best sources first.
    """
    n = len(sources)
    if source_names is None:
        source_names = [f"src_{i}" for i in range(n)]

    twed_d = twed_similarity(target, sources, lam=lam, nu=nu)
    te_d   = transfer_entropy_all(target, sources, lag=lag)

    # Rank: rank 1 = best
    rank_twed = rankdata(twed_d)          # lower distance → lower rank
    rank_te   = rankdata(-te_d)           # higher TE → lower rank (negate)
    combined  = rank_twed + rank_te

    order = np.argsort(combined)
    top_k = top_k or n
    results = [
        (int(order[i]), source_names[order[i]], float(combined[order[i]]))
        for i in range(min(top_k, n))
    ]
    return results
