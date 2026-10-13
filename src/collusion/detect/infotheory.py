"""Discrete information-theoretic estimators over logged action sequences.

All quantities are in bits. Variables are 1-D arrays of hashable symbols (or 2-D arrays,
one row per sample, for joint variables). Plug-in estimates are biased upward for small
samples, so never threshold a raw value: compare against a surrogate null
(`surrogate_pvalue`) built from the same data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def _codes(*cols) -> np.ndarray:
    """Map one or more aligned columns to a single integer code per sample (joint symbol)."""
    if not cols:
        raise ValueError("need at least one column")
    stacked = []
    for c in cols:
        a = np.asarray(c, dtype=object)
        if a.ndim == 1:
            a = a[:, None]
        stacked.append(a)
    joint = np.concatenate(stacked, axis=1)
    _, codes = np.unique([repr(tuple(row)) for row in joint], return_inverse=True)
    return codes


def entropy(*cols, miller_madow: bool = True) -> float:
    """H(X1, ..., Xk) of the joint symbol formed by the given columns."""
    codes = _codes(*cols)
    n = len(codes)
    counts = np.bincount(codes)
    counts = counts[counts > 0]
    p = counts / n
    h = float(-(p * np.log2(p)).sum())
    if miller_madow:
        h += (len(counts) - 1) / (2 * n * np.log(2))
    return h


def mutual_info(x, y, z=None, miller_madow: bool = True) -> float:
    """I(X; Y) or, with z, the conditional mutual information I(X; Y | Z)."""
    if z is None:
        return (
            entropy(x, miller_madow=miller_madow)
            + entropy(y, miller_madow=miller_madow)
            - entropy(x, y, miller_madow=miller_madow)
        )
    return (
        entropy(x, z, miller_madow=miller_madow)
        + entropy(y, z, miller_madow=miller_madow)
        - entropy(x, y, z, miller_madow=miller_madow)
        - entropy(z, miller_madow=miller_madow)
    )


def lagged(series: Sequence, lag: int, length: int) -> list:
    """series[t-lag] for t = max_lag .. len-1, aligned to a window of `length` samples."""
    start = len(series) - length
    return list(series[start - lag : len(series) - lag])


def transfer_entropy(
    source_eps: Sequence[Sequence],
    target_eps: Sequence[Sequence],
    k: int = 1,
    cond_eps: Sequence[Sequence] | None = None,
) -> float:
    """TE_{source -> target} = I(Y_t ; X_{t-1..t-k} | Y_{t-1..t-k}, C_t), pooled over episodes.

    Each argument is a list of per-episode sequences (same length within an episode).
    `cond_eps` optionally adds an extra conditioning variable C_t (e.g. a public signal).
    """
    ys, xs_hist, ys_hist, cs = [], [], [], []
    for e, (x, y) in enumerate(zip(source_eps, target_eps, strict=True)):
        n = len(y) - k
        if n <= 0:
            continue
        ys += lagged(y, 0, n)
        xs_hist += list(zip(*[lagged(x, lag, n) for lag in range(1, k + 1)], strict=True))
        ys_hist += list(zip(*[lagged(y, lag, n) for lag in range(1, k + 1)], strict=True))
        if cond_eps is not None:
            cs += lagged(cond_eps[e], 0, n)
    z = [h + (c,) for h, c in zip(ys_hist, cs, strict=True)] if cond_eps is not None else ys_hist
    return mutual_info(ys, [repr(h) for h in xs_hist], [repr(h) for h in z])


def surrogate_pvalue(
    stat: Callable[[np.random.Generator], float],
    observed: float,
    n_surrogates: int = 500,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """One-sided p-value of `observed` against stat(rng) evaluated on surrogate data.

    `stat` must build its own surrogate (e.g. cross-episode re-pairing or circular shift)
    from the rng it is given. Uses the (k + 1) / (n + 1) estimator so p is never 0.
    """
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng) for _ in range(n_surrogates)])
    p = (1 + np.sum(null >= observed)) / (1 + n_surrogates)
    return float(p), null


def cross_episode_pairs(n_episodes: int, rng: np.random.Generator) -> np.ndarray:
    """A derangement of episode indices: pair agent i of episode e with agent j of episode pi(e).

    This keeps each agent's marginal policy (including priors shared by same-model agents)
    but destroys any within-episode interaction - the null for "they influence each other".
    """
    if n_episodes < 2:
        raise ValueError("cross-episode surrogates need >= 2 episodes")
    while True:
        perm = rng.permutation(n_episodes)
        if not np.any(perm == np.arange(n_episodes)):
            return perm


def circular_shift(series: Sequence, rng: np.random.Generator, min_shift: int = 1) -> list:
    """Rotate a sequence by a random offset; preserves autocorrelation, breaks cross-alignment."""
    n = len(series)
    s = int(rng.integers(min_shift, max(min_shift + 1, n - min_shift + 1)))
    return list(series[s:]) + list(series[:s])
