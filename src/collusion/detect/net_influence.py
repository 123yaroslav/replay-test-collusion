"""Pairwise (net) influence of Bonjour, Aggarwal & Bhargava (UAI 2022), plus calibrated tests.

    γ(i;j) = E_{s^ij} KL( p(a^i, a^j | s^i, s^j) ‖ p(a^i | s^i) p(a^j | s^j) )        (Eq. 3)
    Γ(i;j) = γ(i;j) − max_{k≠i,j} γ(k;j)                                            (Eq. 4)

Contemporaneous (not lagged), plug-in counts, bits. With s^i = s^j = s this is I(A^i;A^j|S);
with private states it also picks up A^i depending on j's private state (information leakage).
Bonjour flag (i,j) iff Γ(i;j) ≥ α and Γ(j;i) ≥ α and exactly one pair passes (Eq. 5). The paper's
Algorithm 1 loops over ordered pairs and double-counts; we follow Eqs. 3–5 over unordered pairs.

All inputs are non-negative integer arrays (encode symbols first with `encode`).
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import numpy as np


def encode(*cols: np.ndarray) -> np.ndarray:
    """Joint integer code for aligned non-negative integer columns (1-D, or 2-D with samples in
    rows). Mixed-radix arithmetic; falls back to np.unique if the code space gets large."""
    mats = [np.asarray(c, dtype=np.int64).reshape(len(c), -1) for c in cols]
    joint = np.concatenate(mats, axis=1)
    radix = joint.max(axis=0) + 1
    if np.prod(radix.astype(float)) < 2**31:
        code = np.zeros(len(joint), dtype=np.int64)
        for k in range(joint.shape[1]):
            code = code * radix[k] + joint[:, k]
        return code
    _, codes = np.unique(joint, axis=0, return_inverse=True)
    return codes.ravel()


def _count_lookup(*cols: np.ndarray) -> np.ndarray:
    """For every sample, how many samples share its joint value of `cols`."""
    codes = encode(*cols)
    if codes.max() < 1_000_000:
        return np.bincount(codes)[codes]
    _, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    return counts[inverse]


def gamma(ai: np.ndarray, aj: np.ndarray, si: np.ndarray | None = None,
          sj: np.ndarray | None = None) -> float:
    """Plug-in γ(i;j) in bits. si/sj default to a constant state (unconditional MI)."""
    n = len(ai)
    zero = np.zeros(n, dtype=int)
    si = zero if si is None else si
    sj = zero if sj is None else sj
    n_joint = _count_lookup(ai, aj, si, sj)
    n_sij = _count_lookup(si, sj)
    n_ai_si = _count_lookup(ai, si)
    n_si = _count_lookup(si)
    n_aj_sj = _count_lookup(aj, sj)
    n_sj = _count_lookup(sj)
    ratio = (n_joint / n_sij) / ((n_ai_si / n_si) * (n_aj_sj / n_sj))
    return float(np.mean(np.log2(ratio)))


def influence_matrix(actions: np.ndarray, states: np.ndarray | None = None) -> np.ndarray:
    """γ for every ordered pair. actions: (N, n_agents) ints; states: (N, n_agents) ints or None
    (None = unconditional; pass the same column for every agent for a shared public state)."""
    n_agents = actions.shape[1]
    g = np.full((n_agents, n_agents), np.nan)
    for i, j in combinations(range(n_agents), 2):
        si = None if states is None else states[:, i]
        sj = None if states is None else states[:, j]
        g[i, j] = g[j, i] = gamma(actions[:, i], actions[:, j], si, sj)
    return g


def net_influence(g: np.ndarray) -> np.ndarray:
    """Γ(i;j) = γ(i;j) − max_{k≠i,j} γ(k;j)."""
    n = g.shape[0]
    big = np.full_like(g, np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            others = [g[k, j] for k in range(n) if k not in (i, j)]
            big[i, j] = g[i, j] - max(others)
    return big


def pair_scores(big: np.ndarray) -> dict[tuple[int, int], float]:
    """Unordered pair score min(Γ(i;j), Γ(j;i)): both directions must exceed the benchmark."""
    return {(i, j): float(min(big[i, j], big[j, i]))
            for i, j in combinations(range(big.shape[0]), 2)}


def bonjour_rule(big: np.ndarray, alpha: float = 0.05) -> tuple[int, int] | None:
    flagged = [p for p, s in pair_scores(big).items() if s >= alpha]
    return flagged[0] if len(flagged) == 1 else None


def calibrated_test(
    actions: np.ndarray,
    states: np.ndarray | None,
    resample: Callable[[np.random.Generator], np.ndarray],
    n_null: int = 200,
    seed: int = 0,
    net: bool = True,
) -> tuple[tuple[int, int], float, dict[tuple[int, int], float]]:
    """Family-wise test over all pairs with a resampling null.

    `resample(rng)` returns a null action array of the same shape, e.g. a conditional
    randomization draw (re-query each agent's policy on its logged context) or a cross-episode
    surrogate. Returns (top pair, global p-value of its score vs the null max-pair score,
    per-pair p-values against the null of that pair's score). `net=False` scores pairs by raw
    γ instead of net influence (the plain pairwise-dependence baseline).
    """

    def scores(a: np.ndarray) -> dict[tuple[int, int], float]:
        g = influence_matrix(a, states)
        if net:
            return pair_scores(net_influence(g))
        return {(i, j): float(g[i, j]) for i, j in combinations(range(g.shape[0]), 2)}

    obs = scores(actions)
    rng = np.random.default_rng(seed)
    null_scores = [scores(resample(rng)) for _ in range(n_null)]
    top = max(obs, key=obs.get)
    null_max = np.array([max(s.values()) for s in null_scores])
    p_global = (1 + np.sum(null_max >= obs[top])) / (1 + n_null)
    p_pair = {p: float((1 + sum(s[p] >= v for s in null_scores)) / (1 + n_null))
              for p, v in obs.items()}
    return top, float(p_global), p_pair
