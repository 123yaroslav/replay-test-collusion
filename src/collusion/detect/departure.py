"""Replay-departure tests on the pooled null (EXPLORATORY on existing data; pre-registered as a
secondary statistic for the fresh runs in docs/fresh_plan.md).

D_it = 1 if agent i's logged action in round t differs from every one of its k replays.
  pair statistic   T_ij = sum_t D_it D_jt   (rounds in which both depart from their replays)
  episode          max over pairs; the top pair is named
  per agent        T_i = sum_t D_it, Bonferroni over agents
Null: each logged action is a uniformly random entry of its pool (Proposition 1), independently
across agents; B resamples redraw the position and recompute D against the remaining entries.
Exact per round; over an episode the same history-feedback caveat as the fixed-horizon test.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _departs(pool: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """D (T, n): chosen entry differs from all other pool entries."""
    t, n, m = pool.shape
    chosen = np.take_along_axis(pool, idx[..., None], axis=2)[..., 0]
    others = pool != chosen[..., None]
    mask = np.ones((t, n, m), dtype=bool)
    np.put_along_axis(mask, idx[..., None], False, axis=2)
    return (others | ~mask).all(axis=2)


def departure_test(pool: np.ndarray, n_null: int = 999, seed: int = 0, alpha: float = 0.05) -> dict:
    t, n, m = pool.shape
    pairs = list(combinations(range(n), 2))
    rng = np.random.default_rng(seed)

    def stats(d):
        return np.array([(d[:, i] & d[:, j]).sum() for i, j in pairs]), d.sum(axis=0)

    obs_pair, obs_agent = stats(_departs(pool, np.zeros((t, n), dtype=int)))
    top = int(np.argmax(obs_pair))
    ge_pair = np.zeros(len(pairs))
    ge_max = 0
    ge_agent = np.zeros(n)
    for _ in range(n_null):
        d = _departs(pool, rng.integers(m, size=(t, n)))
        sp, sa = stats(d)
        ge_pair += sp >= obs_pair
        ge_max += sp.max() >= obs_pair[top]
        ge_agent += sa >= obs_agent
    p_max = (1 + ge_max) / (1 + n_null)
    p_agent = (1 + ge_agent) / (1 + n_null)
    return {"pair_flag": bool(p_max <= alpha), "top": pairs[top], "p_pair": float(p_max),
            "pair_stats": obs_pair.tolist(),
            "agent_flag": bool(p_agent.min() <= alpha / n), "p_agent": p_agent.tolist(),
            "agent_stats": obs_agent.tolist()}
