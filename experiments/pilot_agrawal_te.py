"""Pilot (pre-sprint, $0): directional influence among sellers in Agrawal et al.'s shipped runs.

Per condition, pool episodes and compute, for every ordered seller pair (j -> i), the transfer
entropy of discretised ask moves, TE_{j->i} = I(m_i,t ; m_j,t-1 | m_i,t-1). Compare with a
cross-episode surrogate (i from episode e, j from a different episode of the same condition),
which keeps each seller's marginal behaviour and shared-model priors but removes interaction.
"""

from collections import defaultdict
from itertools import permutations

import numpy as np

from collusion.adapters.agrawal import load_all
from collusion.detect.infotheory import (
    cross_episode_pairs,
    mutual_info,
    surrogate_pvalue,
    transfer_entropy,
)


def moves(asks, tol=0.005):
    out, prev = [], None
    for a in asks:
        if a is None:
            out.append("W")
        elif prev is None:
            out.append("S")
        else:
            out.append("D" if a < prev - tol else "U" if a > prev + tol else "S")
        prev = a if a is not None else prev
    return out


def pooled_te(eps_moves, perm=None):
    """Mean over ordered seller-slot pairs of pooled TE; slot = position in seller_ids."""
    n_slots = len(eps_moves[0])
    vals = []
    for i, j in permutations(range(n_slots), 2):
        tgt = [m[i] for m in eps_moves]
        src = [eps_moves[perm[e]][j] if perm is not None else m[j]
               for e, m in enumerate(eps_moves)]
        vals.append(transfer_entropy(src, tgt))
    return float(np.mean(vals))


def pooled_mi(eps_moves, perm=None):
    n_slots = len(eps_moves[0])
    vals = []
    for i, j in permutations(range(n_slots), 2):
        if i > j:
            continue
        x = [v for m in eps_moves for v in m[i]]
        y = [v for e, m in enumerate(eps_moves)
             for v in (eps_moves[perm[e]][j] if perm is not None else m[j])]
        vals.append(mutual_info(x, y))
    return float(np.mean(vals))


def main():
    by_cond = defaultdict(list)
    for ep in load_all():
        by_cond[ep.condition].append([moves(ep.action_series(s)) for s in ep.meta["seller_ids"]])
    print(f"{'condition':45s} n  TE_obs  TE_null  p_TE   MI_obs  MI_null  p_MI")
    for cond, eps in sorted(by_cond.items()):
        te = pooled_te(eps)
        p_te, null_te = surrogate_pvalue(
            lambda r, eps=eps: pooled_te(eps, cross_episode_pairs(len(eps), r)), te, n_surrogates=200)
        mi = pooled_mi(eps)
        p_mi, null_mi = surrogate_pvalue(
            lambda r, eps=eps: pooled_mi(eps, cross_episode_pairs(len(eps), r)), mi, n_surrogates=200)
        print(f"{cond:45s} {len(eps):2d} {te:6.3f}  {null_te.mean():6.3f}  {p_te:5.3f}  "
              f"{mi:6.3f}  {null_mi.mean():6.3f}  {p_mi:5.3f}")


if __name__ == "__main__":
    main()
