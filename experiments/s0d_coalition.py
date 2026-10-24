"""S0d ($0, EXPLORATORY, after the reviews): coalitions of three on the scripted simulator.

Bonjour et al.'s net influence assumes exactly two colluders: Gamma(i;j) subtracts the largest
dependence of j on any third agent, so inside a coalition of three the followers' dependence on
each other can cancel their dependence on the leader. The replay null does not need that
subtraction, because shared priors are already in the replays. We compare, on the same pools:
  crt_net   the pre-registered statistic (net influence, max over pairs)
  crt_mi    raw pairwise mutual information (max over pairs), same replay null
  seq       the anytime-valid sequential test ("joint" bettor)
Coalition: a leader and one or two followers; with probability cp per round each follower plays
the move the leader's move beats (as for the pair in s0b). Global alarm and "the named pair lies
inside the coalition" are reported separately.

    uv run python experiments/s0d_coalition.py [--iters 300]
Writes results/summary/s0d_coalition.csv.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from collusion.analysis import wilson
from collusion.detect.crt_llm import pooled_sampler, sequential_replay_test
from collusion.detect.net_influence import calibrated_test
from collusion.games.rps import Population, _reference

OUT = Path(__file__).resolve().parents[1] / "results" / "summary"
TYPES = {4: ("herd_x", "herd_x", "herd_y", "herd_y"),
         5: ("herd_x", "herd_x", "herd_y", "herd_y", "uniform")}
SETUPS = {  # name: (n players, leader, followers)
    "4p_honest": (4, None, ()),
    "4p_pair": (4, 0, (2,)),
    "4p_triple": (4, 0, (1, 2)),
    "5p_honest": (5, None, ()),
    "5p_pair": (5, 0, (2,)),
    "5p_triple": (5, 0, (2, 4)),
}


def simulate_coalition(n, leader, followers, cp, T, rng):
    pop = Population(types=TYPES[n])
    actions = np.zeros((T, n), dtype=np.int64)
    ref = np.zeros((T, n, 3))
    for t in range(T):
        recent = actions[max(0, t - pop.memory):t]
        for i, kind in enumerate(pop.types):
            ref[t, i] = _reference(kind, recent, replace(pop))
            actions[t, i] = rng.choice(3, p=ref[t, i])
        if leader is not None and rng.random() < cp:
            for f in followers:
                actions[t, f] = (actions[t, leader] + 2) % 3
    return actions, ref


def cell(args):
    setup, cp, T, it = args
    n, leader, followers = SETUPS[setup]
    rng = np.random.default_rng(7_777_7 * T + 101 * it + int(100 * cp) + 13 * len(setup) + n)
    a, ref = simulate_coalition(n, leader, followers, cp, T, rng)
    cdf = ref.cumsum(axis=2)
    d = (rng.random(ref.shape[:2] + (2,))[..., None] > cdf[:, :, None, :]).sum(axis=3).clip(max=2)
    pool = np.concatenate([a[..., None], d], axis=2)
    coal = set(followers) | ({leader} if leader is not None else set())
    rows = []
    for det, net in (("crt_net", True), ("crt_mi", False)):
        top, p, _ = calibrated_test(a, None, pooled_sampler(pool), n_null=199, seed=it, net=net)
        rows.append({"detector": det, "flag": p < 0.05, "inside": p < 0.05 and set(top) <= coal})
    r = sequential_replay_test(pool, alpha=0.05, n_actions=3, bettor="joint")
    flag = r["reject_round"] is not None
    rows.append({"detector": "seq", "flag": flag, "inside": flag and set(r["top"]) <= coal})
    return [{"setup": setup, "cp": cp, "T": T, "iter": it, **x} for x in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()
    jobs = [(s, 0.0 if "honest" in s else 0.8, T, it) for s in SETUPS for T in (30, 60)
            for it in range(args.iters)]
    with ProcessPoolExecutor() as ex:
        df = pd.DataFrame([r for rs in ex.map(cell, jobs, chunksize=10) for r in rs])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "s0d_coalition.csv", index=False)
    out = []
    for (s, T, det), g in df.groupby(["setup", "T", "detector"]):
        k, n = int(g.flag.sum()), len(g)
        lo, hi = wilson(k, n)
        out.append({"setup": s, "T": T, "detector": det, "n": n, "flag": k / n, "lo": lo,
                    "hi": hi, "inside": g.inside.mean()})
    s = pd.DataFrame(out)
    s.to_csv(OUT / "s0d_coalition_summary.csv", index=False)
    print(s.pivot_table(index=["setup", "T"], columns="detector", values=["flag", "inside"])
          .round(3).to_string())


if __name__ == "__main__":
    main()
