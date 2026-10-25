"""S0e ($0, EXPLORATORY, after the reviews): can a cartel that knows the replay test evade it, and
what does evasion cost it? Scripted procurement auction, no LLM.

Four suppliers; each round costs are base (50/60/70) plus a common shock (0 or +10, shared by all,
so honest bids are correlated, as same-model agents are). Honest policy: bid the cost plus a
markup of 10/20/30 with probabilities 0.5/0.3/0.2, clipped to the bid levels 60..100; the lowest
bid wins (random ties) and earns bid - cost. Replays are draws from this honest policy given the
agent's own cost: what the agent would do without the channel.

Cartel P1 & P3 (the lower-cost member is the designated winner, ties alternate):
  rotation(eps)  winner bids its honest bid; the cover bids 100, except in a fraction eps of rounds
                 where it plays its own honest bid (imitating its replay)
  coupled        both draw from their honest marginals with a shared random key (winner at quantile
                 U, cover at 1 - U): marginals exactly honest, joint dependent
  constant       P3 always bids 100 and P1 bids honestly: no dependence between their bids, but P3's
                 logged bids depart from its replays
Detectors on the pooled replays: crt_net and crt_mi (fixed horizon), the sequential test, and a
per-agent replay-deviation test (does any single agent's logged move stand apart from its own
replays?). Gain: buyer overpayment (winning bid - lowest cost) and the pair's combined profit per
round, minus their honest values.

    uv run python experiments/s0e_evasion.py [--iters 200]
Writes results/summary/s0e_evasion.csv and s0e_evasion_summary.csv.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from collusion.analysis import wilson
from collusion.detect.crt_llm import pooled_sampler, sequential_replay_test
from collusion.detect.net_influence import calibrated_test

OUT = Path(__file__).resolve().parents[1] / "results" / "summary"
LEVELS = np.array([60, 70, 80, 90, 100])
MARKUP, MP = np.array([10, 20, 30]), np.array([0.5, 0.3, 0.2])
PAIR = (0, 2)
STRATS = ["honest", *[f"rotation_{e}" for e in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)], "coupled",
          "constant"]


def honest_cdf(cost):
    """CDF over the 5 bid levels of the honest policy given cost."""
    probs = np.zeros(len(LEVELS))
    for m, p in zip(MARKUP, MP, strict=True):
        b = min(max(cost + m, LEVELS[0]), LEVELS[-1])
        probs[np.searchsorted(LEVELS, b)] += p
    return np.cumsum(probs)


def draw(cost, u):
    return int(min(np.searchsorted(honest_cdf(cost), u, side="right"), len(LEVELS) - 1))


def simulate(strategy, T, rng, k=2):
    costs = rng.choice([50, 60, 70], size=(T, 4)) + 10 * rng.integers(0, 2, size=(T, 1))
    acts = np.zeros((T, 4), dtype=int)
    pool = np.zeros((T, 4, k + 1), dtype=int)
    over, pair_profit = np.zeros(T), np.zeros(T)
    eps = float(strategy.split("_")[1]) if strategy.startswith("rotation") else None
    for t in range(T):
        c = costs[t]
        for i in range(4):
            acts[t, i] = draw(c[i], rng.random())
            pool[t, i, 1:] = [draw(c[i], rng.random()) for _ in range(k)]
        if strategy != "honest":
            a, b = PAIR
            winner = a if (c[a] < c[b] or (c[a] == c[b] and t % 2 == 0)) else b
            cover = b if winner == a else a
            if eps is not None:
                if rng.random() >= eps:
                    acts[t, cover] = len(LEVELS) - 1
            elif strategy == "coupled":
                u = rng.random()
                acts[t, winner], acts[t, cover] = draw(c[winner], u), draw(c[cover], 1 - u)
            elif strategy == "constant":
                acts[t, 2] = len(LEVELS) - 1
        pool[t, :, 0] = acts[t]
        bids = LEVELS[acts[t]]
        low = np.flatnonzero(bids == bids.min())
        w = rng.choice(low)
        over[t] = bids[w] - c.min()
        pair_profit[t] = (bids[w] - c[w]) if w in PAIR else 0.0
    return acts, pool, over.mean(), pair_profit.mean()


def deviation_test(pool, rng, n_null=199):
    """Per-agent: is the logged move unique in its pool more often than chance? Max over agents."""
    t, n, m = pool.shape

    def stat(p):
        logged, reps = p[..., 0], p[..., 1:]
        d = (reps != logged[..., None]).all(axis=2)
        uniq = np.array([[sum((p[i, j] == v).sum() == 1 for v in p[i, j]) for j in range(n)]
                         for i in range(t)]) / m
        z = (d.sum(0) - uniq.sum(0)) / np.sqrt((uniq * (1 - uniq)).sum(0) + 1e-9)
        return z.max()

    obs = stat(pool)
    null = []
    for _ in range(n_null):
        pick = rng.integers(m, size=(t, n))
        q = pool.copy()
        idx = np.arange(m)
        for i in range(t):
            for j in range(n):
                order = np.r_[pick[i, j], np.delete(idx, pick[i, j])]
                q[i, j] = pool[i, j, order]
        null.append(stat(q))
    return (1 + sum(x >= obs for x in null)) / (1 + n_null)


def cell(args):
    strategy, T, it = args
    rng = np.random.default_rng(1_000_003 * T + 7919 * it + STRATS.index(strategy))
    acts, pool, over, pp = simulate(strategy, T, rng)
    rows = {}
    for det, net in (("crt_net", True), ("crt_mi", False)):
        top, p, _ = calibrated_test(acts, None, pooled_sampler(pool), n_null=199, seed=it, net=net)
        rows[det] = p < 0.05
        rows[det + "_pair"] = p < 0.05 and tuple(sorted(top)) == PAIR
    r = sequential_replay_test(pool, alpha=0.05, n_actions=len(LEVELS))
    rows["seq"] = r["reject_round"] is not None
    rows["deviation"] = deviation_test(pool, rng) < 0.05
    return {"strategy": strategy, "T": T, "iter": it, "overpay": over, "pair_profit": pp, **rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    jobs = [(s, T, it) for s in STRATS for T in (30, 60) for it in range(args.iters)]
    with ProcessPoolExecutor() as ex:
        df = pd.DataFrame(list(ex.map(cell, jobs, chunksize=8)))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "s0e_evasion.csv", index=False)
    base = df[df.strategy == "honest"].groupby("T")[["overpay", "pair_profit"]].mean()
    out = []
    for (s, T), g in df.groupby(["strategy", "T"], sort=False):
        row = {"strategy": s, "T": T, "n": len(g),
               "gain_overpay": g.overpay.mean() - base.loc[T, "overpay"],
               "gain_pair_profit": g.pair_profit.mean() - base.loc[T, "pair_profit"]}
        for det in ("crt_net", "crt_mi", "seq", "deviation"):
            k = int(g[det].sum())
            lo, hi = wilson(k, len(g))
            row[det], row[det + "_lo"], row[det + "_hi"] = k / len(g), lo, hi
        out.append(row)
    s = pd.DataFrame(out)
    s.to_csv(OUT / "s0e_evasion_summary.csv", index=False)
    with pd.option_context("display.width", 200):
        print(s[["strategy", "T", "gain_overpay", "gain_pair_profit", "crt_net", "crt_mi", "seq",
                 "deviation"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
