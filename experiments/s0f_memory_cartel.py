"""S0f ($0, EXPLORATORY, after the auction pilot): a cartel whose agreement lives in the agents'
memory, not in the channel. Scripted sellers, no LLM.

Five sellers ask one of six price levels each round. A public shock (0/1, shared) shifts honest
asks up, so honest asks are correlated. Honest policy with memory: with probability rho repeat
one's own last ask (kept in memory), otherwise draw from the shock's base distribution.

Cartel: S1 leads, S2 and S3 follow, S4 and S5 are honest. S1 follows a floor schedule (levels 3-5)
fixed in its instructions, asks the floor with probability c = 0.85 and announces next round's
floor in the channel. A follower stores the latest floor it received in memory and asks it with
probability c (otherwise it plays the honest policy). Variants: `silent` (one announcement in
round 0, then an empty channel), `change_u` (an announcement every round; the floor changes with
probability u).

Two ways to build the pool of "what the agent would do without the channel" (k = 2 each):
  replay  re-query on the logged observation with this round's inbox removed; memory as logged
          (the test in the paper: exact null, measures the direct effect of the latest messages)
  twin    a copy of the agent run from round 0 without any inbox, writing its own memory;
          it sees the logged public shocks. No exact null: its history diverges from the log even
          for honest agents with memory. It shows the total effect; it is not a detector here.
Detectors as in the paper: crt_mi and crt_net (B = 199) and the sequential test, on each pool.
With rho = 0 (no own-ask memory) a twin of an honest agent has the same law as its replay, so the
cartel comparison is clean there; rho > 0 shows the twin's false alarms, and also how the
fixed-horizon raw-MI test behaves when honest asks persist (the pooled null breaks the runs).

    uv run python experiments/s0f_memory_cartel.py [--iters 200]
Writes results/summary/s0f_memory_cartel.csv and s0f_memory_cartel_summary.csv.
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
N, K, LEVELS, C = 5, 2, 6, 0.85
BASE = np.array([[.30, .30, .20, .10, .05, .05], [.05, .30, .30, .20, .10, .05]])
CARTEL, FOLLOWERS = (0, 1, 2), (1, 2)
SCEN = [("honest", 0.0, None), ("honest", 0.2, None), ("honest", 0.5, None), ("honest", 0.8, None),
        *[(k, rho, u) for rho in (0.0, 0.5) for k, u in (("silent", 0.0), ("change", 0.2),
                                                         ("change", 0.5))]]


def honest(last, shock, rho, rng):
    if last is not None and rng.random() < rho:
        return last
    return int(rng.choice(LEVELS, p=BASE[shock]))


def act(agent, floor, last, shock, rho, rng):
    """Policy of one seller given its floor knowledge (None = no floor) and own last ask."""
    if floor is not None and rng.random() < C:
        return floor
    return honest(last, shock, rho, rng)


def simulate(kind, rho, u, T, rng):
    shocks = rng.integers(0, 2, size=T)
    cartel = kind != "honest"
    floors = np.zeros(T + 1, dtype=int)  # floors[t] applies in round t (t >= 1)
    if cartel:
        floors[1] = rng.integers(3, 6)
        for t in range(2, T + 1):
            change = kind == "change" and rng.random() < u
            floors[t] = rng.choice([f for f in (3, 4, 5) if f != floors[t - 1]]) if change \
                else floors[t - 1]
    # announced[t] = floor sent in round t (read by followers in round t + 1)
    announced = [None] * T
    if cartel:
        for t in range(T):
            if kind == "change" or t == 0:
                announced[t] = int(floors[t + 1])

    def leader_floor(t):
        return int(floors[t]) if cartel and t >= 1 else None

    logged = np.zeros((T, N), dtype=int)
    rep = np.zeros((T, N, K), dtype=int)
    twin = np.zeros((T, N, K), dtype=int)
    mem_floor = [None] * N  # follower memory: latest floor received
    twin_last = [[None] * K for _ in range(N)]
    for t in range(T):
        s = shocks[t]
        for i in range(N):
            last = logged[t - 1, i] if t else None
            if cartel and i == 0:
                f_log = f_rep = f_twin = leader_floor(t)  # its schedule is in its instructions
            elif cartel and i in FOLLOWERS:
                f_rep = mem_floor[i]  # memory before this round's inbox
                inbox = announced[t - 1] if t else None
                if inbox is not None:
                    mem_floor[i] = inbox
                f_log, f_twin = mem_floor[i], None
            else:
                f_log = f_rep = f_twin = None
            logged[t, i] = act(i, f_log, last, s, rho, rng)
            rep[t, i] = [act(i, f_rep, last, s, rho, rng) for _ in range(K)]
            for k in range(K):
                twin[t, i, k] = act(i, f_twin, twin_last[i][k], s, rho, rng)
                twin_last[i][k] = twin[t, i, k]
    return logged, rep, twin


def cell(args):
    kind, rho, u, T, it = args
    rng = np.random.default_rng(1_000_003 * T + 7919 * it + 101 * SCEN.index((kind, rho, u)))
    logged, rep, twin = simulate(kind, rho, u, T, rng)
    row = {"scenario": f"{kind}_rho{rho}" + (f"_u{u}" if kind == "change" else ""),
           "T": T, "iter": it}
    fol = list(FOLLOWERS)
    row["direct_effect"] = float(logged[:, fol].mean() - rep[:, fol].mean())
    row["total_effect"] = float(logged[:, fol].mean() - twin[:, fol].mean())
    for name, alt in (("replay", rep), ("twin", twin)):
        pool = np.concatenate([logged[..., None], alt], axis=2)
        for det, net in (("crt_mi", False), ("crt_net", True)):
            top, p, _ = calibrated_test(logged, None, pooled_sampler(pool), n_null=199,
                                        seed=it, net=net)
            row[f"{name}_{det}"] = p < 0.05
            row[f"{name}_{det}_pair"] = p < 0.05 and set(top) <= set(CARTEL)
        r = sequential_replay_test(pool, alpha=0.05, n_actions=LEVELS)
        row[f"{name}_seq"] = r["reject_round"] is not None
        row[f"{name}_seq_pair"] = row[f"{name}_seq"] and set(r["top"]) <= set(CARTEL)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    jobs = [(k, rho, u, T, it) for (k, rho, u) in SCEN for T in (30, 60)
            for it in range(args.iters)]
    with ProcessPoolExecutor() as ex:
        df = pd.DataFrame(list(ex.map(cell, jobs, chunksize=8)))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "s0f_memory_cartel.csv", index=False)
    dets = [f"{m}_{d}" for m in ("replay", "twin") for d in ("crt_mi", "crt_net", "seq")]
    out = []
    for (s, T), g in df.groupby(["scenario", "T"], sort=False):
        row = {"scenario": s, "T": T, "n": len(g),
               "direct_effect": g.direct_effect.mean(), "total_effect": g.total_effect.mean()}
        for d in dets + [x + "_pair" for x in dets]:
            k = int(g[d].sum())
            lo, hi = wilson(k, len(g))
            row[d], row[d + "_lo"], row[d + "_hi"] = k / len(g), lo, hi
        out.append(row)
    s = pd.DataFrame(out)
    s.to_csv(OUT / "s0f_memory_cartel_summary.csv", index=False)
    with pd.option_context("display.width", 250):
        print(s[["scenario", "T", "direct_effect", "total_effect", *dets]].round(3)
              .to_string(index=False))


if __name__ == "__main__":
    main()
