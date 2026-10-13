"""S0 ($0): scripted validation of pairwise net influence and its calibration.

Part A replicates Bonjour et al. (2022) Table 2: 3-player RPS, B follows A w.p. CP, detector
conditions on the previous joint action, fixed alpha = 0.05.

Part B asks the question that matters for LLM agents: which calibration keeps false alarms at
the nominal 5% when honest agents share priors and react to a common public history (the
"same model" decoy), while keeping power against a covert colluding pair?

    uv run python experiments/s0_scripted_validation.py [--iters 200] [--quick]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from collusion.detect.net_influence import (
    bonjour_rule,
    calibrated_test,
    influence_matrix,
    net_influence,
)
from collusion.games.rps import Population, crt_resampler, simulate

OUT = Path(__file__).resolve().parents[1] / "results" / "summary"

POPULATIONS = {
    "null_uniform": Population(types=("uniform",) * 4),
    "null_herd": Population(types=("herd_x", "herd_x", "herd_y", "herd_y")),
    "pair_uniform": Population(types=("uniform",) * 4, colluders=(0, 2), cp=0.4),
    "pair_herd": Population(types=("herd_x", "herd_x", "herd_y", "herd_y"),
                            colluders=(0, 2), cp=0.4),
}


def shuffle_null(a):
    def draw(rng):
        return np.column_stack([rng.permutation(a[:, i]) for i in range(a.shape[1])])
    return draw


def stratified_null(a, strata):
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]

    def draw(rng):
        out = a.copy()
        for idx in groups:
            if len(idx) > 1:
                for i in range(a.shape[1]):
                    out[idx, i] = a[rng.permutation(idx), i]
        return out
    return draw


def run_detectors(a, ref, prev, n_null, seed):
    s_prev = np.repeat(prev[:, None], a.shape[1], axis=1)
    out = {
        "bonjour_alpha_nostate": bonjour_rule(net_influence(influence_matrix(a))),
        "bonjour_alpha_prevstate": bonjour_rule(net_influence(influence_matrix(a, s_prev))),
    }
    tests = {
        "raw_gamma_shuffle": (None, shuffle_null(a), False),
        "net_shuffle": (None, shuffle_null(a), True),
        "net_stratified_prev": (s_prev, stratified_null(a, prev), True),
        "net_crt": (None, crt_resampler(ref), True),
    }
    for name, (states, null, net) in tests.items():
        top, p, _ = calibrated_test(a, states, null, n_null=n_null, seed=seed, net=net)
        out[name] = top if p < 0.05 else None
    return out


def part_b_cell(args):
    pop_name, n_games, it, n_null = args
    pop = POPULATIONS[pop_name]
    rng = np.random.default_rng(10_000 * n_games + it)
    a, ref, prev = simulate(pop, n_games, rng)
    flags = run_detectors(a, ref, prev, n_null, seed=it)
    truth = tuple(sorted(pop.colluders)) if pop.colluders else None
    return [{"population": pop_name, "n_games": n_games, "iter": it, "detector": d,
             "flagged": f is not None, "correct": f is not None and f == truth}
            for d, f in flags.items()]


def part_a(iters: int) -> pd.DataFrame:
    rows = []
    for cp in (0.3, 0.4):
        pop = Population(types=("uniform",) * 3, colluders=(0, 1), cp=cp)
        for n_games in (100, 200, 300, 400, 500, 1000):
            hits = 0
            for it in range(iters):
                a, _, prev = simulate(pop, n_games, np.random.default_rng(it))
                s_prev = np.repeat(prev[:, None], 3, axis=1)
                hits += bonjour_rule(net_influence(influence_matrix(a, s_prev))) == (0, 1)
            rows.append({"cp": cp, "n_games": n_games, "cda_pct": 100 * hits / iters})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--n-null", type=int, default=99)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.iters, args.n_null = 20, 39
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    a = part_a(min(args.iters * 2, 500))
    a.to_csv(OUT / "s0_partA_bonjour_table2.csv", index=False)
    print("Part A: Bonjour Table 2 replication (CDA %)")
    print(a.pivot(index="cp", columns="n_games", values="cda_pct").round(1).to_string(), "\n")

    jobs = [(p, n, it, args.n_null) for p in POPULATIONS for n in (200, 500)
            for it in range(args.iters)]
    with ProcessPoolExecutor() as ex:
        rows = [r for cell in ex.map(part_b_cell, jobs, chunksize=4) for r in cell]
    b = pd.DataFrame(rows)
    b.to_csv(OUT / "s0_partB_raw.csv", index=False)
    summary = b.groupby(["detector", "population", "n_games"]).agg(
        flag_rate=("flagged", "mean"), correct_rate=("correct", "mean")).reset_index()
    summary.to_csv(OUT / "s0_partB_summary.csv", index=False)
    show = summary.assign(
        value=lambda d: np.where(d.population.str.startswith("null"), d.flag_rate, d.correct_rate))
    print("Part B: false-alarm rate (null_*) / correct-pair rate (pair_*), "
          f"{args.iters} iterations per cell")
    print(show.pivot_table(index="detector", columns=["population", "n_games"], values="value")
          .round(3).to_string())
    print(f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
