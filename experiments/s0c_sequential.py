"""S0c ($0, EXPLORATORY, post-freeze): the anytime-valid sequential replay test on the simulator.

Calibration on honest populations (incl. strong history feedback, where the fixed-horizon test
has no finite-sample guarantee) and power / rounds-to-detection for a colluding pair.

    uv run python experiments/s0c_sequential.py [--iters 300]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from collusion.detect.crt_llm import sequential_replay_test
from collusion.games.rps import Population, simulate

OUT = Path(__file__).resolve().parents[1] / "results" / "summary"
BETTOR = os.environ.get("BETTOR", "joint")
HERD = ("herd_x", "herd_x", "herd_y", "herd_y")
POPS = {
    "herd": Population(types=HERD),
    "uniform": Population(types=("uniform",) * 4),
    "herd_strong_feedback": Population(types=HERD, herd_weight=0.9, memory=1),
    "pair_cp0.6": Population(types=HERD, colluders=(0, 2), cp=0.6),
    "pair_cp0.8": Population(types=HERD, colluders=(0, 2), cp=0.8),
    "pair_cp1.0": Population(types=HERD, colluders=(0, 2), cp=1.0),
}


def cell(args):
    pop, T, it = args
    rng = np.random.default_rng(31 * it + 7 * T + sum(map(ord, pop)))
    a, ref, _ = simulate(POPS[pop], T, rng)
    cdf = ref.cumsum(axis=2)
    u = rng.random(ref.shape[:2] + (2,))
    d = (u[..., None] > cdf[:, :, None, :]).sum(axis=3).clip(max=2)
    pool = np.concatenate([a[..., None], d], axis=2)
    r = sequential_replay_test(pool, alpha=0.05, n_actions=3, bettor=BETTOR)
    truth = POPS[pop].colluders
    return {"population": pop, "T": T, "reject": r["reject_round"] is not None,
            "reject_round": r["reject_round"],
            "correct": r["reject_round"] is not None and truth is not None
            and tuple(r["top"]) == truth}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()
    jobs = [(p, T, it) for p in POPS for T in (30, 60, 120) for it in range(args.iters)]
    with ProcessPoolExecutor() as ex:
        df = pd.DataFrame(list(ex.map(cell, jobs, chunksize=20)))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"s0c_sequential{'' if BETTOR == 'joint' else '_' + BETTOR}.csv", index=False)
    s = df.groupby(["population", "T"]).agg(reject=("reject", "mean"), correct=("correct", "mean"),
                                           median_round=("reject_round", "median")).round(3)
    print(s.to_string())


if __name__ == "__main__":
    main()
