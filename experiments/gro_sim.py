"""EXPLORATORY ($0, after the main analysis): the likelihood-ratio bet against the joint bet in the
persistence simulator (experiments/s0f_memory_cartel.py): honest false alarms with and without
own-move memory, and detection of a memory cartel whose floor changes.

    uv run python experiments/gro_sim.py [--iters 300]
Writes results/summary/gro_sim.csv.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import s0f_memory_cartel as s0f
from collusion.detect.crt_llm import sequential_replay_test
from collusion.detect.gro import sequential_gro_test

OUT = Path(__file__).resolve().parents[1] / "results" / "summary"
SCEN = [("honest", 0.0, None), ("honest", 0.5, None), ("honest", 0.8, None),
        ("change", 0.0, 0.2), ("change", 0.5, 0.2), ("change", 0.0, 0.5), ("change", 0.5, 0.5)]


def cell(args):
    kind, rho, u, T, it = args
    rng = np.random.default_rng(2_000_003 * T + 7919 * it + 101 * SCEN.index((kind, rho, u)))
    logged, rep, _ = s0f.simulate(kind, rho, u, T, rng)
    pool = np.concatenate([logged[..., None], rep], axis=2)
    j = sequential_replay_test(pool, alpha=0.05, n_actions=s0f.LEVELS)["reject_round"] is not None
    g = sequential_gro_test(pool, alpha=0.05, n_actions=s0f.LEVELS)["reject_round"] is not None
    return {"scenario": f"{kind}_rho{rho}" + (f"_u{u}" if u else ""), "T": T, "iter": it,
            "joint": j, "gro": g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()
    jobs = [(k, r, u, T, it) for (k, r, u) in SCEN for T in (30, 60) for it in range(args.iters)]
    with ProcessPoolExecutor() as ex:
        df = pd.DataFrame(list(ex.map(cell, jobs, chunksize=16)))
    s = df.groupby(["scenario", "T"], sort=False)[["joint", "gro"]].mean().reset_index()
    s.to_csv(OUT / "gro_sim.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
