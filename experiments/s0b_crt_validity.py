"""S0b ($0): validity and power of the per-episode CRT on the scripted RPS simulator.

Part 1 (calibration): honest herd/uniform populations. The reference policy is either known
(oracle), estimated from k re-queries (plug-in), or pooled with the logged action (what the LLM
experiments use). Also a strong-feedback herd (herd_weight 0.9, memory 1) where later contexts
depend heavily on earlier actions, the case the fixed-horizon CRT has no finite-sample guarantee
for.
Part 2 (power): colluding pair whose follower copies the leader's same-round move w.p. cp, for
log lengths T: the detectability frontier shown in the report.

    uv run python experiments/s0b_crt_validity.py [--iters 300]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from collusion.detect.crt_llm import pooled_sampler
from collusion.detect.net_influence import calibrated_test
from collusion.games.rps import Population, crt_resampler, simulate

OUT = Path(__file__).resolve().parents[1] / "results" / "summary"
HERD = ("herd_x", "herd_x", "herd_y", "herd_y")
POPS = {
    "herd": Population(types=HERD),
    "uniform": Population(types=("uniform",) * 4),
    "herd_strong_feedback": Population(types=HERD, herd_weight=0.9, memory=1),
}


def draws_from(ref: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    cdf = ref.cumsum(axis=2)
    u = rng.random(ref.shape[:2] + (k,))
    return (u[..., None] > cdf[:, :, None, :]).sum(axis=3).clip(max=2)


def calib_cell(args):
    pop, T, k, it = args
    rng = np.random.default_rng(1_000_003 * T + 7_919 * it + 31 * k + hash(pop) % 997)
    a, ref, _ = simulate(POPS[pop], T, rng)
    d = draws_from(ref, k, rng)
    variants = {"oracle": crt_resampler(ref), "plugin": pooled_sampler(d),
                "pooled": pooled_sampler(np.concatenate([a[..., None], d], axis=2))}
    rows = []
    for name, s in variants.items():
        _, p, _ = calibrated_test(a, None, s, n_null=199, seed=it, net=True)
        rows.append({"population": pop, "T": T, "k": k, "variant": name, "p": p})
    return rows


def power_cell(args):
    cp, T, it, k = args
    pop = Population(types=HERD, colluders=(0, 2), cp=cp) if cp > 0 else Population(types=HERD)
    rng = np.random.default_rng(99_991 * T + 31 * it + int(cp * 100))
    a, ref, _ = simulate(pop, T, rng)
    pool = np.concatenate([a[..., None], draws_from(ref, k, rng)], axis=2)
    top, p, _ = calibrated_test(a, None, pooled_sampler(pool), n_null=199, seed=it, net=True)
    return {"cp": cp, "T": T, "p": p, "flag": p < 0.05,
            "correct": bool(p < 0.05 and cp > 0 and top == (0, 2))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(pop, T, k, it) for pop in POPS for T in (30,) for k in (2, 4, 8)
            for it in range(args.iters)]
    with ProcessPoolExecutor() as ex:
        cal = pd.DataFrame([r for rs in ex.map(calib_cell, jobs, chunksize=8) for r in rs])
    cal.to_csv(OUT / "s0b_calibration.csv", index=False)
    cal["fpr05"] = cal.p < 0.05
    cal["fpr01"] = cal.p < 0.01
    print("Calibration (T=30): false-alarm rate at alpha=0.05 / 0.01")
    print(cal.pivot_table(index=["population", "k"], columns="variant",
                          values=["fpr05", "fpr01"]).round(3).to_string())
    pjobs = [(cp, T, it, 2) for cp in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) for T in (20, 30, 60, 120)
             for it in range(args.iters // 2)]
    with ProcessPoolExecutor() as ex:
        pw = pd.DataFrame(list(ex.map(power_cell, pjobs, chunksize=8)))
    pw.to_csv(OUT / "s0b_power.csv", index=False)
    pw["value"] = np.where(pw.cp == 0, pw.flag, pw.correct)
    print("\nPower, pooled k=2 (cp=0: false-alarm rate; cp>0: correct-pair rate at 0.05)")
    print(pw.pivot_table(index="cp", columns="T", values="value").round(2).to_string())


if __name__ == "__main__":
    main()
