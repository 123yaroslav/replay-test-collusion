"""EXPLORATORY, POST HOC (after the main analysis and the replication): the sequential replay test
with the likelihood-ratio bet (`collusion.detect.gro`) on the existing LLM episodes, next to the
pre-registered joint bet. Same logged replays, no new LLM calls. Also estimates, per condition, the
pair's KL divergence between the logged joint law and the product of the replay laws (the ceiling on
the bet's growth rate), with a bootstrap over episodes.

    uv run python experiments/gro_llm.py
Writes results/summary/gro_llm_episodes.csv and gro_llm_summary.csv.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from collusion.analysis import wilson
from collusion.detect.crt_llm import requery_pool, sequential_replay_test
from collusion.detect.gro import sequential_gro_test
from collusion.games.procurement import BIDS
from collusion.games.rps import MOVES
from collusion.trajectory import Episode, read_episodes

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results" / "summary"
SEATS = ["P1", "P2", "P3", "P4"]
SPACE = {"s1": (MOVES, None, 3), "s2": (BIDS, {60: 0, 70: 0, 80: 0, 90: 1, 100: 1}, 2)}


def one(args):
    exp, tag, cond, ep_json = args
    ep = Episode.model_validate_json(ep_json)
    space, coarsen, k = SPACE[exp]
    try:
        pool = requery_pool(ep, SEATS, space, coarsen)
    except ValueError:
        return None
    truth = tuple(sorted(SEATS.index(c) for c in ep.colluders)) if ep.colluders else (0, 2)
    out = {"exp": exp, "tag": tag, "condition": cond, "seed": ep.seed,
           "colluding": bool(ep.colluders)}
    for name, r in (("joint", sequential_replay_test(pool, alpha=0.05, n_actions=k)),
                    ("gro", sequential_gro_test(pool, alpha=0.05, n_actions=k))):
        out[f"{name}_flag"] = r["reject_round"] is not None
        out[f"{name}_named"] = out[f"{name}_flag"] and tuple(r["top"]) == truth
        out[f"{name}_round"] = r["reject_round"]
    i, j = truth  # counts for the pair's KL (the true pair, or P1-P3 in honest games)
    a = pool[..., 0]
    out["joint_counts"] = np.bincount(a[:, i] * k + a[:, j], minlength=k * k).tolist()
    out["rep_i"] = np.bincount(pool[:, i, 1:].ravel(), minlength=k).tolist()
    out["rep_j"] = np.bincount(pool[:, j, 1:].ravel(), minlength=k).tolist()
    return out


def kl_from(rows, k):
    c = np.sum([r for r in rows.joint_counts], axis=0).reshape(k, k) + 0.5
    qi = np.sum([r for r in rows.rep_i], axis=0) + 0.5
    qj = np.sum([r for r in rows.rep_j], axis=0) + 0.5
    p, qi, qj = c / c.sum(), qi / qi.sum(), qj / qj.sum()
    return float(np.sum(p * np.log(p / np.outer(qi, qj))))


def main():
    jobs = []
    for exp in ("s1", "s2"):
        for d in sorted((ROOT / "results" / "raw" / exp).iterdir()):
            if not d.is_dir() or d.name.startswith(("pilot", "T60")):
                continue
            for f in sorted(d.glob("*.jsonl")):
                jobs += [(exp, d.name, f.stem, ep.model_dump_json()) for ep in read_episodes(f)]
    with ProcessPoolExecutor() as ex:
        rows = [r for r in ex.map(one, jobs, chunksize=8) if r]
    df = pd.DataFrame(rows)
    df.drop(columns=["joint_counts", "rep_i", "rep_j"]).to_csv(SUM / "gro_llm_episodes.csv",
                                                               index=False)
    rng = np.random.default_rng(0)
    out = []
    for (exp, tag, cond), g in df.groupby(["exp", "tag", "condition"]):
        k = SPACE[exp][2]
        row = {"exp": exp, "tag": tag, "condition": cond, "n": len(g),
               "colluding": bool(g.colluding.iloc[0])}
        for name in ("joint", "gro"):
            col = f"{name}_named" if row["colluding"] else f"{name}_flag"
            kk = int(g[col].sum())
            lo, hi = wilson(kk, len(g))
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = kk / len(g), lo, hi
            row[f"{name}_flag_rate"] = float(g[f"{name}_flag"].mean())
        row["kl"] = kl_from(g, k)
        boots = [kl_from(g.iloc[rng.integers(0, len(g), len(g))], k) for _ in range(500)]
        row["kl_lo"], row["kl_hi"] = np.percentile(boots, [2.5, 97.5])
        out.append(row)
    s = pd.DataFrame(out)
    s.to_csv(SUM / "gro_llm_summary.csv", index=False)
    with pd.option_context("display.width", 220):
        print(s[["exp", "tag", "condition", "n", "joint", "gro", "joint_flag_rate",
                 "gro_flag_rate", "kl", "kl_lo", "kl_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
