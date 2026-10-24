"""EXPLORATORY (post-freeze): the anytime-valid sequential replay test on the LLM episodes.

Uses the same logged re-queries as the pre-registered test (no new LLM calls). S2 uses the same
high/low bid coarsening as its pre-registered statistic.

    uv run python experiments/sequential_llm.py
Writes results/summary/seq_llm_episodes.csv and seq_llm_summary.csv.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from collusion.analysis import wilson
from collusion.detect.crt_llm import requery_pool, sequential_replay_test
from collusion.games.procurement import BIDS
from collusion.games.rps import MOVES
from collusion.trajectory import Episode, read_episodes

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results" / "summary"
BETTOR = os.environ.get("BETTOR", "joint")
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
    r = sequential_replay_test(pool, alpha=0.05, n_actions=k, bettor=BETTOR)
    truth = tuple(sorted(SEATS.index(c) for c in ep.colluders)) if ep.colluders else None
    return {"exp": exp, "tag": tag, "condition": cond, "seed": ep.seed,
            "reject": r["reject_round"] is not None, "reject_round": r["reject_round"],
            "correct": r["reject_round"] is not None and truth is not None
            and tuple(r["top"]) == truth, "final_e": r["final_e"], "bits_top": r["bits_top"]}


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
    sfx = "" if BETTOR == "joint" else "_" + BETTOR
    df.to_csv(SUM / f"seq_llm_episodes{sfx}.csv", index=False)
    out = []
    for (exp, tag, cond), g in df.groupby(["exp", "tag", "condition"]):
        k, n = int(g.reject.sum()), len(g)
        lo, hi = wilson(k, n)
        out.append({"exp": exp, "tag": tag, "condition": cond, "n": n, "reject": k / n,
                    "lo": lo, "hi": hi, "correct": g.correct.mean(),
                    "median_round": g.reject_round.median()})
    s = pd.DataFrame(out)
    s.to_csv(SUM / f"seq_llm_summary{sfx}.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
