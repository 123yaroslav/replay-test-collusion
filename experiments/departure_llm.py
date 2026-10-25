"""EXPLORATORY, POST HOC: the replay co-departure test (`collusion.detect.departure`) on all existing
LLM games (rock-paper-scissors and the procurement auction; main, validation and replication
samples; pilots excluded). Same logged replays, no new calls.

    uv run python experiments/departure_llm.py
Writes results/summary/departure_llm.csv.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from collusion.analysis import wilson
from collusion.detect.crt_llm import requery_pool
from collusion.detect.departure import departure_test
from collusion.games.procurement import BIDS
from collusion.games.rps import MOVES
from collusion.trajectory import Episode, read_episodes

ROOT = Path(__file__).resolve().parents[1]
SEATS = ["P1", "P2", "P3", "P4"]
SPACE = {"s1": (MOVES, None), "s2": (BIDS, {60: 0, 70: 0, 80: 0, 90: 1, 100: 1})}


def one(args):
    exp, tag, cond, ep_json = args
    ep = Episode.model_validate_json(ep_json)
    space, coarsen = SPACE[exp]
    try:
        pool = requery_pool(ep, SEATS, space, coarsen)
    except ValueError:
        return None
    r = departure_test(pool, n_null=999, seed=ep.seed)
    truth = tuple(sorted(SEATS.index(c) for c in ep.colluders)) if ep.colluders else None
    return {"exp": exp, "tag": tag, "condition": cond, "seed": ep.seed,
            "colluding": truth is not None, "pair_flag": r["pair_flag"],
            "pair_named": r["pair_flag"] and tuple(r["top"]) == truth,
            "agent_flag": r["agent_flag"]}


def main():
    jobs = []
    for exp in ("s1", "s2"):
        for d in sorted((ROOT / "results" / "raw" / exp).iterdir()):
            if not d.is_dir() or d.name.startswith(("pilot", "T60", "smoke", "fresh")):
                continue
            for f in sorted(d.glob("*.jsonl")):
                jobs += [(exp, d.name, f.stem, ep.model_dump_json()) for ep in read_episodes(f)]
    with ProcessPoolExecutor() as ex:
        rows = [r for r in ex.map(one, jobs, chunksize=8) if r]
    df = pd.DataFrame(rows)
    out = []
    for (exp, tag, cond), g in df.groupby(["exp", "tag", "condition"]):
        k, n = int(g.pair_flag.sum()), len(g)
        lo, hi = wilson(k, n)
        out.append({"exp": exp, "tag": tag, "condition": cond, "n": n,
                    "colluding": bool(g.colluding.iloc[0]), "pair_flag": k / n, "lo": lo, "hi": hi,
                    "pair_named": g.pair_named.mean(), "agent_flag": g.agent_flag.mean()})
    s = pd.DataFrame(out)
    s.to_csv(ROOT / "results" / "summary" / "departure_llm.csv", index=False)
    with pd.option_context("display.width", 200):
        print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
