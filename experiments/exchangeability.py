"""EXPLORATORY diagnostic ($0, after the reviews): are logged moves exchangeable with their replays?

The replay test assumes that, without a channel, the logged action and its k re-queries are
exchangeable draws from one serving distribution. OpenRouter routing was not pinned, so this is
an assumption. If it holds, P(logged = replay) equals P(replay_1 = replay_2) in every cell.
We compare the two on cells whose logged prompt equals the replay prompt: every seat in
null_same / null_mixed / talk_no_action / told_no_channel (S1) and null_same (S2), and the two
seats outside the pair in pair_hidden (they receive no message). Difference per episode, then a
bootstrap over episodes.

    uv run python experiments/exchangeability.py
Writes results/summary/exchangeability.csv and exchangeability_providers.csv.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from collusion.trajectory import read_episodes

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results" / "summary"
HONEST = {"s1": ("null_same", "null_mixed", "talk_no_action", "told_no_channel"),
          "s2": ("null_same",)}


def episode_stats(ep, seats):
    log_rep, rep_rep, cells = 0.0, 0.0, 0
    for r in ep.rounds:
        for s in seats:
            reps = r.requery.get(s) or []
            if len(reps) < 2:
                continue
            a = r.actions[s]
            log_rep += ((a == reps[0]) + (a == reps[1])) / 2
            rep_rep += reps[0] == reps[1]
            cells += 1
    return log_rep, rep_rep, cells


def boot_ci(x, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    b = [rng.choice(x, len(x)).mean() for _ in range(n_boot)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    rows = []
    for exp in ("s1", "s2"):
        for d in sorted((ROOT / "results" / "raw" / exp).iterdir()):
            if not d.is_dir() or d.name.startswith(("pilot", "T60", "confirm")):
                continue
            for f in sorted(d.glob("*.jsonl")):
                if f.stem in HONEST[exp]:
                    group = "honest"
                elif exp == "s1" and f.stem == "pair_hidden":
                    group = "outside_pair"
                else:
                    continue
                for ep in read_episodes(f):
                    seats = [a.id for a in ep.agents]
                    if group == "outside_pair":
                        seats = [s for s in seats if s not in ep.colluders]
                    lr, rr, n = episode_stats(ep, seats)
                    if n:
                        rows.append({"exp": exp, "tag": d.name, "group": group,
                                     "condition": f.stem, "seed": ep.seed, "cells": n,
                                     "p_log_rep": lr / n, "p_rep_rep": rr / n})
    df = pd.DataFrame(rows)
    df["diff"] = df.p_log_rep - df.p_rep_rep
    out = []
    for (exp, tag, group), g in df.groupby(["exp", "tag", "group"]):
        lo, hi = boot_ci(g["diff"].to_numpy())
        out.append({"exp": exp, "tag": tag, "group": group, "episodes": len(g),
                    "cells": int(g.cells.sum()), "p_log_rep": g.p_log_rep.mean(),
                    "p_rep_rep": g.p_rep_rep.mean(), "diff": g["diff"].mean(),
                    "diff_lo": lo, "diff_hi": hi})
    s = pd.DataFrame(out)
    s.to_csv(SUM / "exchangeability.csv", index=False)
    print(s.round(4).to_string(index=False))

    con = sqlite3.connect(ROOT / "results" / "cache" / "llm.sqlite")
    prov = pd.read_sql("SELECT model, provider, COUNT(*) AS calls FROM spend GROUP BY model, provider",
                       con)
    prov["share"] = prov.calls / prov.groupby("model").calls.transform("sum")
    summ = prov.groupby("model").agg(providers=("provider", "nunique"), calls=("calls", "sum"),
                                     top_share=("share", "max")).reset_index()
    summ.to_csv(SUM / "exchangeability_providers.csv", index=False)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
