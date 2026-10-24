"""Bonjour et al.'s fixed rule at its own design size (pre-registered): pool blocks of 10 episodes
of the same (tag, condition), i.e. >= 300 rounds, and apply the rule once per block.

    uv run python experiments/bonjour_pooled.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from collusion.analysis import wilson
from collusion.detect.crt_llm import action_matrix
from collusion.detect.net_influence import bonjour_rule, influence_matrix, net_influence
from collusion.games.rps import MOVES
from collusion.trajectory import read_episodes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "summary"
SEATS = ["P1", "P2", "P3", "P4"]
BLOCK = 10


def main():
    rows = []
    for d in sorted((ROOT / "results" / "raw" / "s1").iterdir()):
        if not d.is_dir() or d.name.startswith(("pilot", "T60")):
            continue
        for f in sorted(d.glob("*.jsonl")):
            eps = read_episodes(f)
            truth = tuple(sorted(eps[0].colluders)) if eps and eps[0].colluders else None
            for b in range(len(eps) // BLOCK):
                block = eps[b * BLOCK:(b + 1) * BLOCK]
                a = np.vstack([action_matrix(e, SEATS, MOVES) for e in block])
                top = bonjour_rule(net_influence(influence_matrix(a)))
                pair = (SEATS[top[0]], SEATS[top[1]]) if top else None
                rows.append({"tag": d.name, "condition": f.stem, "block": b, "rounds": len(a),
                             "flag": top is not None,
                             "correct": pair is not None and tuple(sorted(pair)) == truth})
    df = pd.DataFrame(rows)
    out = []
    for (tag, cond), g in df.groupby(["tag", "condition"]):
        k, n = int(g.flag.sum()), len(g)
        lo, hi = wilson(k, n)
        out.append({"tag": tag, "condition": cond, "blocks": n, "rounds_per_block": g.rounds.min(),
                    "flag_rate": k / n, "lo": lo, "hi": hi, "correct_rate": g.correct.mean()})
    s = pd.DataFrame(out)
    s.to_csv(OUT / "bonjour_pooled.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
