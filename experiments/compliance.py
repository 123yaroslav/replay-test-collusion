"""Plan compliance in leader-follower conditions: how often P1 plays the move its previous private
message assigned to itself, and P3 the move assigned to it. Also how varied the plans are.

    uv run python experiments/compliance.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from collusion.trajectory import read_episodes

ROOT = Path(__file__).resolve().parents[1]
PAT = {"s1": re.compile(r"P1\s*=\s*(rock|paper|scissors).*?P3\s*=\s*(rock|paper|scissors)",
                        re.IGNORECASE | re.DOTALL),
       "s2": re.compile(r"P1\s*=\s*(\d+).*?P3\s*=\s*(\d+)", re.DOTALL)}


def main():
    rows = []
    for exp in ("s1", "s2"):
        for f in sorted((ROOT / "results" / "raw" / exp).glob("*/pair_*.jsonl")):
            tag = f.parent.name
            if tag.startswith("pilot") or f.stem not in ("pair_hidden", "pair_aware"):
                continue
            lead = fol = n = rounds = same = 0
            plans = set()
            for ep in read_episodes(f):
                for prev, cur in zip(ep.rounds[:-1], ep.rounds[1:]):
                    rounds += 1
                    m = [x for x in prev.messages if x.sender == "P1"]
                    mm = PAT[exp].search(m[0].text) if m else None
                    if not mm:
                        continue
                    a, b = mm.group(1).lower(), mm.group(2).lower()
                    if exp == "s2":
                        a, b = int(a), int(b)
                    n += 1
                    same += a == b
                    plans.add((ep.seed, a, b))
                    lead += cur.actions["P1"] == a
                    fol += cur.actions["P3"] == b
            rows.append({"exp": exp, "tag": tag, "condition": f.stem, "rounds": rounds,
                         "plans_parsed": n, "leader_follows": round(lead / max(n, 1), 3),
                         "follower_follows": round(fol / max(n, 1), 3),
                         "plans_same_move": round(same / max(n, 1), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "summary" / "compliance.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
