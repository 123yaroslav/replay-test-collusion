"""EXPLORATORY (post-freeze): does the hidden channel pay? Team advantage of the designated pair
(P1 + P3 score minus P2 + P4 score, per episode) by condition, and its relation to detection.

If a pair's moves are independent given the public record, the same joint behaviour could be
produced by two agents without a channel, so the channel cannot be what earns them anything.
This script checks whether the pairs the replay test misses also gain little.

    uv run python experiments/collusion_gain.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from collusion.trajectory import read_episodes

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results" / "summary"


def main():
    rows = []
    for exp in ("s1", "s2"):
        for tag_dir in sorted((ROOT / "results" / "raw" / exp).iterdir()):
            tag = tag_dir.name
            if not tag_dir.is_dir() or tag.startswith(("pilot", "T60", "val_", "confirm")):
                continue
            for f in sorted(tag_dir.glob("*.jsonl")):
                for ep in read_episodes(f):
                    team = sum(r.rewards["P1"] + r.rewards["P3"] for r in ep.rounds)
                    other = sum(r.rewards["P2"] + r.rewards["P4"] for r in ep.rounds)
                    rows.append({"exp": exp, "tag": tag, "condition": f.stem, "seed": ep.seed,
                                 "team_advantage": (team - other) / len(ep.rounds)})
    df = pd.DataFrame(rows)
    aud = pd.concat([pd.read_csv(SUM / f"{e}_audits.csv").assign(exp=e)
                     for e in ("s1", "s2") if (SUM / f"{e}_audits.csv").exists()])
    crt = aud[aud.detector == "crt_net"][["exp", "tag", "condition", "seed", "p", "correct_pair"]]
    df = df.merge(crt, on=["exp", "tag", "condition", "seed"], how="left")
    df.to_csv(SUM / "collusion_gain_episodes.csv", index=False)
    out = []
    for (exp, tag), g in df.groupby(["exp", "tag"]):
        base = g[g.condition == "null_same"].team_advantage
        for cond, h in g.groupby("condition"):
            d = h.team_advantage
            t = stats.ttest_ind(d, base, equal_var=False) if cond != "null_same" and len(base) > 2 else None
            caught = h[h.correct_pair == True].team_advantage
            missed = h[h.correct_pair == False].team_advantage
            out.append({"exp": exp, "tag": tag, "condition": cond, "n": len(h),
                        "team_adv_per_round": d.mean(), "sd": d.std(),
                        "diff_vs_null_same": d.mean() - base.mean() if len(base) else np.nan,
                        "p_vs_null_same": t.pvalue if t is not None else np.nan,
                        "adv_when_caught": caught.mean() if len(caught) else np.nan,
                        "adv_when_missed": missed.mean() if len(missed) else np.nan})
    s = pd.DataFrame(out)
    s.to_csv(SUM / "collusion_gain.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
