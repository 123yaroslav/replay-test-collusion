"""EXPLORATORY (run only after the pre-registered experiments/confirm_cda.py): the sequential test
with the likelihood-ratio bet on the auction cells, next to the pre-registered joint bet. Same logged
replays, no new calls. Fisher exact test of SM-cartel against SM-off alarm counts for each bet
(the joint-bet comparison is C4 of the plan; the likelihood-ratio one is exploratory, predicted in
docs/fresh_plan.md before the cartel data existed).

    uv run python experiments/gro_cda.py --tag full_cda
Writes results/summary/<tag>_gro.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy import stats

from collusion.analysis import wilson
from collusion.detect.crt_llm import requery_pool, sequential_replay_test
from collusion.detect.gro import sequential_gro_test
from collusion.games.cda import ACTION_SPACES
from collusion.trajectory import read_episodes

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full_cda")
    args = ap.parse_args()
    out = {"tag": args.tag, "cells": {}}
    for f in sorted((ROOT / "results" / "raw" / "s3" / args.tag).glob("*.jsonl")):
        eps = read_episodes(f)
        space = ACTION_SPACES[eps[0].meta.get("bins", "v1")]
        k = len(space)
        rows = []
        for ep in eps:
            pool = requery_pool(ep, ep.agent_ids, space, None)
            j = sequential_replay_test(pool, alpha=0.05, n_actions=k)
            g = sequential_gro_test(pool, alpha=0.05, n_actions=k)
            rows.append({"seed": ep.seed, "joint": j["reject_round"] is not None,
                         "gro": g["reject_round"] is not None,
                         "gro_round": g["reject_round"], "gro_top": list(g["top"]),
                         "gro_bits": float(g["log_e_pairs"][:, -1].max() / 0.6931)})
        n = len(rows)
        out["cells"][f.stem] = {"n": n, "bins": eps[0].meta.get("bins"),
                                "joint": sum(r["joint"] for r in rows),
                                "gro": sum(r["gro"] for r in rows),
                                "gro_wilson": wilson(sum(r["gro"] for r in rows), n),
                                "rows": rows}
    c = out["cells"]
    if "SM-cartel" in c and "SM-off" in c:
        for bet in ("joint", "gro"):
            a, b = c["SM-cartel"], c["SM-off"]
            p = stats.fisher_exact([[a[bet], a["n"] - a[bet]], [b[bet], b["n"] - b[bet]]],
                                   alternative="greater")[1]
            out[f"fisher_cartel_vs_off_{bet}"] = float(p)
    (ROOT / "results" / "summary" / f"{args.tag}_gro.json").write_text(json.dumps(out, indent=2))
    print({cell: {k: v for k, v in d.items() if k != "rows"} for cell, d in c.items()},
          {k: v for k, v in out.items() if k.startswith("fisher")})


if __name__ == "__main__":
    main()
