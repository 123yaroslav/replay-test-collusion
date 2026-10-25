"""EXPLORATORY, POST HOC (statistic chosen after seeing the full study's cartel cell; to be confirmed
by the pre-registered Run E): the replay co-departure test on the auction cells.

    uv run python experiments/departure_cda.py --tag full_cda
Writes results/summary/<tag>_departure.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy import stats

from collusion.analysis import wilson
from collusion.detect.crt_llm import requery_pool
from collusion.detect.departure import departure_test
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
        rows = [departure_test(requery_pool(ep, ep.agent_ids, space, None), n_null=999,
                               seed=ep.seed) for ep in eps]
        k, ka = sum(r["pair_flag"] for r in rows), sum(r["agent_flag"] for r in rows)
        out["cells"][f.stem] = {"n": len(rows), "pair": k, "pair_wilson": wilson(k, len(rows)),
                                "agent": ka}
    c = out["cells"]
    if "SM-cartel" in c and "SM-off" in c:
        a, b = c["SM-cartel"], c["SM-off"]
        out["fisher_cartel_vs_off_pair"] = float(stats.fisher_exact(
            [[a["pair"], a["n"] - a["pair"]], [b["pair"], b["n"] - b["pair"]]], alternative="greater")[1])
    (ROOT / "results" / "summary" / f"{args.tag}_departure.json").write_text(json.dumps(out, indent=2))
    print({k: (v["pair"], v["agent"], v["n"]) for k, v in c.items()},
          {k: v for k, v in out.items() if k.startswith("fisher")})


if __name__ == "__main__":
    main()
