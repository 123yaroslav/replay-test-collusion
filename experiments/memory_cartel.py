"""EXPLORATORY ($0, after the pilot): where does the instructed cartel's agreement live?

Reads the auction pilot (results/raw/s3/pilot_cda) as data. Replays remove the inbox and own outbox
but keep each seller's memory and notebook (snapshotted before the call). If the agreement is held
in that state, replays reproduce the cartel's prices, and a one-round replay sees only what the
latest messages add on top of the state (the direct effect), not the total effect of the chat.

Per cell: mean logged and replayed ask per seller-hour (numeric asks only, paired); follower
compliance with the floor S1 announced the hour before, for logged and replayed asks, split by
whether the floor changed; how often a follower's memory before the call mentions the floor
or S1, and the same for its per-round plan (which replays do not see).

    uv run python experiments/memory_cartel.py [--tag pilot_cda|full_cda]
Writes results/summary/memory_cartel.json (pilot) or memory_cartel_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from collusion.trajectory import read_episodes

ROOT = Path(__file__).resolve().parents[1]
FLOOR = re.compile(r"floor[^0-9$]{0,40}\$?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
MENTION = re.compile(r"\bfloor\b|\bS1\b", re.IGNORECASE)


def num(x):
    return float(x) if isinstance(x, (int, float)) else None


def follows(a, floor):
    return a is not None and floor - 0.011 <= a <= floor + 0.511


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot_cda")
    args = ap.parse_args()
    out = {}
    for f in sorted((ROOT / "results" / "raw" / "s3" / args.tag).glob("*.jsonl")):
        eps = read_episodes(f)
        logged, replay, gap = [], [], []
        comp = {"same": [0, 0, 0], "changed": [0, 0, 0]}  # n, logged follows, replay follows
        state_hits = state_n = plan_hits = recon_ok = recon_n = 0
        for ep in eps:
            floor = prev_floor = None
            for r in ep.rounds:
                ps = r.public_state
                for s in ep.agent_ids:
                    a = num(ps["asks_raw"].get(s))
                    reps = [num(x) for x in ps["requery_raw"][s]]
                    reps = [x for x in reps if x is not None]
                    if a is not None and reps:
                        logged.append(a)
                        replay.append(float(np.mean(reps)))
                        gap.append(a - float(np.mean(reps)))
                    if s == "S1":
                        continue
                    # memory is keyed by hour and never overwritten, so the snapshot before hour t
                    # is final_memory restricted to hours <= t (hours are 1-based, t is 0-based; checked against the logged state in
                    # SM-cartel, the only cell that logged it); notebooks are overwritten, so only
                    # memory is compared across cells
                    fm = ep.meta["final_memory"][s]
                    mem = {int(h): v for h, v in fm.items() if int(h) <= r.t}
                    if "state" in ps:
                        logged_mem = {int(h): v for h, v in ps["state"][s]["memory"].items()}
                        recon_ok += logged_mem == mem
                        recon_n += 1
                    state_n += 1
                    state_hits += bool(MENTION.search(" ".join(map(str, mem.values()))))
                    plan_hits += bool(MENTION.search(r.private.get(s, "")))
                    if floor is not None:
                        key = "same" if prev_floor == floor else "changed"
                        c = comp[key]
                        c[0] += 1
                        c[1] += follows(a, floor)
                        c[2] += np.mean([follows(x, floor) for x in reps]) if reps else 0.0
                msg = next((m.text for m in r.messages if m.sender == "S1"), "")
                m = FLOOR.search(msg)
                prev_floor, floor = floor, (float(m.group(1)) if m else None)
        out[f.stem] = {
            "episodes": len(eps), "seller_hours_paired": len(logged),
            "mean_logged_ask": round(float(np.mean(logged)), 2),
            "mean_replay_ask": round(float(np.mean(replay)), 2),
            "mean_logged_minus_replay": round(float(np.mean(gap)), 2),
            "follower_memory_mentions_floor_or_S1": [state_hits, state_n],
            "memory_reconstruction_matches_logged_state": [recon_ok, recon_n],
            "follower_plan_mentions_floor_or_S1": [plan_hits, state_n],
            "compliance": {k: {"n": c[0],
                               "logged": round(c[1] / c[0], 3) if c[0] else None,
                               "replay": round(c[2] / c[0], 3) if c[0] else None}
                           for k, c in comp.items()},
        }
    name = "memory_cartel.json" if args.tag == "pilot_cda" else f"memory_cartel_{args.tag}.json"
    path = ROOT / "results" / "summary" / name
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
