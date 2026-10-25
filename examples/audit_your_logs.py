"""Audit a JSONL log with the replay test. Without an argument it writes and audits a toy log:
four agents share a prior (they move alike), and B secretly copies A in 80% of rounds.

    uv run python examples/audit_your_logs.py [my_logs.jsonl]
"""

import json
import sys

import numpy as np

from collusion.audit import audit

MOVES, PRIOR = ["rock", "paper", "scissors"], [0.6, 0.3, 0.1]  # shared prior: monoculture


def toy_log(path, n_episodes=3, T=30, seed=0):
    rng = np.random.default_rng(seed)
    with open(path, "w") as f:
        for e in range(n_episodes):
            rounds = []
            for _ in range(T):
                draw = lambda: MOVES[rng.choice(3, p=PRIOR)]
                acts = {a: draw() for a in "ABCD"}
                if rng.random() < 0.8:  # hidden channel: B plays what A played
                    acts["B"] = acts["A"]
                # replays: the same policy asked again without the channel (k = 2)
                rounds.append({"actions": acts, "requery": {a: [draw(), draw()] for a in "ABCD"}})
            f.write(json.dumps({"episode_id": f"toy-{e}", "rounds": rounds}) + "\n")
    return path


path = sys.argv[1] if len(sys.argv) > 1 else toy_log("toy_logs.jsonl")
for r in audit(path, action_space=["rock", "paper", "scissors"]):
    verdict = f"FLAG {r['pair']}" if r["flag"] else "no flag"
    print(f"{r['episode_id']}: {verdict}  p = {r['p_value']:.3f}  "
          f"sequential alarm: round {r['seq_alarm_round']}, pair {r['seq_pair']}")
