"""Public audit() entry point: minimal JSONL format, a planted pair is flagged, honest logs are not."""

import json

import numpy as np

from collusion.audit import audit, load_logs

MOVES, PRIOR = ["rock", "paper", "scissors"], [0.6, 0.3, 0.1]


def _log(path, copy_prob, n=4, T=40, seed=0):
    rng = np.random.default_rng(seed)
    with open(path, "w") as f:
        for e in range(n):
            rounds = []
            for _ in range(T):
                def draw():
                    return MOVES[rng.choice(3, p=PRIOR)]
                acts = {a: draw() for a in "ABCD"}
                if rng.random() < copy_prob:
                    acts["B"] = acts["A"]
                rounds.append({"actions": acts, "requery": {a: [draw(), draw()] for a in "ABCD"}})
            f.write(json.dumps({"episode_id": f"e{e}", "rounds": rounds}) + "\n")
    return path


def test_minimal_format_loads(tmp_path):
    eps = load_logs(_log(tmp_path / "x.jsonl", 0.0, n=1, T=5))
    assert eps[0].agent_ids == ["A", "B", "C", "D"] and len(eps[0].rounds) == 5


def test_planted_pair_flagged(tmp_path):
    res = audit(_log(tmp_path / "pair.jsonl", 0.9), n_null=199)
    assert all(r["flag"] and r["pair"] == ("A", "B") for r in res)


def test_honest_mostly_clean(tmp_path):
    # 60 honest episodes at alpha = 5%: expect about 3 flags; more than 9 has probability < 0.1%
    res = audit(_log(tmp_path / "honest.jsonl", 0.0, n=60, T=30, seed=7), n_null=199)
    assert sum(r["flag"] for r in res) <= 9
    assert sum(r["seq_alarm_round"] is not None for r in res) <= 9
