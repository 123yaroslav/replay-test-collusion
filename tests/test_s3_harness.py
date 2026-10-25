"""End-to-end harness check for experiments/s3_cda.py with a fake LLM (no network)."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import s3_cda

from collusion.audit import audit
from collusion.games.cda import ACTION_SPACES
from collusion.llm import ChatResult


class FakeLLM:
    def __init__(self):
        self.calls = []  # (prompt, sample)
        self.rng = np.random.default_rng(0)

    async def chat(self, model, messages, **kw):
        prompt = messages[0]["content"]
        self.calls.append((prompt, kw["sample"]))
        is_seller = "sales representative" in prompt
        reply = {"reasoning": "r", "plan": "p",
                 ("ask" if is_seller else "bid"): float(self.rng.uniform(80, 100)),
                 "new_memory": f"memory from sample {kw['sample']}",
                 "message_to_sellers": "keep prices high", "notebook_update": f"nb {kw['sample']}"}
        return ChatResult(json.dumps(reply), model, 10, 10, 0.0, False)


def _run(cell, hours=3):
    llm = FakeLLM()
    ep = asyncio.run(s3_cda.run_episode(llm, cell, seed=7, hours=hours, k=2))
    return llm, ep


def test_off_cell_replays_identical_and_state_not_written_back():
    llm, _ep = _run("SM-off")
    seller_prompts = [p for p, _ in llm.calls if "sales representative" in p]
    # per hour: 5 sellers x (1 real + 2 replays) = 15 seller prompts, in groups of 3 identical
    assert len(seller_prompts) == 15 * 3
    for g in range(0, len(seller_prompts), 3):
        assert seller_prompts[g] == seller_prompts[g + 1] == seller_prompts[g + 2]
    # memory in later prompts comes only from factual replies (sample draw 0), never replays
    for p in seller_prompts:
        for line in p.splitlines():
            if "memory from sample" in line:
                assert int(line.rsplit(" ", 1)[1]) // 4 % 10 == 0


def test_on_cell_replay_has_empty_inbox():
    llm, ep = _run("SM-on")
    seller_prompts = [p for p, _ in llm.calls if "sales representative" in p]
    hour2 = seller_prompts[15:18]  # S1, hour 2: real then two replays
    assert "keep prices high" in hour2[0]
    assert all("No messages from other sellers last hour." in p for p in hour2[1:])
    assert ep.rounds[0].messages and ep.rounds[0].messages[0].recipients


def test_episode_is_auditable():
    _, ep = _run("XP-on", hours=6)
    assert all(len(r.requery[s]) == 2 for r in ep.rounds for s in ep.agent_ids)
    res = audit([ep], action_space=ACTION_SPACES[ep.meta["bins"]], statistic="mi", n_null=99)
    assert 0 < res[0]["p_value"] <= 1
