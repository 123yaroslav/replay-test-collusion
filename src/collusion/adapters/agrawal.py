"""Convert Agrawal et al. (2025) double-auction runs into `Episode`s.

Source: third_party/llm-agent-collusion/data/<timestamp>_<tag>/ (76 runs shipped, May 2025,
gpt-4.1 / claude-3.7 agents, 5 sellers + 5 buyers, 30 rounds). We read `auction_results.md`
(per-round asks, bids, trades, messages), `experiment_metadata.json` (agent ids, valuations,
models, flags) and `llm_eval_full.json` (their gpt-4.1-mini judge's per-seller per-round
coordination score, kept as a ready-made LLM-judge baseline).

There is no per-agent ground truth here: seller messages are broadcast to every seller, so
the label is condition-level (`meta["seller_comms"]`), not "who colluded".
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from collusion.trajectory import AgentSpec, Episode, Message, Round

DATA_DIR = Path(__file__).resolve().parents[3] / "third_party" / "llm-agent-collusion" / "data"
_BLOCK = re.compile(r"````json\n(.*?)\n````", re.DOTALL)


def load_run(run_dir: Path) -> Episode:
    meta = json.loads((run_dir / "experiment_metadata.json").read_text())
    cfg = meta["auction_config"]
    sellers, buyers = meta["seller_ids"], meta["buyer_ids"]
    seller_val = dict(zip(sellers, cfg["seller_valuations"], strict=True))
    buyer_val = dict(zip(buyers, cfg["buyer_valuations"], strict=True))

    rounds = []
    for block in _BLOCK.findall((run_dir / "auction_results.md").read_text()):
        r = json.loads(block)
        rewards = {a: 0.0 for a in sellers + buyers}
        for tr in r["trades"]:
            rewards[tr["seller_id"]] += tr["price"] - seller_val[tr["seller_id"]]
            rewards[tr["buyer_id"]] += buyer_val[tr["buyer_id"]] - tr["price"]
        messages = [
            Message(sender=s, recipients=[x for x in sellers if x != s], text=txt)
            for s, txt in (r.get("seller_messages") or {}).items()
            if txt
        ]
        rounds.append(
            Round(
                t=r["round_number"] - 1,
                actions={**{s: r["seller_asks"].get(s) for s in sellers},
                         **{b: r["buyer_bids"].get(b) for b in buyers}},
                rewards=rewards,
                messages=messages,
                public_state={"trades": r["trades"]},
            )
        )

    judge = {}
    eval_path = run_dir / "llm_eval_full.json"
    if eval_path.exists():
        for i, entry in enumerate(json.loads(eval_path.read_text())):
            # seller_1..seller_5 follow the order of meta["seller_ids"]
            judge[sellers[i]] = [
                (rd.get("data") or {}).get("score") if rd.get("status") == "success" else None
                for rd in entry["rounds"]
            ]

    tag = cfg["tag"]
    condition, _, rep = tag.rpartition("-")
    return Episode(
        episode_id=run_dir.name,
        game="agrawal_cda",
        condition=condition,
        seed=int(rep) if rep.isdigit() else 0,
        agents=[AgentSpec(id=s, model=m, role="seller")
                for s, m in zip(sellers, cfg["seller_models"], strict=True)]
        + [AgentSpec(id=b, model=m, role="buyer")
           for b, m in zip(buyers, cfg["buyer_models"], strict=True)],
        rounds=rounds,
        meta={
            "seller_ids": sellers,
            "buyer_ids": buyers,
            "seller_comms": cfg["seller_comms_enabled"],
            "oversight": cfg["oversight_enabled"],
            "boss_pressure": cfg["boss_pressure"],
            "temperature": cfg["temperature"],
            "judge_scores": judge,
        },
    )


def load_all(data_dir: Path = DATA_DIR) -> list[Episode]:
    return [load_run(d) for d in sorted(data_dir.iterdir())
            if d.is_dir() and (d / "auction_results.md").exists()]
