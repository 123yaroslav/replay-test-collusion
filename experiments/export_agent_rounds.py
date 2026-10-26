"""Flatten the episode logs into one table row per agent and round, for the Hugging Face viewer.

The raw logs (results/raw/<game>/<study>/<condition>.jsonl, one episode per line) stay the source
of every number in the report. This table is a convenience copy of the audited agents' rounds.
Pilot folders are left out. Double-auction buyers are left out because they have no replays.

    uv run --with pyarrow python experiments/export_agent_rounds.py
Writes results/tables/<game>/<split>.parquet and prints row counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "tables"

GAMES = {"s1": "rock_paper_scissors", "s2": "procurement_auction", "s3": "double_auction"}
SPLITS = {  # (game folder, study folder) -> split name
    ("s1", "deepseek"): "main", ("s1", "haiku"): "main", ("s1", "mixed"): "main",
    ("s1", "val_gpt-6-luna"): "validation_models", ("s1", "val_gemma-3-12b-it"): "validation_models",
    ("s1", "val_gpt-oss-20b"): "validation_models",
    ("s1", "confirm_haiku"): "replication",
    ("s1", "fresh_deepseek_pinned"): "fresh", ("s1", "fresh_haiku_stego"): "fresh",
    ("s1", "fresh_haiku_evasive"): "fresh",
    ("s2", "deepseek"): "main", ("s2", "fresh_haiku_auction"): "fresh",
    ("s3", "full_cda"): "main", ("s3", "fresh_cartel2"): "fresh",
}


def _num(x):
    """A price as a float. A seller that posted no ask is logged as "none" or "null" and gives None."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def rows_of(ep: dict, game: str, study: str) -> list[dict]:
    meta = ep.get("meta", {})
    undelivered = set(meta.get("undelivered", []))
    colluders = set(ep.get("colluders", []))
    if game == "double_auction" and ep["condition"] == "SM-cartel":
        colluders = {a["id"] for a in ep["agents"]}  # every seller is instructed, the raw field is empty
    model = {a["id"]: a["model"] for a in ep["agents"]}
    role = {a["id"]: a.get("role", "") for a in ep["agents"]}
    out = []
    for r in ep["rounds"]:
        ps = r.get("public_state", {})
        for agent, replays in r["requery"].items():
            sent = [{"to": m["recipients"], "text": m["text"]} for m in r["messages"]
                    if m["sender"] == agent]
            got = [{"from": m["sender"], "text": m["text"]} for m in r["messages"]
                   if m["sender"] != agent and m["sender"] not in undelivered
                   and (agent in m["recipients"] or "*" in m["recipients"])]
            act = r["actions"][agent]
            if game == "double_auction":  # actions are price bins, raw asks sit in public_state
                value, replay_values = ps["asks_raw"][agent], ps["requery_raw"][agent]
            elif game == "procurement_auction":
                value, replay_values = act, replays
            else:
                value, replay_values = None, None
            out.append({
                "game": game, "study": study, "condition": ep["condition"],
                "episode_id": ep["episode_id"], "seed": ep["seed"], "t": r["t"],
                "agent": agent, "model": model[agent], "role": role[agent],
                "colluder": agent in colluders,
                "action": str(act), "replays": [str(x) for x in replays],
                "action_value": _num(value),
                "replay_values": None if replay_values is None else [_num(x) for x in replay_values],
                "reward": _num(r["rewards"].get(agent)),
                "rationale": r.get("private", {}).get(agent, ""),
                "private_info": json.dumps(r.get("private_info", {}).get(agent, {})),
                "messages_sent": json.dumps(sent, ensure_ascii=False),
                "messages_received": json.dumps(got, ensure_ascii=False),
            })
    return out


def main() -> None:
    tables: dict[tuple[str, str], list[dict]] = {}
    for (g, study), split in SPLITS.items():
        for f in sorted((RAW / g / study).glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    ep = json.loads(line)
                    tables.setdefault((GAMES[g], split), []).extend(rows_of(ep, GAMES[g], study))
    for (game, split), rows in sorted(tables.items()):
        df = pd.DataFrame(rows)
        path = OUT / game / f"{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"{path.relative_to(ROOT)}  rows={len(df)}  episodes={df.episode_id.nunique()}  "
              f"{path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
