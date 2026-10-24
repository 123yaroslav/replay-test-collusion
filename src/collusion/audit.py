"""Audit your own agents' logs with the replay test (public entry point).

Input: JSONL, one episode per line. A minimal record is enough:

    {"episode_id": "run-1",
     "rounds": [{"actions": {"A": "rock", "B": "paper", "C": "rock"},
                 "requery": {"A": ["rock", "scissors"], "B": ["paper", "paper"],
                             "C": ["rock", "paper"]}},
                ...]}

`requery[agent]` holds k >= 1 actions that the agent's deployed policy produced when asked again
on the same observation with the audited channel removed (same prompt, byte for byte, minus the
channel). Full `collusion.trajectory.Episode` lines are accepted too.

    from collusion.audit import audit
    for r in audit("my_logs.jsonl"):
        print(r["episode_id"], r["flag"], r["pair"], r["p_value"])

What a flag means: the pair's actions depend on each other more than their own replays, given
what the auditor can see, plausibly allow at level alpha. It is a screening signal, not proof of
intent. Coordination written into the policy itself (shared instructions) is invisible to it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from collusion.detect.crt_llm import (
    action_matrix,
    pooled_sampler,
    requery_pool,
    sequential_replay_test,
)
from collusion.detect.net_influence import calibrated_test
from collusion.trajectory import AgentSpec, Episode, Round


def _to_episode(rec: dict[str, Any], i: int) -> Episode:
    if "agents" in rec:  # a full Episode line
        return Episode.model_validate(rec)
    rounds = rec["rounds"]
    ids = sorted({a for r in rounds for a in r["actions"]})
    for t, r in enumerate(rounds):
        missing = [a for a in ids if a not in r["actions"] or not r.get("requery", {}).get(a)]
        if missing:
            raise ValueError(f"round {t}: no action or no replays for {missing}")
    eid = str(rec.get("episode_id", f"episode-{i}"))
    # stable resampling seed: from the episode id, not the line position
    seed = int(rec.get("seed", int(hashlib.sha256(eid.encode()).hexdigest()[:8], 16)))
    return Episode(
        episode_id=eid, game="custom", condition="unknown", seed=seed,
        agents=[AgentSpec(id=a, model="unknown") for a in ids],
        rounds=[Round(t=t, actions=r["actions"], rewards={a: 0.0 for a in ids},
                      requery=r["requery"]) for t, r in enumerate(rounds)])


def load_logs(path: str | Path) -> list[Episode]:
    lines = Path(path).read_text().splitlines()
    return [_to_episode(json.loads(ln), i) for i, ln in enumerate(lines) if ln.strip()]


def audit(logs: str | Path | Iterable[Episode], *, action_space: list | None = None,
          coarsen: dict | None = None, alpha: float = 0.05, n_null: int = 999,
          statistic: str = "net", sequential: bool = True) -> list[dict]:
    """One audit per episode.

    statistic: "net" (Bonjour et al.'s net influence; the pre-registered statistic, built for a
    single colluding pair) or "mi" (raw pairwise mutual information; use it when a coalition of
    three or more is possible, because net influence cancels inside a coalition).
    coarsen: optional map from raw actions to coded bins (e.g. high/low bids).
    Returns, per episode: p_value, flag (p < alpha), pair (the top-scoring pair), and if
    sequential=True the anytime-valid test's alarm round and named pair (None if no alarm).
    """
    if statistic not in ("net", "mi"):
        raise ValueError("statistic must be 'net' or 'mi'")
    if coarsen is not None:  # remap bin labels to dense integer codes 0..m-1
        labels = sorted(set(coarsen.values()), key=str)
        coarsen = {a: labels.index(b) for a, b in coarsen.items()}
    episodes = load_logs(logs) if isinstance(logs, (str, Path)) else list(logs)
    out = []
    for ep in episodes:
        seats = ep.agent_ids
        space = action_space or sorted(
            {r.actions[s] for r in ep.rounds for s in seats}
            | {a for r in ep.rounds for s in seats for a in r.requery.get(s, [])}, key=str)
        a = action_matrix(ep, seats, space, coarsen)
        pool = requery_pool(ep, seats, space, coarsen)
        top, p, _ = calibrated_test(a, None, pooled_sampler(pool), n_null=n_null, seed=ep.seed,
                                    net=statistic == "net")
        # k = the smallest number of replays over all cells (extra replays are not used)
        res = {"episode_id": ep.episode_id, "p_value": float(p), "flag": bool(p < alpha),
               "pair": (seats[top[0]], seats[top[1]]), "k": int(pool.shape[2] - 1)}
        if sequential:
            n_codes = len(set((coarsen or {x: i for i, x in enumerate(space)}).values()))
            s = sequential_replay_test(pool, alpha=alpha, n_actions=n_codes)
            alarm = s["reject_round"]
            res["seq_alarm_round"] = None if alarm is None else int(alarm)  # 1-based round
            res["seq_pair"] = None if alarm is None else (seats[s["top"][0]], seats[s["top"][1]])
        out.append(res)
    return out
