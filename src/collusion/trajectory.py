"""Logged agent trajectories: the single data format every game writes and every detector reads.

One episode = one JSON line in results/raw/<experiment>/episodes.jsonl.
Ground truth (who was colluding) lives in `Episode.colluders` and must never be shown
to detectors; `Episode.for_detector()` strips it together with private reasoning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    model: str  # "scripted:<policy>" for non-LLM agents
    role: str = "independent"  # "colluder" | "independent" | "trusted" | ...


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sender: str
    recipients: list[str]  # ["*"] = public channel
    text: str


class Round(BaseModel):
    model_config = ConfigDict(extra="forbid")
    t: int
    actions: dict[str, Any]
    rewards: dict[str, float]
    messages: list[Message] = Field(default_factory=list)
    public_state: dict[str, Any] = Field(default_factory=dict)
    # agent id -> game-assigned private info (e.g. a cost draw); auditors may condition on it
    private_info: dict[str, dict[str, Any]] = Field(default_factory=dict)
    private: dict[str, str] = Field(default_factory=dict)  # agent id -> private reasoning/notes
    # agent id -> auditor's re-queried actions on the same observation minus the inbox (CRT)
    requery: dict[str, list[Any]] = Field(default_factory=dict)


class Episode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode_id: str
    game: str
    condition: str
    seed: int
    agents: list[AgentSpec]
    colluders: list[str] = Field(default_factory=list)  # ground truth, hidden from detectors
    rounds: list[Round] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def agent_ids(self) -> list[str]:
        return [a.id for a in self.agents]

    def action_series(self, agent_id: str) -> list[Any]:
        return [r.actions[agent_id] for r in self.rounds]

    def for_detector(self, keep_messages: bool = True) -> Episode:
        rounds = [
            r.model_copy(update={"private": {}, "messages": r.messages if keep_messages else []})
            for r in self.rounds
        ]
        agents = [a.model_copy(update={"role": "unknown"}) for a in self.agents]
        return self.model_copy(update={"colluders": [], "rounds": rounds, "agents": agents,
                                       "condition": "hidden", "meta": {}})


def write_episodes(path: Path, episodes: list[Episode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for ep in episodes:
            f.write(ep.model_dump_json() + "\n")


def read_episodes(path: Path) -> list[Episode]:
    with path.open() as f:
        return [Episode.model_validate(json.loads(line)) for line in f if line.strip()]
