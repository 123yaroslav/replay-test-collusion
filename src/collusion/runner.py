"""Generic repeated N-player game loop shared by every environment.

Per round: every agent sees its observation (rules, own private info, public history and its
inbox), returns an action and optionally a private message; messages are delivered at the
start of the next round. One LLM call per agent per round.
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from collusion.trajectory import AgentSpec, Episode, Message, Round


@dataclass
class Decision:
    action: Any
    message: str | None = None  # delivered to the agent's channel partners next round
    private: str = ""  # reasoning/notes, logged but hidden from detectors


@dataclass
class Observation:
    agent_id: str
    t: int
    n_rounds: int
    private_info: dict[str, Any]
    history: list[Round]
    inbox: list[Message]
    action_space: list[Any]
    channel_partners: list[str] = field(default_factory=list)


class Game(ABC):
    name: str

    def __init__(self, n_rounds: int):
        self.n_rounds = n_rounds

    @abstractmethod
    def reset(self, agent_ids: list[str], rng: np.random.Generator) -> None: ...

    @abstractmethod
    def action_space(self, agent_id: str) -> list[Any]: ...

    def private_info(self, agent_id: str, t: int) -> dict[str, Any]:
        return {}

    def public_state(self, t: int) -> dict[str, Any]:
        return {}

    @abstractmethod
    def payoffs(self, actions: dict[str, Any], t: int) -> dict[str, float]: ...

    @abstractmethod
    def rules_text(self, agent_id: str) -> str:
        """Plain-language rules shown to LLM agents."""

    def history_text(self, agent_id: str, history: list[Round], window: int = 20) -> str:
        """Public history as shown to LLM agents: everyone's actions, your own reward."""
        if not history:
            return "No rounds played yet."
        lines = []
        for r in history[-window:]:
            acts = ", ".join(f"{a}={v}" for a, v in r.actions.items())
            lines.append(f"round {r.t + 1}: {acts}; your reward {r.rewards[agent_id]:g}")
        return "\n".join(lines)


class Agent(ABC):
    spec: AgentSpec

    @abstractmethod
    async def act(self, game: Game, obs: Observation) -> Decision: ...


async def run_episode(
    game: Game,
    agents: list[Agent],
    *,
    condition: str,
    seed: int,
    channels: dict[str, list[str]] | None = None,
    colluders: list[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> Episode:
    """Play one episode. `channels[a]` lists who receives a's private messages."""
    rng = np.random.default_rng(seed)
    ids = [a.spec.id for a in agents]
    channels = channels or {}
    game.reset(ids, rng)
    ep = Episode(
        episode_id=f"{game.name}-{condition}-{seed}-{uuid.uuid4().hex[:6]}",
        game=game.name,
        condition=condition,
        seed=seed,
        agents=[a.spec for a in agents],
        colluders=colluders or [],
        meta=meta or {},
    )
    pending: list[Message] = []
    for t in range(game.n_rounds):
        public = game.public_state(t)
        observations = [
            Observation(
                agent_id=a.spec.id,
                t=t,
                n_rounds=game.n_rounds,
                private_info=game.private_info(a.spec.id, t),
                history=ep.rounds,
                inbox=[m for m in pending if a.spec.id in m.recipients or "*" in m.recipients],
                action_space=game.action_space(a.spec.id),
                channel_partners=channels.get(a.spec.id, []),
            )
            for a in agents
        ]
        decisions = await asyncio.gather(*(a.act(game, o) for a, o in zip(agents, observations)))
        actions = {a.spec.id: d.action for a, d in zip(agents, decisions)}
        messages = [
            Message(sender=a.spec.id, recipients=channels[a.spec.id], text=d.message)
            for a, d in zip(agents, decisions)
            if d.message and channels.get(a.spec.id)
        ]
        ep.rounds.append(
            Round(
                t=t,
                actions=actions,
                rewards=game.payoffs(actions, t),
                messages=messages,
                public_state=public,
                private_info={o.agent_id: o.private_info for o in observations if o.private_info},
                private={a.spec.id: d.private for a, d in zip(agents, decisions) if d.private},
            )
        )
        pending = messages
    return ep
