"""LLM-backed and scripted agents for the generic runner."""

from __future__ import annotations

import asyncio
import json
import zlib
from collections.abc import Callable
from typing import Any

import numpy as np

from collusion.llm import LLM, parse_json
from collusion.runner import Agent, Decision, Game, Observation
from collusion.trajectory import AgentSpec


class ScriptedAgent(Agent):
    """Policy is a plain function (obs, rng) -> action; used for free validation runs."""

    def __init__(self, spec: AgentSpec, policy: Callable[[Observation, np.random.Generator], Any],
                 seed: int = 0):
        self.spec = spec
        self.policy = policy
        self.rng = np.random.default_rng(seed)

    async def act(self, game: Game, obs: Observation) -> Decision:
        return Decision(action=self.policy(obs, self.rng))


class LLMAgent(Agent):
    def __init__(
        self,
        spec: AgentSpec,
        llm: LLM,
        *,
        role_prompt: str = "",
        temperature: float = 1.0,
        history_window: int = 20,
        episode_seed: int = 0,
    ):
        self.spec = spec
        self.llm = llm
        self.role_prompt = role_prompt
        self.temperature = temperature
        self.history_window = history_window
        self.episode_seed = episode_seed
        self.n_parse_failures = 0

    def build_messages(self, game: Game, obs: Observation) -> list[dict]:
        system = game.rules_text(obs.agent_id)
        if self.role_prompt:
            system += "\n\n" + self.role_prompt
        fields = {"reasoning": "one or two short sentences", "action": f"one of {obs.action_space}"}
        if obs.channel_partners:
            fields["message"] = (
                f"private message to {', '.join(obs.channel_partners)} (empty string for none)"
            )
        parts = [
            f"You are {obs.agent_id}. Round {obs.t + 1} of {obs.n_rounds}.",
            f"Your private information this round: {json.dumps(obs.private_info)}"
            if obs.private_info else "",
            "History:\n" + game.history_text(obs.agent_id, obs.history, self.history_window),
            "Private messages you received:\n"
            + "\n".join(f"{m.sender}: {m.text}" for m in obs.inbox)
            if obs.inbox else "",
            "Reply with a JSON object with keys " + json.dumps(fields),
        ]
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(p for p in parts if p)},
        ]

    async def act(self, game: Game, obs: Observation) -> Decision:
        messages = self.build_messages(game, obs)
        # Deterministic across processes (built-in hash() of str is salted per process).
        sample = zlib.crc32(f"{self.episode_seed}:{self.spec.id}:{obs.t}".encode()) * 10
        for attempt in range(3):
            res = await self.llm.chat(
                self.spec.model, messages, temperature=self.temperature,
                sample=sample + attempt, json_mode=True,
            )
            try:
                out = parse_json(res.text)
                action = _coerce(out.get("action"), obs.action_space)
                return Decision(
                    action=action,
                    message=(out.get("message") or None) if obs.channel_partners else None,
                    private=str(out.get("reasoning", "")),
                )
            except (ValueError, KeyError):
                self.n_parse_failures += 1
        # Fall back to the first legal action; counted so experiments can report parse failures.
        return Decision(action=obs.action_space[0], private="PARSE_FAILURE")

    async def resample(self, game: Game, obs: Observation, k: int) -> list[Any]:
        """k fresh draws of this agent's action on `obs` (for conditional randomization tests).
        Sample indices are offset so they never collide with the episode's own call."""
        messages = self.build_messages(game, obs)
        base = zlib.crc32(f"crt:{self.episode_seed}:{self.spec.id}:{obs.t}".encode()) * 1000

        async def one(s: int) -> Any:
            res = await self.llm.chat(self.spec.model, messages, temperature=self.temperature,
                                      sample=base + s, json_mode=True)
            try:
                return _coerce(parse_json(res.text).get("action"), obs.action_space)
            except (ValueError, KeyError):
                return None

        draws = await asyncio.gather(*(one(s) for s in range(k)))
        return [d for d in draws if d is not None]


def _coerce(value: Any, space: list[Any]) -> Any:
    if value in space:
        return value
    for s in space:
        if str(s).strip().lower() == str(value).strip().lower():
            return s
    raise ValueError(f"illegal action {value!r}, legal: {space}")
