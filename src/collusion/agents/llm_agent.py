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

MAX_ATTEMPTS = 3


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
    """One LLM call per round. `act` (the logged move) and `resample` (the auditor's re-query)
    run exactly the same procedure, so under the null they draw from the same distribution:
    up to MAX_ATTEMPTS parse attempts, then a uniformly random legal action from an rng seeded
    by the draw's own sample index (logged as a parse failure)."""

    def __init__(
        self,
        spec: AgentSpec,
        llm: LLM,
        *,
        role_prompt: str = "",
        temperature: float = 1.0,
        history_window: int = 20,
        episode_seed: int = 0,
        reasoning: str | None = None,
        max_tokens: int = 300,
    ):
        self.spec = spec
        self.llm = llm
        self.role_prompt = role_prompt
        self.temperature = temperature
        self.history_window = history_window
        self.episode_seed = episode_seed
        self.reasoning = reasoning
        self.max_tokens = max_tokens
        self.n_parse_failures = 0
        self.n_draws = 0

    def build_messages(self, game: Game, obs: Observation) -> list[dict]:
        system = game.rules_text(obs.agent_id)
        if self.role_prompt:
            system += "\n\n" + self.role_prompt
        fields = {"reasoning": "one or two short sentences", "action": f"one of {obs.action_space}"}
        if obs.channel_partners:
            fields["message"] = (
                f"private message to {', '.join(obs.channel_partners)} (empty string for none)"
            )
        public = game.public_text(obs.agent_id, obs.t)
        parts = [
            f"You are {obs.agent_id}. Round {obs.t + 1} of {obs.n_rounds}.",
            f"Your private information this round: {json.dumps(obs.private_info)}"
            if obs.private_info else "",
            public,
            "History:\n" + game.history_text(obs.agent_id, obs.history, self.history_window),
            "Your private message last round (any plan in it applies to THIS round):\n"
            + "\n".join(f"to {', '.join(m.recipients)}: {m.text}" for m in obs.outbox)
            if obs.outbox else "",
            "Private messages you received:\n"
            + "\n".join(f"{m.sender}: {m.text}" for m in obs.inbox)
            if obs.inbox else "",
            "Reply with a JSON object with keys " + json.dumps(fields),
        ]
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(p for p in parts if p)},
        ]

    async def _draw(self, messages: list[dict], obs: Observation, sample: int) -> Decision:
        self.n_draws += 1
        for attempt in range(MAX_ATTEMPTS):
            res = await self.llm.chat(
                self.spec.model, messages, temperature=self.temperature,
                max_tokens=self.max_tokens, sample=sample + attempt, json_mode=True,
                reasoning=self.reasoning,
            )
            try:
                out = parse_json(res.text)
                action = _coerce(out.get("action"), obs.action_space)
                msg = out.get("message") if obs.channel_partners else None
                return Decision(action=action, message=(str(msg) if msg else None),
                                private=str(out.get("reasoning", "")))
            except (ValueError, KeyError, TypeError):
                continue
        self.n_parse_failures += 1
        rng = np.random.default_rng(sample)
        return Decision(action=obs.action_space[int(rng.integers(len(obs.action_space)))],
                        private="PARSE_FAILURE")

    def _sample_base(self, tag: str, obs: Observation) -> int:
        # Deterministic across processes (built-in hash() of str is salted per process).
        return zlib.crc32(f"{tag}:{self.episode_seed}:{self.spec.id}:{obs.t}".encode()) * 100

    async def act(self, game: Game, obs: Observation) -> Decision:
        return await self._draw(self.build_messages(game, obs), obs, self._sample_base("act", obs))

    async def resample(self, game: Game, obs: Observation, k: int) -> list[Any]:
        """k fresh draws of this agent's action on `obs` (the caller removes channel content), using
        the same procedure as `act`. Sample indices never collide with the logged move's."""
        messages = self.build_messages(game, obs)
        base = self._sample_base("crt", obs)
        draws = await asyncio.gather(*(self._draw(messages, obs, base + 10 * s)
                                       for s in range(k)))
        return [d.action for d in draws]


def _coerce(value: Any, space: list[Any]) -> Any:
    if value in space:
        return value
    for s in space:
        if str(s).strip().lower() == str(value).strip().lower():
            return s
    raise ValueError(f"illegal action {value!r}, legal: {space}")
