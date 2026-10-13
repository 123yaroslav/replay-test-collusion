"""Conditional randomization test (CRT) for LLM agents via policy re-querying.

Null hypothesis for agent j: its action at round t depends only on what an auditor can see,
i.e. j's deployed policy (model + system prompt) applied to the *public* context: rules, its
own private game info, the public history. The private inbox is removed. We re-query j's policy
k times on exactly that context and use the empirical distribution as j's reference policy.

Redrawing every action from its reference policy gives null datasets that keep shared model
priors and reactions to the common history (they come from the same model on the same history),
but drop any dependence carried by the private channel. So the test separates "they talked and
it changed what they did" from "same model, same history".
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import numpy as np

from collusion.agents.llm_agent import LLMAgent
from collusion.runner import Game, Observation
from collusion.trajectory import Episode


async def reference_probs(
    episode: Episode,
    game: Game,
    agents: dict[str, LLMAgent],
    seats: list[str],
    action_space: list,
    k: int = 8,
) -> np.ndarray:
    """(T, n_seats, n_actions) empirical reference distributions for one episode."""
    game.reset(seats, np.random.default_rng(episode.seed))

    async def cell(t: int, agent_id: str) -> np.ndarray:
        r = episode.rounds[t]
        obs = Observation(
            agent_id=agent_id, t=t, n_rounds=len(episode.rounds),
            private_info=r.private_info.get(agent_id, {}), history=episode.rounds[:t],
            inbox=[], action_space=action_space, channel_partners=[],
        )
        draws = await agents[agent_id].resample(game, obs, k)
        counts = np.array([draws.count(a) for a in action_space], dtype=float)
        return counts / counts.sum() if counts.sum() else np.full(len(action_space),
                                                                   1 / len(action_space))

    rows = await asyncio.gather(*(
        asyncio.gather(*(cell(t, a) for a in seats)) for t in range(len(episode.rounds))
    ))
    return np.array([np.stack(r) for r in rows])


def resampler(ref: np.ndarray) -> Callable[[np.random.Generator], np.ndarray]:
    """Draw a full null action array (rows = pooled rounds, cols = seats) from reference probs."""
    cdf = ref.cumsum(axis=2)
    n_actions = ref.shape[2]

    def draw(rng: np.random.Generator) -> np.ndarray:
        u = rng.random(ref.shape[:2])[..., None]
        return (u > cdf).sum(axis=2).clip(max=n_actions - 1)

    return draw
