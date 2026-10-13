import asyncio

from collusion.agents.llm_agent import ScriptedAgent
from collusion.runner import Game, run_episode
from collusion.trajectory import AgentSpec, Episode


class Coordination(Game):
    """Toy game: reward 1 for matching the majority action."""

    name = "toy"

    def reset(self, agent_ids, rng):
        self.ids = agent_ids

    def action_space(self, agent_id):
        return ["A", "B"]

    def payoffs(self, actions, t):
        n_a = sum(v == "A" for v in actions.values())
        major = "A" if n_a * 2 >= len(actions) else "B"
        return {i: float(v == major) for i, v in actions.items()}

    def rules_text(self, agent_id):
        return "Pick A or B."


def test_run_episode_logs_actions_messages_and_strips_ground_truth():
    agents = [
        ScriptedAgent(AgentSpec(id=f"p{i}", model="scripted:const", role="colluder" if i < 2 else "independent"),
                      lambda obs, rng: "A")
        for i in range(3)
    ]
    ep = asyncio.run(
        run_episode(Coordination(n_rounds=5), agents, condition="test", seed=0,
                    colluders=["p0", "p1"])
    )
    assert isinstance(ep, Episode)
    assert len(ep.rounds) == 5
    assert ep.action_series("p2") == ["A"] * 5
    assert all(r.rewards == {"p0": 1.0, "p1": 1.0, "p2": 1.0} for r in ep.rounds)
    blind = ep.for_detector()
    assert blind.colluders == [] and all(a.role == "unknown" for a in blind.agents)
