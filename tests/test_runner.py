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


class _Talker:
    """Records every observation it sees; always sends a message to its partner."""

    def __init__(self, sid, partner):
        self.spec = AgentSpec(id=sid, model="scripted:talker")
        self.partner = partner
        self.seen = []
        self.requeried = []

    async def act(self, game, obs):
        from collusion.runner import Decision
        self.seen.append(obs)
        return Decision(action="A", message=f"hi from {self.spec.id}")

    async def resample(self, game, obs, k):
        self.requeried.append(obs)
        return ["B"] * k


def _talk_episode(undelivered):
    a, b = _Talker("p0", "p1"), _Talker("p1", "p0")
    ep = asyncio.run(run_episode(Coordination(n_rounds=3), [a, b], condition="t", seed=0,
                                 channels={"p0": ["p1"], "p1": ["p0"]}, requery_k=2,
                                 undelivered=undelivered))
    return ep, a, b


def test_messages_delivered_next_round_and_requery_drops_only_inbox():
    ep, a, _ = _talk_episode(undelivered=None)
    assert [len(o.inbox) for o in a.seen] == [0, 1, 1]
    assert [len(o.outbox) for o in a.seen] == [0, 1, 1]  # own last message is remembered
    for o, r in zip(a.seen, a.requeried):
        assert r.inbox == [] and r.outbox == [] and r.channel_partners == o.channel_partners
        assert r.history == o.history and r.private_info == o.private_info and r.t == o.t
    assert all(r.requery == {"p0": ["B", "B"], "p1": ["B", "B"]} for r in ep.rounds)


def test_undelivered_messages_are_logged_but_never_seen():
    ep, a, b = _talk_episode(undelivered={"p0", "p1"})
    assert all(len(r.messages) == 2 for r in ep.rounds)
    assert all(o.inbox == [] and o.outbox == [] for o in a.seen + b.seen)


def test_detector_view_hides_condition_and_meta():
    ep, _, _ = _talk_episode(undelivered=None)
    ep = ep.model_copy(update={"colluders": ["p0", "p1"], "meta": {"models": {"p0": "x"}}})
    blind = ep.for_detector()
    assert blind.colluders == [] and blind.condition == "hidden" and blind.meta == {}
