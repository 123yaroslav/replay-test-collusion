"""Re-implemented double auction: matching, order replacement, bins, and the replay prompt rules."""

import numpy as np

from collusion.games.cda import AgentState, Market, ask_bin, seller_prompt


def _market():
    m = Market(["S1", "S2"], ["B1", "B2"], np.random.default_rng(0))
    m.asks, m.bids = {"S1": 99.0, "S2": 99.0}, {"B1": 80.0, "B2": 80.0}
    return m


def test_matching_midpoint_and_removal():
    m = _market()
    trades = m.apply(1, {"S1": 90.0, "S2": 95.0}, {"B1": 92.0, "B2": 85.0})
    assert trades == [(1, "B1", "S1", 91.0)]
    assert m.asks == {"S2": 95.0} and m.bids == {"B2": 85.0}


def test_replace_withdraw_and_unchanged():
    m = _market()
    m.apply(1, {"S1": "null", "S2": None}, {"B1": 81.5})
    assert "S1" not in m.asks and m.asks["S2"] == 99.0 and m.bids["B1"] == 81.5


def test_bins():
    assert [ask_bin(x) for x in (80, 85, 89.99, 90, 97.5, 100, 120)] == \
        ["lt85", "85-90", "85-90", "90-95", "95-100", "ge100", "ge100"]
    assert ask_bin("null") == "null" and ask_bin(None) == "none" and ask_bin("abc") == "invalid"


def test_replay_prompt_rules():
    m = _market()
    v = m.view(3)
    st = AgentState(memory={1: "asked 99", 2: "asked 97"}, notebook="hold high")
    off_real = seller_prompt("S1", "Acme", 80, 15, v, st, chat=False, inbox={"S2": "keep 99"})
    off_replay = seller_prompt("S1", "Acme", 80, 15, v, st, chat=False, inbox={})
    assert off_real == off_replay  # chat off: byte-identical, the exact null
    on_real = seller_prompt("S1", "Acme", 80, 15, v, st, chat=True, inbox={"S2": "keep 99"})
    on_replay = seller_prompt("S1", "Acme", 80, 15, v, st, chat=True, inbox={})
    assert len(on_real.splitlines()) == len(on_replay.splitlines())
    diff = [(a, b) for a, b in zip(on_real.splitlines(), on_replay.splitlines()) if a != b]
    assert diff == [("- S2: keep 99", "- No messages from other sellers last hour.")]


def test_view_is_frozen():
    m = _market()
    v = m.view(1)
    m.apply(1, {"S1": 70.0}, {"B1": 90.0})
    assert v.asks == [(99.0, "S1"), (99.0, "S2")]  # the view does not change after orders apply


def test_history_shows_placed_prices_and_opening_orders():
    m = _market()
    m.opening = ({"S1": 97.0, "S2": 98.0}, {"B1": 82.0, "B2": 83.0})
    m.apply(1, {"S1": 96.0, "S2": None}, {"B1": "null"})
    h = m.history[0]
    assert h["asks"] == {"S1": 96.0, "S2": 98.0}  # S2 kept its opening order
    assert "B1" in h["bids"] and h["bids"]["B1"] == 82.0  # withdrawals are not shown


def test_trade_list_is_truncated_like_reference():
    from collusion.games.cda import View, _trades
    trades = [(h, "B1", "S1", 90.0) for h in range(60)]
    lines = _trades(View(1, [], [], [], trades)).splitlines()
    assert lines.count("- ...") == 2 and len(lines) == 16 * 3 + 2
