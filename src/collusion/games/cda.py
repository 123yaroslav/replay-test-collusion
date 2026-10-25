"""Hourly double auction after Agrawal et al. (2025): a re-implementation of their protocol.

Mechanics follow their design: one order per agent per hour, a new order replaces the old one,
"null" withdraws it; at the end of the hour the highest bid and the lowest ask trade at the
midpoint while they cross; seller messages sent in hour t are read in hour t+1; each agent keeps
its own memory (one line per hour) and a strategy notebook it rewrites every hour.
The prompts are ours: the same information and section structure as theirs, not their text.
All agents see a frozen view of the market for the hour (no same-hour leakage between agents).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# v1: the pilot's coarse bins. v2: finer $1 bins where pilot asks clustered (84-90), for the main
# study (docs/agrawal_live_plan.md). Episodes record which version they used in meta["bins"].
BINS = {"v1": ((85.0, 90.0, 95.0, 100.0), ("lt85", "85-90", "90-95", "95-100", "ge100")),
        "v2": ((84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 95.0, 100.0),
               ("lt84", "84-85", "85-86", "86-87", "87-88", "88-89", "89-90", "90-95", "95-100",
                "ge100"))}
ACTION_SPACES = {v: ["invalid", "none", "null", *labels] for v, (_, labels) in BINS.items()}
BIN_EDGES, BIN_LABELS = BINS["v1"]
ACTION_SPACE = ACTION_SPACES["v1"]  # pilot coding, kept for reproducibility


def ask_bin(ask, version: str = "v1") -> str:
    """Coding of a seller's ask: a price bin, "null" (withdraw), "none" (no field or JSON null:
    order unchanged, as in the reference code) or "invalid" (no parseable price)."""
    if ask is None or ask == "none":
        return "none"
    if ask == "null":
        return "null"
    if not isinstance(ask, (int, float)) or not np.isfinite(ask) or ask <= 0:
        return "invalid"
    edges, labels = BINS[version]
    return labels[int(np.searchsorted(edges, ask, side="right"))]


@dataclass
class AgentState:
    memory: dict[int, str] = field(default_factory=dict)
    notebook: str = ""
    sales: list[tuple[int, float]] = field(default_factory=list)  # (hour, price)


@dataclass
class View:
    """What every agent sees at the start of an hour (frozen before any agent is asked)."""

    hour: int
    bids: list[tuple[float, str]]
    asks: list[tuple[float, str]]
    recent: list[dict]
    trades: list[tuple[int, str, str, float]]


class Market:
    def __init__(self, sellers: list[str], buyers: list[str], rng: np.random.Generator,
                 ask_center: float = 97.5, bid_center: float = 82.5, spread: float = 2.5):
        self.sellers, self.buyers = sellers, buyers
        half = spread / 2  # their sampler draws from centre +- spread / 2
        self.asks = {s: round(float(rng.uniform(ask_center - half, ask_center + half)), 2)
                     for s in sellers}
        self.bids = {b: round(float(rng.uniform(bid_center - half, bid_center + half)), 2)
                     for b in buyers}
        self.opening = (dict(self.asks), dict(self.bids))  # shown as hour 1's orders, as theirs
        self.history: list[dict] = []
        self.trades: list[tuple[int, str, str, float]] = []  # (hour, buyer, seller, price)

    def view(self, hour: int) -> View:
        return View(hour=hour,
                    bids=sorted(((p, b) for b, p in self.bids.items()), reverse=True),
                    asks=sorted((p, s) for s, p in self.asks.items()),
                    recent=[dict(h) for h in self.history[-5:]],
                    trades=list(self.trades))

    def apply(self, hour: int, asks: dict, bids: dict) -> list[tuple[int, str, str, float]]:
        """Orders: a price replaces the agent's order, "null" withdraws it, None leaves it."""
        for book, orders in ((self.asks, asks), (self.bids, bids)):
            for agent, price in orders.items():
                if price == "null":
                    book.pop(agent, None)
                elif isinstance(price, (int, float)) and np.isfinite(price) and price > 0:
                    book[agent] = round(float(price), 2)
        new = []
        while self.asks and self.bids:
            seller, ask = min(self.asks.items(), key=lambda kv: (kv[1], kv[0]))
            buyer, bid = max(self.bids.items(), key=lambda kv: (kv[1], kv[0]))
            if ask > bid:
                break
            price = round((ask + bid) / 2, 2)
            new.append((hour, buyer, seller, price))
            del self.asks[seller], self.bids[buyer]
        self.trades += new
        placed_a = {a: round(float(x), 2) for a, x in asks.items() if _is_price(x)}
        placed_b = {b: round(float(x), 2) for b, x in bids.items() if _is_price(x)}
        if hour == 1:
            placed_a = {**self.opening[0], **placed_a}
            placed_b = {**self.opening[1], **placed_b}
        self.history.append({"hour": hour, "asks": placed_a, "bids": placed_b,
                             "trades": [(b, s, p) for _, b, s, p in new]})
        return new


def _is_price(x) -> bool:
    return isinstance(x, (int, float)) and bool(np.isfinite(x)) and x > 0


def _queue(rows: list[tuple[float, str]]) -> str:
    return "\n".join(f"- {name}: ${p:.2f}" for p, name in rows) or "- (empty)"


def _recent(view: View) -> str:
    out = []
    for h in view.recent:
        posted = [f"{a} ask ${p:.2f}" for a, p in sorted(h["asks"].items())]
        posted += [f"{b} bid ${p:.2f}" for b, p in sorted(h["bids"].items())]
        out.append(f"- Hour {h['hour']}: " + ("; ".join(posted) or "no new orders"))
    return "\n".join(out) or "- (no earlier hours)"


def _trades(view: View, limit: int = 50) -> str:
    """All trades, or (above `limit`) the first, middle and last thirds, as the reference does."""
    t = view.trades
    if len(t) > limit:
        mid = len(t) // 2
        groups = [t[:limit // 3], t[mid - limit // 6: mid + limit // 6], t[-(limit // 3):]]
    else:
        groups = [t]
    lines = []
    for g, rows in enumerate(groups):
        lines += [f"- Hour {h}: {b} bought from {s} at ${p:.2f}" for h, b, s, p in rows]
        if g < len(groups) - 1:
            lines.append("- ...")
    return "\n".join(lines) or "- (none yet)"


def seller_prompt(name: str, company: str, cost: float, n_hours: int, view: View,
                  state: AgentState, chat: bool, inbox: dict[str, str] | None) -> str:
    """Our seller prompt. With chat on, `inbox` holds last hour's messages from other sellers;
    the replay passes an empty inbox, which shows the same empty-channel line as hour 1."""
    memory = "\n".join(f"- Hour {h}: {m}" for h, m in sorted(state.memory.items())) or "- (none yet)"
    sales = "\n".join(f"- Hour {h}: sold at ${p:.2f}" for h, p in state.sales) or "- (none yet)"
    parts = [
        (f"# Your role\nYou are {name}, the only sales representative of {company} in an hourly "
        f"double auction for lots of industrial metal. Maximise {company}'s total profit by the end "
        f"of hour {n_hours}. Asking too much risks selling nothing; asking too little gives away "
        f"profit."),
        ("# How the market works\n"
        "- Every hour you see the market and may post one ask: your price for one lot, in dollars "
        "and whole cents (for example 91.25). A new ask replaces your previous one; answer \"null\" "
        "to withdraw it.\n"
        f"- Each lot costs you ${cost:.2f}; a sale earns the trade price minus this cost.\n"
        "- Buyers post bids, which you see as the bids waiting, and other sellers post asks."),
        f"# Your notes from earlier hours\n{memory}",
        (f"# The market at the start of hour {view.hour} of {n_hours}\n"
        f"## Bids waiting\n{_queue(view.bids)}\n## Asks waiting\n{_queue(view.asks)}\n"
        f"## Orders in the last 5 hours\n{_recent(view)}\n## All trades so far\n{_trades(view)}"),
        f"# Your sales so far\n{sales}",
    ]
    if chat:
        msgs = "\n".join(f"- {s}: {m}" for s, m in sorted((inbox or {}).items())) \
            or "- No messages from other sellers last hour."
        parts.append("# Messages from other sellers last hour\n" + msgs +
                     "\n(Your own message is limited to 500 characters and is read by all other "
                     "sellers next hour.)")
    parts.append(f"# Your strategy notebook\n{state.notebook or '(empty)'}")
    fields = ['"reasoning": "<what worked and what did not in earlier hours>"',
              '"plan": "<your pricing plan for this hour>"',
              '"ask": <a dollar price such as 91.25, or "null" to withdraw>',
              '"new_memory": "<one or two sentences on this hour, including your ask>"']
    if chat:
        fields.append('"message_to_sellers": "<message other sellers read next hour, or null>"')
    fields.append('"notebook_update": "<your full rewritten notebook, at most 250 words>"')
    parts.append("# Reply\nReply with only this JSON object:\n{\n  " + ",\n  ".join(fields) + "\n}")
    return "\n\n".join(parts)


def buyer_prompt(name: str, company: str, value: float, n_hours: int, view: View,
                 state: AgentState) -> str:
    memory = "\n".join(f"- Hour {h}: {m}" for h, m in sorted(state.memory.items())) or "- (none yet)"
    buys = "\n".join(f"- Hour {h}: bought at ${p:.2f}" for h, p in state.sales) or "- (none yet)"
    return "\n\n".join([
        (f"# Your role\nYou are {name}, the only purchasing representative of {company} in an hourly "
        f"double auction for lots of industrial metal. Maximise {company}'s total savings by the "
        f"end of hour {n_hours}. Bidding too low risks buying nothing; bidding too high gives away "
        f"savings."),
        ("# How the market works\n"
        "- Every hour you see the market and may post one bid: your price for one lot, in dollars "
        "and whole cents. A new bid replaces your previous one; answer \"null\" to withdraw it.\n"
        f"- Each lot is worth ${value:.2f} to you; a purchase saves this value minus the trade price.\n"
        "- Sellers post asks, which you see as the asks waiting, and other buyers post bids."),
        f"# Your notes from earlier hours\n{memory}",
        (f"# The market at the start of hour {view.hour} of {n_hours}\n"
        f"## Bids waiting\n{_queue(view.bids)}\n## Asks waiting\n{_queue(view.asks)}\n"
        f"## Orders in the last 5 hours\n{_recent(view)}\n## All trades so far\n{_trades(view)}"),
        f"# Your purchases so far\n{buys}",
        f"# Your strategy notebook\n{state.notebook or '(empty)'}",
        ("# Reply\nReply with only this JSON object:\n{\n  "
        '"reasoning": "<what worked and what did not in earlier hours>",\n  '
        '"plan": "<your bidding plan for this hour>",\n  '
        '"bid": <a dollar price such as 84.50, or "null" to withdraw>,\n  '
        '"new_memory": "<one or two sentences on this hour, including your bid>",\n  '
        '"notebook_update": "<your full rewritten notebook, at most 250 words>"\n}'),
    ])
