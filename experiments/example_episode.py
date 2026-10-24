"""One real episode for the report: a vendor_leak auction episode with its costs, bids, the vendor's
private designation and the replay test's p-value. Picks the flagged episode with the median p.

    uv run python experiments/example_episode.py
Writes results/summary/example_episode.md (a markdown table the report includes) and
results/summary/example_episode.json (seed, p, top pair).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from collusion.trajectory import read_episodes

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results" / "summary"
SEATS = ["P1", "P2", "P3", "P4"]


def main(n_rounds: int = 4):
    src = SUM / "s2_audits.csv"
    if not src.exists():
        src = SUM / "s2_partial_audits.csv"
    aud = pd.read_csv(src)
    a = aud[(aud.tag == "deepseek") & (aud.condition == "vendor_leak") & (aud.detector == "crt_net")]
    flagged = a[a.correct_pair].sort_values("p")
    row = flagged.iloc[len(flagged) // 2]
    eps = {e.seed: e for e in read_episodes(ROOT / "results/raw/s2/deepseek/vendor_leak.jsonl")}
    ep = eps[int(row.seed)]
    costs = ep.meta["costs"]
    lines = ["| Round | Costs P1 / P2 / P3 / P4 | Vendor designates | Bids P1 / P2 / P3 / P4 | Lowest bid |",
             "|---|---|---|---|---|"]
    for r in ep.rounds[:n_rounds]:
        msg = next((m.text for m in r.messages if m.sender == "VENDOR" and m.recipients == ["P1"]), "")
        designated = "P1" if "You are designated" in msg else "P3"
        bids = [r.actions[s] for s in SEATS]
        low = min(bids)
        winner = f"{low} ({', '.join(s for s in SEATS if r.actions[s] == low)})"
        c = " / ".join(str(costs[s][r.t]) for s in SEATS)
        b = " / ".join(f"**{x}**" if s in ("P1", "P3") else str(x) for s, x in zip(SEATS, bids))
        lines.append(f"| {r.t + 1} | {c} | {designated} | {b} | {winner} |")
    (SUM / "example_episode.md").write_text("\n".join(lines))
    info = {"seed": int(row.seed), "p": float(row.p), "top": row.top, "rounds": len(ep.rounds),
            "shown": n_rounds}
    (SUM / "example_episode.json").write_text(json.dumps(info))
    print("\n".join(lines))
    print(info)


if __name__ == "__main__":
    main()
