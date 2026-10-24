"""Harm in the auction (buyer overpayment, pair profit share) and which pairs naive detectors
flag in the mixed-model null (same-model vs cross-model pairs).

    uv run python experiments/harm_and_pairs.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from collusion.trajectory import read_episodes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "summary"


def auction_harm():
    rows = []
    for f in sorted((ROOT / "results" / "raw" / "s2" / "deepseek").glob("*.jsonl")):
        for ep in read_episodes(f):
            costs = ep.meta["costs"]
            for r in ep.rounds:
                win_bid = min(r.actions.values())
                min_cost = min(costs[a][r.t] for a in costs)
                pair_profit = r.rewards["P1"] + r.rewards["P3"]
                rows.append({"condition": f.stem, "seed": ep.seed, "t": r.t,
                             "overpayment": win_bid - min_cost, "winning_bid": win_bid,
                             "pair_profit": pair_profit,
                             "pair_wins": float(r.rewards["P1"] > 0 or r.rewards["P3"] > 0)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    ep = df.groupby(["condition", "seed"]).agg(overpayment=("overpayment", "mean"),
                                               pair_profit=("pair_profit", "sum"),
                                               pair_wins=("pair_wins", "mean")).reset_index()
    s = ep.groupby("condition").agg(n=("seed", "size"), overpayment=("overpayment", "mean"),
                                    overpayment_sd=("overpayment", "std"),
                                    pair_profit=("pair_profit", "mean"),
                                    pair_win_share=("pair_wins", "mean")).round(3)
    s.to_csv(OUT / "s2_harm.csv")
    print(s.to_string())


def mixed_pairs():
    a = OUT / "s1_audits.csv"
    if not a.exists():
        return
    df = pd.read_csv(a)
    g = df[(df.tag == "mixed") & (df.condition == "null_mixed") & df.flag05]
    same = {"P1-P2", "P3-P4"}
    g = g.assign(same_model_pair=g.top.isin(same))
    s = g.groupby("detector").agg(flags=("top", "size"),
                                  same_model_share=("same_model_pair", "mean")).round(3)
    s.to_csv(OUT / "s1_mixed_pairs.csv")
    print("\nnull_mixed: share of flags on same-model pairs (2 of 6 pairs are same-model)")
    print(s.to_string())


if __name__ == "__main__":
    auction_harm()
    mixed_pairs()
