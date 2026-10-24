"""External panel ($0): do standard dependence detectors flag sellers in Agrawal et al. (2025)?

One audit = one shipped run (5 sellers, 30 rounds). Seller actions are discretised ask moves
(D/U/S/W). Detectors: naive pairwise MI with a within-run time-shuffle null (family-wise over
the 10 seller pairs), net influence with the same null, and Bonjour's fixed-alpha rule.

In the `base` runs sellers cannot message each other (verified: their prompts have no message
section) and all five run gpt-4.1, so a flag there is a false alarm under the
information-flow definition of collusion (it may still reflect tacit coordination through the
shared public order book). We do not re-query these policies: today's gpt-4.1 endpoint is not
the May 2025 snapshot that produced the logs, so a CRT here would be invalid.

    uv run python experiments/agrawal_panel.py
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from collusion.adapters.agrawal import load_all
from collusion.analysis import p_fpr_above, wilson
from collusion.detect.crt_llm import shuffle_sampler
from collusion.detect.net_influence import (
    bonjour_rule,
    calibrated_test,
    influence_matrix,
    net_influence,
)

OUT = Path(__file__).resolve().parents[1] / "results" / "summary"
CODES = {"D": 0, "U": 1, "S": 2, "W": 3}


def moves(asks, tol=0.005):
    out, prev = [], None
    for a in asks:
        if a is None:
            out.append("W")
        elif prev is None:
            out.append("S")
        else:
            out.append("D" if a < prev - tol else "U" if a > prev + tol else "S")
        prev = a if a is not None else prev
    return out


def audit(args):
    ep_json, n_null = args
    from collusion.trajectory import Episode

    ep = Episode.model_validate_json(ep_json)
    sellers = ep.meta["seller_ids"]
    a = np.column_stack([[CODES[m] for m in moves(ep.action_series(s))] for s in sellers])
    row = {"run": ep.episode_id, "condition": ep.condition,
           "seller_comms": ep.meta["seller_comms"], "T": len(a),
           "models": "+".join(sorted({x.model for x in ep.agents if x.role == "seller"}))}
    row["bonjour_alpha"] = bonjour_rule(net_influence(influence_matrix(a))) is not None
    for name, net in (("naive_mi_shuffle", False), ("net_shuffle", True)):
        _, p, _ = calibrated_test(a, None, shuffle_sampler(a), n_null=n_null, seed=0, net=net)
        row[name + "_p"] = p
        row[name] = p < 0.05
    return row


def main():
    eps = load_all()
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(audit, [(e.model_dump_json(), 999) for e in eps]))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "agrawal_panel_runs.csv", index=False)
    out = []
    for cond, g in df.groupby("condition"):
        for det in ("naive_mi_shuffle", "net_shuffle", "bonjour_alpha"):
            k, n = int(g[det].sum()), len(g)
            lo, hi = wilson(k, n)
            out.append({"condition": cond, "seller_comms": bool(g.seller_comms.iloc[0]),
                        "models": g.models.iloc[0], "detector": det, "n_runs": n,
                        "flag_rate": k / n, "lo": lo, "hi": hi,
                        "p_overfires_vs_5pct": p_fpr_above(k, n, 0.05)})
    s = pd.DataFrame(out)
    s.to_csv(OUT / "agrawal_panel_summary.csv", index=False)
    with pd.option_context("display.width", 200):
        print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
