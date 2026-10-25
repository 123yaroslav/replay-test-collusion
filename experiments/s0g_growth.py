"""S0g ($0, EXPLORATORY, after the main analysis): exact growth rates of the sequential test's bets.

One pair, three actions, k replays per agent. Under the alternative the logged pair has joint law P
and each replay is an independent draw from the agent's no-channel law Q (here the same Q for both
agents). For a betting function q the round's e-value is
    e = q(a_i, a_j) / mean_{x in pool_i, y in pool_j} q(x, y),
and its growth rate is G = E_P[log e]. Bets compared (all with the true laws, to isolate the form):
    joint        q = 0.5 P + 0.5 U          (the pre-registered form, with a perfect estimate of P)
    interaction  q = P / (P_i x P_j)        (P's own marginals)
    ratio        q = P / (Q_i x Q_j)        (likelihood ratio against the replay laws)
The ratio bet is the likelihood ratio of "which pool entry was logged", so it maximises the growth
rate among these bets; its growth is at most KL(P || Q_i x Q_j) and approaches it as k grows.

Channels: `copy` (j copies i's move with probability eps; marginals unchanged) and `rare` (with
probability eps both play the least likely move; marginals shift). Priors: uniform and skewed.

    uv run python experiments/s0g_growth.py
Writes results/summary/s0g_growth.csv and prints a check against Monte Carlo.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results" / "summary"
PRIORS = {"uniform": np.array([1 / 3, 1 / 3, 1 / 3]), "skewed": np.array([0.6, 0.3, 0.1])}


def joint_law(q, channel, eps):
    a = len(q)
    base = np.outer(q, q)
    if channel == "copy":
        return (1 - eps) * base + eps * np.diag(q)
    rare = np.zeros((a, a))
    rare[np.argmin(q), np.argmin(q)] = 1.0
    return (1 - eps) * base + eps * rare


def bets(p, q):
    a = len(q)
    u = np.full((a, a), 1 / a**2)
    pi, pj = p.sum(1), p.sum(0)
    return {"joint": 0.5 * p + 0.5 * u,
            "interaction": p / np.outer(pi, pj),
            "ratio": p / np.outer(q, q)}


def growth_exact(p, q, bet, k):
    """E[log e] by enumerating the logged pair and all k replays of each agent."""
    a = len(q)
    total = 0.0
    reps = list(itertools.product(range(a), repeat=k))
    wq = {r: float(np.prod(q[list(r)])) for r in reps}
    for x0, y0 in itertools.product(range(a), range(a)):
        if p[x0, y0] == 0:
            continue
        for rx in reps:
            px = [x0, *rx]
            for ry in reps:
                py = [y0, *ry]
                den = bet[np.ix_(px, py)].mean()
                total += p[x0, y0] * wq[rx] * wq[ry] * np.log(bet[x0, y0] / den)
    return total


def growth_mc(p, q, bet, k, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    a = len(q)
    flat = rng.choice(a * a, size=n, p=p.ravel())
    x0, y0 = flat // a, flat % a
    rx = rng.choice(a, size=(n, k), p=q)
    ry = rng.choice(a, size=(n, k), p=q)
    px = np.concatenate([x0[:, None], rx], 1)
    py = np.concatenate([y0[:, None], ry], 1)
    den = bet[px[:, :, None], py[:, None, :]].mean(axis=(1, 2))
    return float(np.mean(np.log(bet[x0, y0] / den)))


def kl(p, q):
    r = np.outer(q, q)
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / r[m])))


def main():
    rows = []
    for prior, q in PRIORS.items():
        for channel in ("copy", "rare"):
            for eps in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
                p = joint_law(q, channel, eps)
                b = bets(p, q)
                row = {"prior": prior, "channel": channel, "eps": eps, "kl": kl(p, q)}
                for k in (1, 2, 4):
                    for name, bet in b.items():
                        row[f"{name}_k{k}"] = growth_exact(p, q, bet, k) if k <= 2 else \
                            growth_mc(p, q, bet, k)
                rows.append(row)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "s0g_growth.csv", index=False)
    with pd.option_context("display.width", 220):
        cols = ["prior", "channel", "eps", "kl", "joint_k2", "interaction_k2", "ratio_k2",
                "ratio_k1", "ratio_k4"]
        print(df[cols].round(4).to_string(index=False))
    # Monte Carlo check of the exact enumeration at one point
    q = PRIORS["skewed"]
    p = joint_law(q, "copy", 0.3)
    for name, bet in bets(p, q).items():
        print(f"check {name}: exact {growth_exact(p, q, bet, 2):.4f}  MC {growth_mc(p, q, bet, 2):.4f}")


if __name__ == "__main__":
    main()
