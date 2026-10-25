"""Sequential replay test with a likelihood-ratio bet (EXPLORATORY, added after the main analysis).

For a pair (i, j), the round's e-value is e_t = q_t(a_it, a_jt) / mean_{x in pool_it, y in pool_jt}
q_t(x, y), as in `crt_llm.sequential_replay_test`, but with the bet
    q_t(x, y) = P_t(x, y) / (Q_it(x) Q_jt(y)),
where P_t is a Krichevsky-Trofimov estimate of the pair's joint law from its logged actions before
round t and Q_it, Q_jt are KT estimates of each agent's replay law from its replays before round t.
Against an alternative in which the logged pair has law P and replays have laws Q_i, Q_j, the bet
P / (Q_i x Q_j) is the likelihood ratio of which pool entry was logged, so it maximises the expected
log-growth among all bets; the joint bet of the pre-registered test (P mixed with uniform) does not
remove the replay marginals and can have negative growth when agents share strong priors.
q_t depends only on rounds before t, so under H0 and A1 E[e_t | past, pools] = 1 and the average
over pairs of the running products is an e-process (Ville's inequality applies unchanged).
"""

from __future__ import annotations

import numpy as np


def sequential_gro_test(pool: np.ndarray, alpha: float = 0.05, n_actions: int | None = None,
                        prior: float = 0.5) -> dict:
    """pool: (T, n_agents, k+1) action codes, logged action at index 0. n_actions must be the
    alphabet size fixed before the episode (pass it explicitly; inferring it from the whole episode
    would look into the future)."""
    t_max, n, m = pool.shape
    if n_actions is None:
        raise ValueError("pass n_actions: the action alphabet must be fixed before round 1")
    K = n_actions
    a = pool[..., 0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rep_counts = np.zeros((n, K))
    log_e = np.zeros((len(pairs), t_max + 1))
    joint = [np.zeros((K, K)) for _ in pairs]
    for t in range(t_max):
        q_marg = (rep_counts + prior) / (rep_counts.sum(1, keepdims=True) + prior * K)
        for p_idx, (i, j) in enumerate(pairs):
            c = joint[p_idx]
            p_hat = (c + prior) / (c.sum() + prior * K * K)
            q = p_hat / np.outer(q_marg[i], q_marg[j])
            num = q[a[t, i], a[t, j]]
            den = q[np.ix_(pool[t, i], pool[t, j])].mean()
            log_e[p_idx, t + 1] = log_e[p_idx, t] + np.log(num / den)
            c[a[t, i], a[t, j]] += 1
        for i in range(n):  # replays of round t become usable from round t + 1
            for x in pool[t, i, 1:]:
                rep_counts[i, x] += 1
    e_avg = np.exp(log_e).mean(axis=0)
    hit = np.flatnonzero(e_avg >= 1 / alpha)
    stop = int(hit[0]) if len(hit) else t_max
    top = pairs[int(np.argmax(log_e[:, stop]))]
    return {"reject_round": int(hit[0]) if len(hit) else None, "top": top,
            "final_e": float(e_avg[-1]), "max_e": float(e_avg.max()), "e_path": e_avg,
            "log_e_pairs": log_e}
