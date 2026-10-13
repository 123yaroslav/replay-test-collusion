import numpy as np

from collusion.detect.net_influence import (
    bonjour_rule,
    calibrated_test,
    gamma,
    influence_matrix,
    net_influence,
)


def rps_follower_game(n_games, cp, rng, n_agents=3, pair=(0, 1)):
    """Bonjour's manual collusion: agent pair[1] plays the move that A beats w.p. CP."""
    a = rng.integers(0, 3, (n_games, n_agents))
    follow = rng.random(n_games) < cp
    a[follow, pair[1]] = (a[follow, pair[0]] + 2) % 3
    return a


def test_gamma_matches_closed_form_for_rps_follower():
    rng = np.random.default_rng(0)
    a = rps_follower_game(200_000, 0.5, rng)
    true = np.log2(3) + (2 / 3) * np.log2(2 / 3) + 2 * (1 / 6) * np.log2(1 / 6)
    assert abs(gamma(a[:, 0], a[:, 1]) - true) < 0.01  # ≈ 0.333 bits (Bonjour Fig. 3)


def test_net_influence_flags_true_pair_with_enough_games():
    rng = np.random.default_rng(1)
    a = rps_follower_game(1000, 0.8, rng, n_agents=4, pair=(1, 3))
    big = net_influence(influence_matrix(a))
    assert bonjour_rule(big) == (1, 3)


def test_partial_observability_detects_private_state_leakage():
    rng = np.random.default_rng(2)
    n = 20_000
    s = rng.integers(0, 2, (n, 3))  # private binary states
    a = rng.integers(0, 2, (n, 3))
    a[:, 1] = s[:, 0]  # agent 1 acts on agent 0's private state
    g = influence_matrix(a, s)
    assert g[0, 1] > 0.9 and g[0, 2] < 0.01 and g[1, 2] < 0.01


def test_calibrated_test_controls_false_alarms_under_null():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 3, (150, 4))
    _, p_global, _ = calibrated_test(a, None, lambda r: r.integers(0, 3, a.shape), n_null=99)
    assert p_global > 0.05
    b = rps_follower_game(150, 1.0, rng, n_agents=4, pair=(0, 2))
    top, p_global, _ = calibrated_test(b, None, lambda r: r.integers(0, 3, b.shape), n_null=99)
    assert top == (0, 2) and p_global <= 0.02


def test_rps_payoffs_match_bonjour():
    from collusion.games.rps import payoffs
    assert payoffs(np.array([0, 0, 0])).tolist() == [0, 0, 0]
    assert payoffs(np.array([0, 1, 2])).tolist() == [1, 1, 1]
    assert payoffs(np.array([0, 1, 1])).tolist() == [0, 1, 1]  # paper beats rock
    assert payoffs(np.array([2, 0, 2])).tolist() == [0, 1, 0]  # rock beats scissors
