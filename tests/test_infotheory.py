import numpy as np

from collusion.detect.infotheory import (
    cross_episode_pairs,
    entropy,
    mutual_info,
    surrogate_pvalue,
    transfer_entropy,
)


def test_entropy_uniform_and_constant():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, 20000)
    assert abs(entropy(x, miller_madow=False) - 2.0) < 0.01
    assert entropy(np.zeros(100), miller_madow=False) == 0.0


def test_mutual_info_identical_vs_independent():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 3, 5000)
    y = rng.integers(0, 3, 5000)
    assert abs(mutual_info(x, x) - np.log2(3)) < 0.02
    assert mutual_info(x, y) < 0.01


def test_conditional_mi_explains_away_common_cause():
    rng = np.random.default_rng(2)
    z = rng.integers(0, 3, 5000)
    noise = rng.random((2, 5000)) < 0.1
    x = np.where(noise[0], rng.integers(0, 3, 5000), z)
    y = np.where(noise[1], rng.integers(0, 3, 5000), z)
    assert mutual_info(x, y) > 0.5  # correlated through the shared signal
    assert mutual_info(x, y, z) < 0.02  # independent given it


def test_transfer_entropy_is_directional():
    rng = np.random.default_rng(3)
    episodes_x, episodes_y = [], []
    for _ in range(20):
        x = rng.integers(0, 3, 200)
        y = np.empty_like(x)
        y[0] = 0
        y[1:] = (x[:-1] + 1) % 3  # y copies x with one step delay
        episodes_x.append(list(x))
        episodes_y.append(list(y))
    assert transfer_entropy(episodes_x, episodes_y) > 1.4
    assert transfer_entropy(episodes_y, episodes_x) < 0.02


def test_cross_episode_surrogate_rejects_only_real_coupling():
    rng = np.random.default_rng(4)
    xs = [list(rng.integers(0, 3, 50)) for _ in range(30)]
    ys = [[0] + [(v + 1) % 3 for v in x[:-1]] for x in xs]

    def te_surrogate(r):
        perm = cross_episode_pairs(len(xs), r)
        return transfer_entropy([xs[i] for i in perm], ys)

    p, _ = surrogate_pvalue(te_surrogate, transfer_entropy(xs, ys), n_surrogates=100)
    assert p < 0.05

    zs = [list(rng.integers(0, 3, 50)) for _ in range(30)]

    def te_surrogate_null(r):
        perm = cross_episode_pairs(len(xs), r)
        return transfer_entropy([xs[i] for i in perm], zs)

    p_null, _ = surrogate_pvalue(te_surrogate_null, transfer_entropy(xs, zs), n_surrogates=100)
    assert p_null > 0.05
