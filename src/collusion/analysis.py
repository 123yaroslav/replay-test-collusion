"""Episode-level evaluation helpers: rates with intervals, the pre-registered binomial tests, ROC."""

from __future__ import annotations

import numpy as np
from scipy import stats


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, float(c - h)), min(1.0, float(c + h)))


def p_fpr_below(k: int, n: int, bound: float) -> float:
    """One-sided exact binomial p-value for H0: FPR >= bound (small = FPR is below bound)."""
    return float(stats.binom.cdf(k, n, bound))


def p_fpr_above(k: int, n: int, alpha: float) -> float:
    """One-sided exact binomial p-value for H0: FPR <= alpha (small = detector over-fires)."""
    return float(stats.binom.sf(k - 1, n, alpha)) if k > 0 else 1.0


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a positive scores above a negative (ties count 1/2)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0,
                 stat=np.mean) -> tuple[float, float]:
    """Percentile CI over episodes (the unit of analysis)."""
    values = np.asarray(values, float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = [stat(rng.choice(values, len(values))) for _ in range(n_boot)]
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
