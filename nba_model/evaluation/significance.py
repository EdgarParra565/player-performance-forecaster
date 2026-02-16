from math import sqrt
from typing import Dict

from scipy.stats import norm


def breakeven_probability(american_odds: int = -110) -> float:
    """Compute breakeven win probability for fixed-odds betting."""
    if american_odds == 0:
        raise ValueError("american_odds cannot be 0")
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    return abs(american_odds) / (abs(american_odds) + 100.0)


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Dict[str, float]:
    """
    Wilson score interval for a Bernoulli proportion.

    Returns a dictionary with lower/upper bounds in [0, 1].
    """
    if trials <= 0:
        return {"lower": 0.0, "upper": 0.0}
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    p_hat = successes / trials
    z = norm.ppf(1 - (1 - confidence) / 2)
    z2 = z * z
    denom = 1 + z2 / trials
    center = (p_hat + z2 / (2 * trials)) / denom
    margin = (z / denom) * sqrt((p_hat * (1 - p_hat) / trials) + (z2 / (4 * trials * trials)))
    return {"lower": max(0.0, center - margin), "upper": min(1.0, center + margin)}


def z_test_proportion(successes: int, trials: int, p0: float) -> Dict[str, float]:
    """
    Two-sided one-sample z-test for proportion.

    Tests H0: p = p0 against H1: p != p0.
    """
    if trials <= 0:
        return {"z_score": 0.0, "p_value": 1.0}
    if not 0 < p0 < 1:
        raise ValueError("p0 must be between 0 and 1")

    p_hat = successes / trials
    standard_error = sqrt(p0 * (1 - p0) / trials)
    if standard_error == 0:
        return {"z_score": 0.0, "p_value": 1.0}

    z_score = (p_hat - p0) / standard_error
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    return {"z_score": float(z_score), "p_value": float(p_value)}


def win_rate_significance_summary(
    wins: int,
    bets: int,
    confidence: float = 0.95,
    breakeven_p: float = None,
    american_odds: int = -110,
) -> Dict[str, float]:
    """Aggregate CI and z-test summary for backtest win rate."""
    if breakeven_p is None:
        breakeven_p = breakeven_probability(american_odds)

    interval = wilson_interval(wins, bets, confidence=confidence)
    z_test = z_test_proportion(wins, bets, p0=breakeven_p)

    return {
        "win_rate_ci_lower": interval["lower"],
        "win_rate_ci_upper": interval["upper"],
        "breakeven_prob": float(breakeven_p),
        "z_score_vs_breakeven": z_test["z_score"],
        "p_value_vs_breakeven": z_test["p_value"],
        "significant_at_5pct": float(z_test["p_value"]) < 0.05,
    }
