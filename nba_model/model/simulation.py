"""Simulation utilities for NBA prop outcome distributions and calibration."""

from typing import Optional

import numpy as np

# -----------------------------------------------------------------------------
# Production defaults by stat type (from distribution_sweep review).
#
# 2026-06-04 sweep summary
#   command: python3 -m nba_model.evaluation.run_distribution_sweep \
#              --windows 5 7 10 15 --stat-types points assists rebounds pra \
#              --start-date 2024-11-01 --end-date 2025-03-15
#   artifacts: nba_model/evaluation/artifacts/distribution_sweep_2026-06-04_*
#
#   Methodology caveat: the sweep's default "line = expected_value" makes
#   symmetric distributions (normal / student_t / uniform) degenerate —
#   prob_over evaluates to ~0.5 at the mean and the 0.55-edge bet filter
#   rejects every wager, producing 0 bets for those families. Asymmetric
#   distributions (poisson, NB, exponential, lognormal, power_law) do produce
#   bets and so dominate the avg_roi ranking by default; this is a property
#   of the backtest harness, not a claim that they fit player props better
#   than normal in production. Market-line coverage in the DB
#   (`betting_lines`) only spans 2026-03-14..2026-03-15, so --use-market-lines
#   over the full sweep window is not yet possible.
#
#   What the data *does* support:
#     - rebounds: poisson gave the only positive avg_roi (+8.78%, n=482
#       bets, p=0.96 vs breakeven — directionally positive, not significant).
#       Picking poisson here also aligns with the chart-overlay vocabulary
#       (poisson is one of the three families surfaced for counts).
#     - assists / points / pra: every distribution that placed bets posted
#       a negative avg_roi. Evidence is too noisy to switch from normal.
#       Keep "normal" for these three stats and revisit once full-season
#       market-line coverage is backfilled.
#
# Update procedure: re-run the sweep, then update this dict + the README
# "Production defaults" section together. See README §"Production defaults
# (from benchmarks)".
# -----------------------------------------------------------------------------
DEFAULT_DISTRIBUTION_BY_STAT = {
    "points": "normal",      # sweep inconclusive (all negative ROI); keep baseline
    "assists": "normal",     # sweep inconclusive; keep baseline
    "rebounds": "poisson",   # only family with positive avg_roi in 2026-06-04 sweep
    "pra": "normal",         # sweep inconclusive; keep baseline
}

SUPPORTED_DISTRIBUTIONS = [
    "normal",
    "student_t",
    "binomial",
    "negative_binomial",
    "poisson",
    "exponential",
    "uniform",
    "lognormal",
    "power_law",
]

_DISTRIBUTION_ALIASES = {
    "normal": "normal",
    "gaussian": "normal",
    "student_t": "student_t",
    "studentt": "student_t",
    "t": "student_t",
    "t_distribution": "student_t",
    "binomial": "binomial",
    "bernoulli_trials": "binomial",
    "negative_binomial": "negative_binomial",
    "negativebinomial": "negative_binomial",
    "neg_binomial": "negative_binomial",
    "nbinom": "negative_binomial",
    "negbin": "negative_binomial",
    "poisson": "poisson",
    "exponential": "exponential",
    "uniform": "uniform",
    "lognormal": "lognormal",
    "log_normal": "lognormal",
    "powerlaw": "power_law",
    "power_law": "power_law",
    "pareto": "power_law",
}


def normalize_distribution_name(distribution: str) -> str:
    """Normalize distribution aliases to canonical names."""
    key = str(distribution or "").strip().lower()
    mapped = _DISTRIBUTION_ALIASES.get(key)
    if not mapped:
        raise ValueError(
            f"Unsupported distribution '{distribution}'. "
            f"Supported: {SUPPORTED_DISTRIBUTIONS}"
        )
    return mapped


def get_default_distribution(stat_type: str) -> str:
    """Return the production default distribution for a stat type.
    Uses DEFAULT_DISTRIBUTION_BY_STAT; falls back to 'normal' for unknown stats.
    """
    key = (str(stat_type or "").strip().lower()) or "points"
    return DEFAULT_DISTRIBUTION_BY_STAT.get(key, "normal")


def _draw_samples(
    mu: float,
    sigma: float,
    n: int,
    distribution: str,
    sample_size: int | None = None,
) -> np.ndarray:
    """Draw random samples from a selected distribution family."""
    rng = np.random.default_rng()
    dist = normalize_distribution_name(distribution)
    mean = float(mu)
    std = max(float(sigma), 1e-6)

    if dist == "normal":
        return rng.normal(mean, std, size=n)

    if dist == "student_t":
        if sample_size is not None and int(sample_size) > 1:
            dof = float(int(sample_size) - 1)
        else:
            dof = 6.0
        dof = max(2.0, dof)
        return mean + std * rng.standard_t(dof, size=n)

    if dist == "poisson":
        lam = max(0.0, mean)
        return rng.poisson(lam=lam, size=n).astype(float)

    if dist == "binomial":
        if mean <= 1e-9:
            return np.zeros(n, dtype=float)
        variance = max(std * std, 1e-6)
        p = float(np.clip(1.0 - (variance / mean), 1e-4, 0.999))
        n_trials = int(np.clip(np.ceil(mean / p), 1, 5000))
        return rng.binomial(n=n_trials, p=p, size=n).astype(float)

    if dist == "negative_binomial":
        # Overdispersed count model: variance > mean.
        # Method-of-moments: with mean mu, variance v, set p = mu / v
        # and r = mu * p / (1 - p). If v <= mu (no overdispersion), fall
        # back to Poisson so the sampler still produces sensible counts.
        if mean <= 1e-9:
            return np.zeros(n, dtype=float)
        variance = max(std * std, 1e-6)
        if variance <= mean * 1.0001:
            return rng.poisson(lam=mean, size=n).astype(float)
        p = float(np.clip(mean / variance, 1e-4, 0.999))
        r = float(mean * p / max(1.0 - p, 1e-6))
        r = float(np.clip(r, 1e-3, 1e6))
        return rng.negative_binomial(n=r, p=p, size=n).astype(float)

    if dist == "exponential":
        scale = std
        shift = mean - scale
        return shift + rng.exponential(scale=scale, size=n)

    if dist == "uniform":
        half_range = np.sqrt(3.0) * std
        low = mean - half_range
        high = mean + half_range
        return rng.uniform(low=low, high=high, size=n)

    if dist == "lognormal":
        positive_mean = max(mean, 1e-3)
        variance = max(std * std, 1e-6)
        phi = np.sqrt(variance + positive_mean * positive_mean)
        log_sigma = np.sqrt(
            max(np.log((phi * phi) / (positive_mean * positive_mean)), 1e-9))
        log_mu = np.log((positive_mean * positive_mean) / phi)
        return rng.lognormal(mean=log_mu, sigma=log_sigma, size=n)

    if dist == "power_law":
        positive_mean = max(mean, 1e-3)
        variance = max(std * std, 1e-6)
        ratio = variance / (positive_mean * positive_mean)
        alpha = 1.0 + np.sqrt(1.0 + (1.0 / max(ratio, 1e-6)))
        alpha = max(float(alpha), 2.05)
        x_m = positive_mean * (alpha - 1.0) / alpha
        return x_m * (1.0 + rng.pareto(alpha, size=n))

    raise ValueError(
        f"Unsupported distribution '{distribution}'. "
        f"Supported: {SUPPORTED_DISTRIBUTIONS}"
    )


def blend_team_prior(
    mu: float,
    sigma: float,
    *,
    pace_factor: Optional[float] = None,
    implied_team_total: Optional[float] = None,
    team_recent_avg_total: Optional[float] = None,
    alpha: float = 0.3,
) -> tuple[float, float]:
    """Adjust a player's (``mu``, ``sigma``) using the team's cross-book prior.

    Two complementary signals can shift the projection:

    1. **Pace** (``pace_factor`` from ``team_priors``): how many more / fewer
       possessions the game is priced for vs the league baseline. Counting
       stats (points / rebounds / assists / etc.) scale roughly linearly
       with possessions, so we scale ``mu`` by
       ``1 + alpha * (pace_factor - 1)``.
    2. **Team total** (``implied_team_total`` from ``team_priors``): how
       many points the *team* is expected to score, relative to the
       player's team's recent baseline. When ``team_recent_avg_total`` is
       supplied we layer that scaling on top.

    ``alpha`` ∈ [0, 1] controls how heavily the prior overrides the player's
    own historical mean. ``alpha=0`` is "no blend" (status quo);
    ``alpha=0.3`` is a sensible default (modest pull toward the market's
    view). ``sigma`` is scaled by the same factor so the relative
    coefficient of variation is preserved.

    All inputs are optional — when none are supplied the returned mu/sigma
    equal the inputs, which makes this safe to call unconditionally from
    the player pipeline.
    """
    factor = 1.0
    if pace_factor is not None:
        try:
            pf = float(pace_factor)
            if pf > 0:
                factor *= 1.0 + alpha * (pf - 1.0)
        except (TypeError, ValueError):
            pass
    if implied_team_total is not None and team_recent_avg_total:
        try:
            tt = float(implied_team_total)
            base = float(team_recent_avg_total)
            if tt > 0 and base > 0:
                ratio = tt / base
                factor *= 1.0 + alpha * (ratio - 1.0)
        except (TypeError, ValueError):
            pass
    # Clamp to a sane range — the prior should *nudge* projections, not
    # produce ±50 % swings even if the inputs are extreme.
    factor = max(0.5, min(factor, 1.5))
    return float(mu) * factor, float(sigma) * factor


def monte_carlo_over(
    mu: float,
    sigma: float,
    line: float,
    n: int = 10000,
    distribution: str = "normal",
    sample_size: int | None = None,
) -> float:
    """
    Monte Carlo estimate of over probability under chosen distribution.
    """
    if int(n) <= 0:
        raise ValueError("n must be > 0")

    sims = _draw_samples(
        mu=mu,
        sigma=sigma,
        n=int(n),
        distribution=distribution,
        sample_size=sample_size,
    )
    return float((sims > float(line)).mean())
