import numpy as np

# -----------------------------------------------------------------------------
# Production defaults by stat type (from distribution_sweep review).
# Update after running:
#   python3 -m nba_model.evaluation.run_distribution_sweep \
#     --windows 5 7 10 15 --stat-types points assists rebounds pra \
#     --start-date 2024-11-01 --end-date 2025-03-15
# and choosing best distribution per stat (e.g. by avg_roi or significance).
# See README "Production defaults (from benchmarks)".
# -----------------------------------------------------------------------------
DEFAULT_DISTRIBUTION_BY_STAT = {
    "points": "normal",
    "assists": "normal",
    "rebounds": "normal",
    "pra": "normal",
}

SUPPORTED_DISTRIBUTIONS = [
    "normal",
    "student_t",
    "binomial",
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
