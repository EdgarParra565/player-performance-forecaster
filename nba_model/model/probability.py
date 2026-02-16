import numpy as np
from scipy.stats import binom, expon, lognorm, norm, pareto, poisson, t, uniform


def prob_over(line: float, mu: float, sigma: float) -> float:
    """
    Closed-form probability of going over the line
    """
    if sigma <= 0:
        return 0.0

    return 1 - norm.cdf(line, mu, sigma)


def prob_over_distribution(
    line: float,
    mu: float,
    sigma: float,
    distribution: str = "normal",
    sample_size: int | None = None,
) -> float:
    """
    Closed-form over probability under several distribution assumptions.

    Supported distributions:
      - normal
      - student_t
      - binomial
      - poisson
      - exponential
      - uniform
      - lognormal
      - power_law (Pareto approximation)
    """
    dist = str(distribution or "normal").strip().lower()
    x = float(line)
    mean = float(mu)
    std = max(float(sigma), 1e-6)

    if dist in {"normal", "gaussian"}:
        return float(1.0 - norm.cdf(x, loc=mean, scale=std))

    if dist in {"student_t", "studentt", "t", "t_distribution"}:
        if sample_size is not None and int(sample_size) > 1:
            dof = float(int(sample_size) - 1)
        else:
            dof = 6.0
        dof = max(2.0, dof)
        return float(1.0 - t.cdf(x, df=dof, loc=mean, scale=std))

    if dist == "poisson":
        lam = max(0.0, mean)
        k = int(np.floor(x))
        return float(1.0 - poisson.cdf(k, mu=lam))

    if dist in {"binomial", "bernoulli_trials"}:
        if mean <= 1e-9:
            return float(1.0 if x < 0 else 0.0)
        variance = max(std * std, 1e-6)
        p = float(np.clip(1.0 - (variance / mean), 1e-4, 0.999))
        n_trials = int(np.clip(np.ceil(mean / p), 1, 5000))
        k = int(np.floor(x))
        return float(1.0 - binom.cdf(k, n=n_trials, p=p))

    if dist == "exponential":
        scale = std
        shift = mean - scale
        if x < shift:
            return 1.0
        return float(1.0 - expon.cdf(x, loc=shift, scale=scale))

    if dist == "uniform":
        half_range = np.sqrt(3.0) * std
        low = mean - half_range
        high = mean + half_range
        if high <= low:
            return float(1.0 if mean > x else 0.0)
        return float(1.0 - uniform.cdf(x, loc=low, scale=(high - low)))

    if dist in {"lognormal", "log_normal"}:
        positive_mean = max(mean, 1e-3)
        variance = max(std * std, 1e-6)
        phi = np.sqrt(variance + positive_mean * positive_mean)
        log_sigma = np.sqrt(max(np.log((phi * phi) / (positive_mean * positive_mean)), 1e-9))
        log_mu = np.log((positive_mean * positive_mean) / phi)
        if x <= 0:
            return 1.0
        return float(1.0 - lognorm.cdf(x, s=log_sigma, scale=np.exp(log_mu)))

    if dist in {"power_law", "powerlaw", "pareto"}:
        positive_mean = max(mean, 1e-3)
        variance = max(std * std, 1e-6)
        ratio = variance / (positive_mean * positive_mean)
        alpha = 1.0 + np.sqrt(1.0 + (1.0 / max(ratio, 1e-6)))
        alpha = max(float(alpha), 2.05)
        x_m = positive_mean * (alpha - 1.0) / alpha
        if x <= x_m:
            return 1.0
        return float(1.0 - pareto.cdf(x, b=alpha, scale=x_m))

    raise ValueError(
        f"Unsupported distribution '{distribution}'. "
        "Supported: normal, student_t, binomial, poisson, exponential, uniform, lognormal, power_law."
    )
