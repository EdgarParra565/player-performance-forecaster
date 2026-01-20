import numpy as np


def monte_carlo_over(
    mu: float,
    sigma: float,
    line: float,
    n: int = 10000
) -> float:
    """
    Monte Carlo estimate of over probability
    """
    if sigma <= 0:
        return 0.0

    sims = np.random.normal(mu, sigma, n)
    return (sims > line).mean()
