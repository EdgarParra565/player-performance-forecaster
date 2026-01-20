import numpy as np

def simulate_sgp(
    mu_points: float,
    sigma_points: float,
    minutes_mu: float,
    minutes_sigma: float,
    point_line: float,
    minute_line: float,
    corr: float = 0.6,
    n: int = 20000
) -> float:
    """
    Simulates a Same-Game Parlay with correlation

    corr: correlation between minutes and points
    """
    cov = corr * sigma_points * minutes_sigma

    cov_matrix = [
        [sigma_points**2, cov],
        [cov, minutes_sigma**2]
    ]

    means = [mu_points, minutes_mu]

    samples = np.random.multivariate_normal(means, cov_matrix, n)

    points = samples[:, 0]
    minutes = samples[:, 1]

    hits = (points > point_line) & (minutes > minute_line)

    return hits.mean()

def simulate_multi_leg_sgp(
    means: list,
    cov_matrix: np.ndarray,
    lines: list,
    n: int = 20000
):
    """
    Simulates N-leg same-game parlay using multivariate normal
    means: list of expected values for each stat
    cov_matrix: NxN covariance matrix
    lines: list of betting lines for each stat
    """
    samples = np.random.multivariate_normal(means, cov_matrix, n)
    hits = np.all(samples > lines, axis=1)
    return hits.mean()
