from scipy.stats import norm


def prob_over(line: float, mu: float, sigma: float) -> float:
    """
    Closed-form probability of going over the line
    """
    if sigma <= 0:
        return 0.0

    return 1 - norm.cdf(line, mu, sigma)
