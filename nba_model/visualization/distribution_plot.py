"""Visualization helpers for model output distributions."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_distribution(mu: float, sigma: float, line: float):
    """
    Plot normal distribution of projected stat with over/under line marker.

    Args:
        mu: Expected value (mean) of distribution.
        sigma: Standard deviation of distribution.
        line: Sportsbook line to overlay as vertical marker.
    """
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    y = norm.pdf(x, mu, sigma)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.axvline(line, linestyle="--")
    plt.title("Player Points Distribution")
    plt.xlabel("Points")
    plt.ylabel("Density")
    plt.show()
