import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_distribution(mu: float, sigma: float, line: float):
    """
    Plots points distribution with over/under line
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
