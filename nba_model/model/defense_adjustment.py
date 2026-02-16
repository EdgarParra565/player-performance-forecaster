"""Defense-based expectation adjustment utilities."""


def adjust_mu_for_defense(
    mu: float,
    opponent_def_rating: float,
    league_avg_def_rating: float = 113.0,
    sensitivity: float = 0.4
) -> float:
    """
    Adjust expected stat expectation based on opponent defensive rating.

    Args:
        mu: Baseline expected value before defense adjustment.
        opponent_def_rating: Opponent defensive rating (lower is better defense).
        league_avg_def_rating: League-average defensive rating baseline.
        sensitivity: Controls how strongly defense differential affects expectation.

    Returns:
        Defense-adjusted expected value.
    """
    diff = league_avg_def_rating - opponent_def_rating
    adjustment = diff * sensitivity / 10

    return mu + adjustment
