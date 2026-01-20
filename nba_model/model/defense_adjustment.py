def adjust_mu_for_defense(
    mu: float,
    opponent_def_rating: float,
    league_avg_def_rating: float = 113.0,
    sensitivity: float = 0.4
) -> float:
    """
    Adjusts expected points based on opponent defense

    sensitivity controls how strongly defense affects scoring
    """
    diff = league_avg_def_rating - opponent_def_rating
    adjustment = diff * sensitivity / 10

    return mu + adjustment
