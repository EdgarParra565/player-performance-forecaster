def american_to_implied_prob(odds: int) -> float:
    """
    Converts American odds to implied probability
    """
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)


def expected_value(
    prob: float,
    odds: int,
    stake: float = 1.0
) -> float:
    """
    Calculates EV per unit stake
    """
    if odds > 0:
        payout = odds / 100
    else:
        payout = 100 / abs(odds)

    return (prob * payout) - ((1 - prob) * stake)

def odds_to_prob(odds: int) -> float:
    return american_to_implied_prob(odds)
