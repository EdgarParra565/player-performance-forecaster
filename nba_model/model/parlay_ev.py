def calculate_parlay_ev(prob: float, odds: int, stake: float = 1.0) -> float:
    """
    Expected value of a parlay bet
    """
    if odds > 0:
        payout = odds / 100
    else:
        payout = 100 / abs(odds)
    return (prob * payout) - ((1 - prob) * stake)

def filter_profitable_parlays(parlay_list: list, ev_threshold: float = 0.05):
    """
    parlay_list: list of tuples (parlay_name, prob, odds)
    Returns only parlays with EV above threshold
    """
    return [p for p in parlay_list if calculate_parlay_ev(p[1], p[2]) > ev_threshold]
