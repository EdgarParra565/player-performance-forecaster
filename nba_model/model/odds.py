"""Odds conversion and expected-value utility functions."""

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
    stake: float = 1.0,
    push_prob: float = 0.0,
) -> float:
    """
    Calculates EV per unit stake.

    ``push_prob`` is the probability the bet pushes (stake refunded): it is
    removed from the losing mass, so a side's loss probability is
    ``1 - prob - push_prob`` rather than ``1 - prob``.
    """
    if odds > 0:
        payout = odds / 100
    else:
        payout = 100 / abs(odds)

    return (prob * payout) - ((1 - prob - push_prob) * stake)

def odds_to_prob(odds: int) -> float:
    return american_to_implied_prob(odds)


def _implied_or_none(odds):
    """American odds → implied probability; None for missing / |odds| < 100."""
    try:
        o = int(odds)
    except (TypeError, ValueError):
        return None
    if abs(o) < 100:
        return None
    return american_to_implied_prob(o)


def main_line_index(candidates, line_key: str = "line_value") -> int:
    """Index of a book's MAIN line within one scrape's alt-line ladder.

    Books (Bovada, Odds API alternates) post several rungs per player/stat in
    a single scrape; "latest line" and CLV must use the main one. Rule (same
    as ``api.services._main_line``): the most balanced market — smallest
    ``|implied(over) - implied(under)|``, ties → lower line. Rungs without
    both prices fall back to the median line (lower-middle).
    ``candidates`` are mappings with ``line_key``, ``over_odds``, ``under_odds``.
    """
    if len(candidates) <= 1:
        return 0
    priced = []
    for i, c in enumerate(candidates):
        po = _implied_or_none(c.get("over_odds"))
        pu = _implied_or_none(c.get("under_odds"))
        if po is not None and pu is not None:
            priced.append((abs(po - pu), float(c.get(line_key)), i))
    if priced:
        return min(priced)[2]
    ordered = sorted(range(len(candidates)),
                     key=lambda i: (float(candidates[i].get(line_key)), i))
    return ordered[(len(ordered) - 1) // 2]
