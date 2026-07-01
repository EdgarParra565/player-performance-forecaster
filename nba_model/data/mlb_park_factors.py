"""MLB park-factor hook — the MLB analog to the NBA defense adjustment.

Park factors materially shift offensive baselines (Coors Field inflates runs,
Petco / pitcher parks suppress them). This module is a light first-pass lookup
keyed by the home-team code; it nudges a hitter's projected mean up/down by the
park's run environment, leaving pitcher and yes/no markets alone unless asked.

A factor of 1.00 is neutral; >1 inflates offense, <1 suppresses it.

TODO(validate): these values are rough, widely-cited multi-year run-factor
approximations, NOT validated against this repo's own data. Replace with
empirically-derived factors (e.g. from Statcast via the optional pybaseball
supplement, or season-park splits) before trusting them in production.
"""

from __future__ import annotations

# Home-team code -> approximate run park factor (neutral = 1.00).
# TODO(validate): unvalidated placeholders; see module docstring.
PARK_FACTORS: dict[str, float] = {
    "COL": 1.12,  # Coors Field — altitude, the canonical hitter park
    "BOS": 1.06,  # Fenway
    "CIN": 1.05,  # Great American Ball Park
    "TEX": 1.04,
    "PHI": 1.03,
    "ARI": 1.03,
    "BAL": 1.02,
    "KC":  1.02,
    "TOR": 1.01,
    "LAA": 1.01,
    "ATL": 1.00,
    "HOU": 1.00,
    "MIN": 1.00,
    "WSH": 1.00,
    "CHC": 1.00,
    "STL": 0.99,
    "NYY": 0.99,
    "MIL": 0.99,
    "PIT": 0.98,
    "LAD": 0.98,
    "NYM": 0.98,
    "CWS": 0.98,
    "TB":  0.97,
    "CLE": 0.97,
    "SEA": 0.96,  # T-Mobile Park — pitcher-friendly
    "OAK": 0.96,
    "DET": 0.96,
    "MIA": 0.95,  # loanDepot park
    "SD":  0.94,  # Petco Park
    "SF":  0.93,  # Oracle Park — marine layer, deep gaps
}

NEUTRAL_FACTOR = 1.00

# Stat types whose baseline scales with the park's run environment. Pitcher
# counting stats and discrete yes/no markets are deliberately left untouched
# by default (their park sensitivity is different and not yet modeled).
_PARK_SENSITIVE_STATS = frozenset({
    "hits", "total_bases", "home_runs", "rbis", "runs_scored", "singles",
})


def park_factor(home_team_code: str) -> float:
    """Return the run park factor for a home-team code (1.00 when unknown)."""
    return PARK_FACTORS.get(str(home_team_code or "").strip().upper(), NEUTRAL_FACTOR)


def adjust_for_park(
    projected_value: float,
    home_team_code: str,
    stat_type: str,
) -> float:
    """Scale a hitter projection by the park factor for park-sensitive stats.

    Non-park-sensitive stats (pitcher counting stats, yes/no markets) and
    unknown parks return ``projected_value`` unchanged.
    """
    if str(stat_type or "").strip().lower() not in _PARK_SENSITIVE_STATS:
        return float(projected_value)
    return float(projected_value) * park_factor(home_team_code)
