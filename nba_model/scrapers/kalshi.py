"""Kalshi prediction-market scraper config + moneyline extractor.

Kalshi lists basketball games as decimal-odds contracts:

    "Game 3: New York at Philadelphia May 8 @ 7:00PM "
    "New York 2.01 x 47 % Philadelphia 1.86 x 52 % $2,013,976 vol"

We convert decimal odds → American odds for cross-book consistency.
Spread and total markets exist on Kalshi too but live on a per-game detail
page (linked as "Spread and Total 2 markets") — not extractable from the
games-index page alone, so this module emits moneylines only.
"""

from __future__ import annotations

import re

from nba_model.scrapers.base import BookScraper, SessionMarkers
from nba_model.scrapers.team_names import normalize_team


_DECIMAL = r"\d+\.\d{1,3}"
_PROB = r"\d{1,3}"  # implied probability percent (47 etc.)

# Game line shape:
#   "<Away_full> at <Home_full> <Month> <Day> @ <Time>(AM|PM)"
#   "<Away_full> <decimal_away> x <prob_away> %"
#   "<Home_full> <decimal_home> x <prob_home> %"
# We allow team names to be 1-3 words; the surrounding " at "/decimal odds
# disambiguate from prose.  Trailing words may be a single letter because
# Kalshi truncates "Los Angeles Lakers" → "Los Angeles L".
_TEAM = r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]*){0,2}"

_GAME_RE = re.compile(
    r"(?P<away_a>" + _TEAM + r")"
    r"\s+at\s+"
    r"(?P<home_a>" + _TEAM + r")"
    r"\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}"
    r"\s+@\s+\d{1,2}:\d{2}(?:AM|PM|am|pm)\s+"
    r"(?P<away_b>" + _TEAM + r")"
    r"\s+(?P<away_decimal>" + _DECIMAL + r")\s+x\s+" + _PROB + r"\s+%\s+"
    r"(?P<home_b>" + _TEAM + r")"
    r"\s+(?P<home_decimal>" + _DECIMAL + r")\s+x\s+" + _PROB + r"\s+%"
)


def _decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds (e.g. 2.01) to American odds (e.g. +101).

    decimal >= 2.0 → positive (underdog): (d-1) * 100
    decimal <  2.0 → negative (favorite): -100 / (d-1)
    """
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1.0) * 100))
    if decimal_odds <= 1.0:
        # Degenerate / malformed decimal (≤ 1.0 implies ≥100% or free money):
        # not a real price. Return 0 so the caller can skip it rather than
        # dividing by zero and dropping the whole snapshot.
        return 0
    return int(round(-100.0 / (decimal_odds - 1.0)))


def extract_team_lines(text: str) -> list[dict]:
    """Return moneyline records for each NBA game found in Kalshi text."""
    out: list[dict] = []
    for m in _GAME_RE.finditer(text):
        # Trust the second mention of each team (right next to its odds).
        away = normalize_team(m.group("away_b")) or normalize_team(m.group("away_a"))
        home = normalize_team(m.group("home_b")) or normalize_team(m.group("home_a"))
        if not away or not home:
            continue
        try:
            away_dec = float(m.group("away_decimal"))
            home_dec = float(m.group("home_decimal"))
        except ValueError:
            continue
        raw = m.group(0)[:300]
        common = {"away_team": away, "home_team": home, "raw_text": raw}
        out.append({**common, "market_type": "moneyline", "side": "away",
                    "team": away, "line_value": None,
                    "odds_american": _decimal_to_american(away_dec)})
        out.append({**common, "market_type": "moneyline", "side": "home",
                    "team": home, "line_value": None,
                    "odds_american": _decimal_to_american(home_dec)})
    return out


SCRAPER = BookScraper(
    name="kalshi",
    domain="kalshi.com",
    wait_selectors=(
        "[class*='market']",
        "[class*='contract']",
        "[class*='event']",
        "[data-testid*='market']",
    ),
    extra_wait_seconds=4.0,
    session_markers=SessionMarkers(
        login_wall=(
            "log in",
            "sign up",
            "create account",
        ),
        authenticated=(
            "yes",
            "no",
            "market",
            "nba",
            "spread and total",
        ),
        min_authenticated_hits=3,
    ),
    team_line_extractor=extract_team_lines,
)
