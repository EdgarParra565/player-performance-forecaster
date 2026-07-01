"""BetRivers sportsbook scraper config + team-line extractor.

BetRivers runs on the Kambi platform; its NBA lobby renders full team names
and groups spread / total / moneyline per game. Representative visible-text
shape (whitespace-collapsed):

    "Knicks at 76ers +1.5 -110 -1.5 -110 O 213.5 -110 U 213.5 -110 +100 -120"

Token order per game:
    <Away> at <Home>
    <away_spread> <away_spread_odds> <home_spread> <home_spread_odds>
    O <total> <over_odds> U <total> <under_odds>
    <away_ml> <home_ml>

TODO(real-capture): this parser + its test fixture were authored WITHOUT a
live authenticated BetRivers snapshot. Capture one via the Chrome :9222 CDP
host (`betrivers.com` basketball lobby) and re-validate the token order /
team-name rendering before trusting these rows in consensus.
"""

from __future__ import annotations

import re

from nba_model.scrapers.base import BookScraper, SessionMarkers
from nba_model.scrapers.team_names import TEAM_NAME_PATTERN, normalize_team


_ODDS = r"[+\-]\d{2,4}"
_SPREAD = r"[+\-]\d+(?:\.\d+)?"
_TOTAL = r"\d{2,3}(?:\.\d+)?"

# BetRivers renders full canonical names (no leading abbrev), e.g. "Knicks".
_GAME_RE = re.compile(
    r"(?P<away>" + TEAM_NAME_PATTERN + r")"
    r"\s+(?:@|at|vs\.?)\s+"
    r"(?P<home>" + TEAM_NAME_PATTERN + r")"
    r"\s+(?P<away_spread>" + _SPREAD + r")\s+(?P<away_spread_odds>" + _ODDS + r")"
    r"\s+(?P<home_spread>" + _SPREAD + r")\s+(?P<home_spread_odds>" + _ODDS + r")"
    r"\s+O\s+(?P<total>" + _TOTAL + r")\s+(?P<over_odds>" + _ODDS + r")"
    r"\s+U\s+" + _TOTAL + r"\s+(?P<under_odds>" + _ODDS + r")"
    r"\s+(?P<away_ml>" + _ODDS + r")\s+(?P<home_ml>" + _ODDS + r")"
)


def _normalize_minus_signs(text: str) -> str:
    """Map Unicode minus (U+2212) and en-dash (U+2013) to ASCII '-'."""
    return text.replace("−", "-").replace("–", "-")


def extract_team_lines(text: str) -> list[dict]:
    """Return one record per (game, market, side) found in BetRivers text."""
    text = _normalize_minus_signs(text)
    out: list[dict] = []
    for m in _GAME_RE.finditer(text):
        away = normalize_team(m.group("away"))
        home = normalize_team(m.group("home"))
        if not away or not home:
            continue
        raw = m.group(0)[:300]
        common = {"away_team": away, "home_team": home, "raw_text": raw}

        out.append({**common, "market_type": "spread", "side": "away",
                    "team": away,
                    "line_value": float(m.group("away_spread")),
                    "odds_american": int(m.group("away_spread_odds"))})
        out.append({**common, "market_type": "spread", "side": "home",
                    "team": home,
                    "line_value": float(m.group("home_spread")),
                    "odds_american": int(m.group("home_spread_odds"))})

        total = float(m.group("total"))
        out.append({**common, "market_type": "total", "side": "over",
                    "team": None, "line_value": total,
                    "odds_american": int(m.group("over_odds"))})
        out.append({**common, "market_type": "total", "side": "under",
                    "team": None, "line_value": total,
                    "odds_american": int(m.group("under_odds"))})

        out.append({**common, "market_type": "moneyline", "side": "away",
                    "team": away, "line_value": None,
                    "odds_american": int(m.group("away_ml"))})
        out.append({**common, "market_type": "moneyline", "side": "home",
                    "team": home, "line_value": None,
                    "odds_american": int(m.group("home_ml"))})
    return out


SCRAPER = BookScraper(
    name="betrivers",
    domain="betrivers.com",
    wait_selectors=(
        "[class*='market']",
        "[class*='outcome']",
        "[class*='selection']",
        "[data-testid*='odd']",
    ),
    extra_wait_seconds=4.0,
    session_markers=SessionMarkers(
        login_wall=(
            "log in",
            "sign up",
            "register",
            "create account",
        ),
        authenticated=(
            "over",
            "under",
            "points",
            "rebounds",
            "assists",
            "nba",
        ),
        min_authenticated_hits=4,
    ),
    team_line_extractor=extract_team_lines,
)
