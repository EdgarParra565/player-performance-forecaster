"""Caesars sportsbook scraper config + team-line extractor.

Format observed on the basketball lobby page:

    "NYK NY Knicks New York Knicks +1.5 -105 +105 213.5 -110 vs "
    "PHI PHI 76ers Philadelphia 76ers -1.5 -115 -125 213.5 -110"

Each side is: <abbrev> <short_with_abbrev> <full_name> <spread> <spread_odds>
<moneyline> <total> <total_odds>.  The two sides are joined by " vs ".
Total appears twice (once per side) but they're identical, so we keep one.
"""

from __future__ import annotations

import re

from nba_model.scrapers.base import BookScraper, SessionMarkers
from nba_model.scrapers.team_names import TEAM_NAME_PATTERN, normalize_team


_ODDS = r"[+\-]\d{2,4}"
_SPREAD = r"[+\-]\d+(?:\.\d+)?"
_TOTAL = r"\d{2,3}(?:\.\d+)?"

# A Caesars side block: <abbrev> <abbrev> <Full Team Name> <spread>
# <spread_odds> <ml_odds> <total> <total_odds>.  We capture only the parts
# we need; leading abbreviations are skipped via [A-Z]{2,4}.
_SIDE = (
    r"[A-Z]{2,4}\s+[A-Z]{2,4}\s+"
    r"(?P<TEAM>" + TEAM_NAME_PATTERN + r"(?:\s+" + TEAM_NAME_PATTERN + r")?)"
    r"\s+(?P<SPREAD>" + _SPREAD + r")"
    r"\s+(?P<SPREAD_ODDS>" + _ODDS + r")"
    r"\s+(?P<ML>" + _ODDS + r")"
    r"\s+(?P<TOTAL>" + _TOTAL + r")"
    r"\s+(?P<TOTAL_ODDS>" + _ODDS + r")"
)
_GAME_RE = re.compile(
    _SIDE.replace("TEAM", "away").replace("SPREAD_ODDS", "away_spread_odds")
         .replace("SPREAD", "away_spread").replace("ML", "away_ml")
         .replace("TOTAL_ODDS", "away_total_odds").replace("TOTAL", "away_total")
    + r"\s+vs\s+"
    + _SIDE.replace("TEAM", "home").replace("SPREAD_ODDS", "home_spread_odds")
           .replace("SPREAD", "home_spread").replace("ML", "home_ml")
           .replace("TOTAL_ODDS", "home_total_odds").replace("TOTAL", "home_total")
)


def extract_team_lines(text: str) -> list[dict]:
    """Return one record per (game, market, side) found in Caesars text."""
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

        # Caesars repeats the total on each side; they're always identical
        # but we still split into over/under for the schema.  Caesars doesn't
        # show separate over/under odds — the two reported are away_total_odds
        # and home_total_odds.  Treat away as 'over', home as 'under'.
        total = float(m.group("away_total"))
        out.append({**common, "market_type": "total", "side": "over",
                    "team": None, "line_value": total,
                    "odds_american": int(m.group("away_total_odds"))})
        out.append({**common, "market_type": "total", "side": "under",
                    "team": None, "line_value": total,
                    "odds_american": int(m.group("home_total_odds"))})

        out.append({**common, "market_type": "moneyline", "side": "away",
                    "team": away, "line_value": None,
                    "odds_american": int(m.group("away_ml"))})
        out.append({**common, "market_type": "moneyline", "side": "home",
                    "team": home, "line_value": None,
                    "odds_american": int(m.group("home_ml"))})
    return out


SCRAPER = BookScraper(
    name="caesars",
    domain="caesars.com",
    wait_selectors=(
        "[class*='market']",
        "[class*='outcome']",
        "[class*='selection']",
        "[data-testid*='market']",
    ),
    extra_wait_seconds=4.0,
    session_markers=SessionMarkers(
        login_wall=(
            "sign in",
            "log in",
            "register",
            "create account",
        ),
        authenticated=(
            "spread",
            "total",
            "money",
            "nba",
            "today",
            "tomorrow",
        ),
        min_authenticated_hits=4,
    ),
    team_line_extractor=extract_team_lines,
)
