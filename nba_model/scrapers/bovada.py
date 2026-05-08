"""Bovada sportsbook scraper config + team-line extractor.

Format observed on the basketball lobby (NBA - Next Events block):

    "5/8/26 7:00 PM "
    "New York Knicks "
    "Philadelphia 76ers "
    "+ 692 Bets "
    "+2.5 (-115) -2.5 (-105) +115 -135 O 213.5 (-110) U 213.5 (-110)"

Order in the odds line:
  away_spread (away_spread_odds)
  home_spread (home_spread_odds)
  away_moneyline home_moneyline
  O total (over_odds)
  U total (under_odds)
"""

from __future__ import annotations

import re

from nba_model.scrapers.base import BookScraper, SessionMarkers
from nba_model.scrapers.team_names import TEAM_NAME_PATTERN, normalize_team


_ODDS = r"[+\-]\d{2,4}"
_SPREAD = r"[+\-]\d+(?:\.\d+)?"
_TOTAL = r"\d{2,3}(?:\.\d+)?"

_GAME_RE = re.compile(
    r"(?P<away>" + TEAM_NAME_PATTERN + r")\s+"
    r"(?P<home>" + TEAM_NAME_PATTERN + r")\s+"
    r"\+\s*\d+\s*Bets\s+"
    r"(?P<away_spread>" + _SPREAD + r")\s+\((?P<away_spread_odds>" + _ODDS + r")\)\s+"
    r"(?P<home_spread>" + _SPREAD + r")\s+\((?P<home_spread_odds>" + _ODDS + r")\)\s+"
    r"(?P<away_ml>" + _ODDS + r")\s+(?P<home_ml>" + _ODDS + r")\s+"
    r"O\s+(?P<total>" + _TOTAL + r")\s+\((?P<over_odds>" + _ODDS + r")\)\s+"
    r"U\s+" + _TOTAL + r"\s+\((?P<under_odds>" + _ODDS + r")\)"
)


def extract_team_lines(text: str) -> list[dict]:
    """Return one record per (game, market, side) found in Bovada text."""
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
    name="bovada",
    domain="bovada.lv",
    wait_selectors=(
        "[class*='market']",
        "[class*='outcome']",
        "[class*='line']",
        "[data-testid*='market']",
    ),
    extra_wait_seconds=4.0,
    session_markers=SessionMarkers(
        login_wall=("log in", "join now", "create account", "sign in"),
        authenticated=("spread", "total", "money", "nba", "today", "tomorrow"),
        min_authenticated_hits=4,
    ),
    team_line_extractor=extract_team_lines,
)
