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

Second layout (live capture 2026-09-28, /sports/basketball/nba "NBA - NEXT
EVENTS"): odds come BEFORE the team names, without parentheses, and the
column labels appear only ahead of the first game; ``EVEN`` = +100:

    "10/20/2026 12:00 PM Spread +1.5 -105 -1.5 -115 Win +105 -125 "
    "Total o 221.5 -110 u 221.5 -110 Boston Celtics Detroit Pistons + 115 "
    "10/20/2026 4:00 PM +5.5 -110 -5.5 -110 +165 -195 o 231.5 -110 "
    "u 231.5 -110 Philadelphia 76ers New York Knicks + 117"

Same away-first order (cross-checked against Caesars' capture of the same
slate: PHI +5.5 / +162 at Caesars, +5.5 / +165 here).
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


_ODDS_OR_EVEN = r"(?:[+\-]\d{2,4}|EVEN)"

_GAME_RE_2026 = re.compile(
    r"\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*[AP]M\s+"
    r"(?:Spread\s+)?"
    r"(?P<away_spread>" + _SPREAD + r")\s+(?P<away_spread_odds>" + _ODDS_OR_EVEN + r")\s+"
    r"(?P<home_spread>" + _SPREAD + r")\s+(?P<home_spread_odds>" + _ODDS_OR_EVEN + r")\s+"
    r"(?:Win\s+)?"
    r"(?P<away_ml>" + _ODDS_OR_EVEN + r")\s+(?P<home_ml>" + _ODDS_OR_EVEN + r")\s+"
    r"(?:Total\s+)?"
    r"[oO]\s+(?P<total>" + _TOTAL + r")\s+(?P<over_odds>" + _ODDS_OR_EVEN + r")\s+"
    r"[uU]\s+" + _TOTAL + r"\s+(?P<under_odds>" + _ODDS_OR_EVEN + r")\s+"
    r"(?P<away>" + TEAM_NAME_PATTERN + r")\s+"
    r"(?P<home>" + TEAM_NAME_PATTERN + r")\s+\+\s*\d+"
)


def _odds(value: str) -> int:
    return 100 if value.upper() == "EVEN" else int(value)


def extract_team_lines(text: str) -> list[dict]:
    """Return one record per (game, market, side) found in Bovada text.

    Tries both known lobby layouts (team-names-first with parenthesized odds,
    and the 2026-09 odds-first layout)."""
    out: list[dict] = []
    matches = list(_GAME_RE.finditer(text)) + list(_GAME_RE_2026.finditer(text))
    for m in matches:
        away = normalize_team(m.group("away"))
        home = normalize_team(m.group("home"))
        if not away or not home:
            continue
        raw = m.group(0)[:300]
        common = {"away_team": away, "home_team": home, "raw_text": raw}

        out.append({**common, "market_type": "spread", "side": "away",
                    "team": away,
                    "line_value": float(m.group("away_spread")),
                    "odds_american": _odds(m.group("away_spread_odds"))})
        out.append({**common, "market_type": "spread", "side": "home",
                    "team": home,
                    "line_value": float(m.group("home_spread")),
                    "odds_american": _odds(m.group("home_spread_odds"))})

        total = float(m.group("total"))
        out.append({**common, "market_type": "total", "side": "over",
                    "team": None, "line_value": total,
                    "odds_american": _odds(m.group("over_odds"))})
        out.append({**common, "market_type": "total", "side": "under",
                    "team": None, "line_value": total,
                    "odds_american": _odds(m.group("under_odds"))})

        out.append({**common, "market_type": "moneyline", "side": "away",
                    "team": away, "line_value": None,
                    "odds_american": _odds(m.group("away_ml"))})
        out.append({**common, "market_type": "moneyline", "side": "home",
                    "team": home, "line_value": None,
                    "odds_american": _odds(m.group("home_ml"))})
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
