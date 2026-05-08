"""BetMGM sportsbook scraper config + team-line extractor.

Player props live behind a per-game click-through, so the player-prop side
remains a stub. The basketball lobby itself shows full game lines for each
upcoming game in this format:

    "Today • 7:10 PM • Amazon Spread Total Money Knicks 53-29 76ers 45-37 "
    "+1.5 -110 -1.5 -110 O 213.5 -110 U 213.5 -110 +100 -120"

That's enough to extract spread + total + moneyline per game, which is what
this module does.
"""

from __future__ import annotations

import re

from nba_model.scrapers.base import BookScraper, SessionMarkers
from nba_model.scrapers.team_names import TEAM_NAME_PATTERN, normalize_team


# American odds: +110 / -110 / +100 / -120 (3-4 digits, leading sign required).
_ODDS = r"[+\-]\d{2,4}"
# Spread (e.g. +1.5, -8.5): always signed, always one decimal.
_SPREAD = r"[+\-]\d+(?:\.\d+)?"
# Total (e.g. 213.5): unsigned, optional decimal.
_TOTAL = r"\d{2,3}(?:\.\d+)?"

# Sequence we expect after the team-name + record block:
#   <away_spread> <away_spread_odds> <home_spread> <home_spread_odds>
#   O <total> <over_odds> U <total> <under_odds>
#   <away_ml> <home_ml>
_GAME_LINE_RE = re.compile(
    r"(?P<away>" + TEAM_NAME_PATTERN + r")"
    r"\s+\d+-\d+\s+"                              # away record (W-L)
    r"(?P<home>" + TEAM_NAME_PATTERN + r")"
    r"\s+\d+-\d+\s+"                              # home record
    r"(?P<away_spread>" + _SPREAD + r")\s+(?P<away_spread_odds>" + _ODDS + r")\s+"
    r"(?P<home_spread>" + _SPREAD + r")\s+(?P<home_spread_odds>" + _ODDS + r")\s+"
    r"O\s+(?P<total>" + _TOTAL + r")\s+(?P<over_odds>" + _ODDS + r")\s+"
    r"U\s+" + _TOTAL + r"\s+(?P<under_odds>" + _ODDS + r")\s+"
    r"(?P<away_ml>" + _ODDS + r")\s+(?P<home_ml>" + _ODDS + r")"
)


def extract_team_lines(text: str) -> list[dict]:
    """Return one record per (game, market, side) found in BetMGM text."""
    out: list[dict] = []
    for m in _GAME_LINE_RE.finditer(text):
        away = normalize_team(m.group("away"))
        home = normalize_team(m.group("home"))
        if not away or not home:
            continue
        raw = m.group(0)[:300]
        common = {
            "away_team": away,
            "home_team": home,
            "raw_text": raw,
        }
        # spread (one row per side)
        out.append({**common, "market_type": "spread", "side": "away",
                    "team": away,
                    "line_value": float(m.group("away_spread")),
                    "odds_american": int(m.group("away_spread_odds"))})
        out.append({**common, "market_type": "spread", "side": "home",
                    "team": home,
                    "line_value": float(m.group("home_spread")),
                    "odds_american": int(m.group("home_spread_odds"))})
        # total (over / under)
        total = float(m.group("total"))
        out.append({**common, "market_type": "total", "side": "over",
                    "team": None,
                    "line_value": total,
                    "odds_american": int(m.group("over_odds"))})
        out.append({**common, "market_type": "total", "side": "under",
                    "team": None,
                    "line_value": total,
                    "odds_american": int(m.group("under_odds"))})
        # moneyline (no line value)
        out.append({**common, "market_type": "moneyline", "side": "away",
                    "team": away,
                    "line_value": None,
                    "odds_american": int(m.group("away_ml"))})
        out.append({**common, "market_type": "moneyline", "side": "home",
                    "team": home,
                    "line_value": None,
                    "odds_american": int(m.group("home_ml"))})
    return out


SCRAPER = BookScraper(
    name="betmgm",
    domain="betmgm.com",
    wait_selectors=(
        "[class*='option-pane']",
        "[class*='option-indicator']",
        "[class*='player-prop']",
        "[data-testid*='option']",
    ),
    extra_wait_seconds=4.0,
    session_markers=SessionMarkers(
        login_wall=(
            "log in",
            "register",
            "create account",
            "sign in",
        ),
        authenticated=(
            "spread",
            "total",
            "money",
            "nba",
            "tomorrow",
            "today",
        ),
        min_authenticated_hits=4,
    ),
    team_line_extractor=extract_team_lines,
)
