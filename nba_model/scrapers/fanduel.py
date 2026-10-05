"""FanDuel scraper config (sportsbook + DFS) + team-line extractor.

The DFS site (``www.fanduel.com``) lists tournaments only — no player or
team prop lines.  The sportsbook lives on a different host
(``sportsbook.fanduel.com``).  We register both via ``aliases`` so either
tab is picked up by the registry.

Sportsbook page format (basketball lobby), DraftKings-like but with an ``@``
game separator and a date/time block between the teams and the odds:

    "NY Knicks @ PHI 76ers Today 7:10pm +1.5 -108 O 213.5 -110 +102 "
    "-1.5 -112 U 213.5 -110 -122"

Token order per game:
    <away_abbrev> <Away> @ <home_abbrev> <Home> [<day> <time>]
    <away_spread> <away_spread_odds> O <total> <over_odds> <away_ml>
    <home_spread> <home_spread_odds> U <total> <under_odds> <home_ml>

Second layout (live capture 2026-09-28, /navigation/nba, logged out): full
team names with no abbreviation or ``@``, and the moneyline sits between the
spread and the total:

    "Boston Celtics Detroit Pistons +1.5 -110 +100 O 221.5 -115 "
    "-1.5 -110 -118 U 221.5 -105 same game parlay available Oct 20, 3:00pm ET"

    <Away> <Home>
    <away_spread> <away_spread_odds> <away_ml> O <total> <over_odds>
    <home_spread> <home_spread_odds> <home_ml> U <total> <under_odds>

Away-first, cross-checked against Caesars + Bovada captures of the same slate.
"""

from __future__ import annotations

import re

from nba_model.scrapers.base import BookScraper, SessionMarkers
from nba_model.scrapers.game_dates import find_date_hint
from nba_model.scrapers.team_names import TEAM_NAME_PATTERN, normalize_team


_ODDS = r"[+\-]\d{2,4}"
_SPREAD = r"[+\-]\d+(?:\.\d+)?"
_TOTAL = r"\d{2,3}(?:\.\d+)?"

# Team mention is "<abbrev> <Team Short>" (e.g. "NY Knicks", "PHI 76ers").
_TEAM = r"[A-Z]{2,4}\s+(?P<NAME>" + TEAM_NAME_PATTERN + r")"

_GAME_RE = re.compile(
    _TEAM.replace("NAME", "away")
    + r"\s+(?:@|at|vs\.?)\s+"
    + _TEAM.replace("NAME", "home")
    # Optional "Today 7:10pm" / "Sat 1:00pm" block before the odds.
    + r"\s+(?:(?P<date>[A-Za-z]{3,9})\s+\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?\s+)?"
    + r"(?P<away_spread>" + _SPREAD + r")"
      r"\s+(?P<away_spread_odds>" + _ODDS + r")"
      r"\s+O\s+(?P<total>" + _TOTAL + r")"
      r"\s+(?P<over_odds>" + _ODDS + r")"
      r"\s+(?P<away_ml>" + _ODDS + r")"
      r"\s+(?P<home_spread>" + _SPREAD + r")"
      r"\s+(?P<home_spread_odds>" + _ODDS + r")"
      r"\s+U\s+" + _TOTAL +
      r"\s+(?P<under_odds>" + _ODDS + r")"
      r"\s+(?P<home_ml>" + _ODDS + r")"
)


_GAME_RE_2026 = re.compile(
    r"(?P<away>" + TEAM_NAME_PATTERN + r")\s+(?P<home>" + TEAM_NAME_PATTERN + r")\s+"
    r"(?P<away_spread>" + _SPREAD + r")\s+(?P<away_spread_odds>" + _ODDS + r")\s+"
    r"(?P<away_ml>" + _ODDS + r")\s+"
    r"O\s+(?P<total>" + _TOTAL + r")\s+(?P<over_odds>" + _ODDS + r")\s+"
    r"(?P<home_spread>" + _SPREAD + r")\s+(?P<home_spread_odds>" + _ODDS + r")\s+"
    r"(?P<home_ml>" + _ODDS + r")\s+"
    r"U\s+" + _TOTAL + r"\s+(?P<under_odds>" + _ODDS + r")"
)


def _normalize_minus_signs(text: str) -> str:
    """Map Unicode minus (U+2212) and en-dash (U+2013) to ASCII '-'."""
    return text.replace("−", "-").replace("–", "-")


def extract_team_lines(text: str) -> list[dict]:
    """Return one record per (game, market, side) found in FanDuel text
    (both the legacy abbrev/@ layout and the 2026-09 full-name layout)."""
    text = _normalize_minus_signs(text)
    out: list[dict] = []
    matches = list(_GAME_RE.finditer(text)) + list(_GAME_RE_2026.finditer(text))
    for m in matches:
        away = normalize_team(m.group("away"))
        home = normalize_team(m.group("home"))
        if not away or not home:
            continue
        raw = m.group(0)[:300]
        # Legacy: "Today 7:10pm" inside the match; 2026: "... same game parlay
        # available Oct 20, 3:00pm ET" right after it.
        hint = m.groupdict().get("date") or find_date_hint(text, m.start(), m.end(), after=60)
        common = {"away_team": away, "home_team": home, "raw_text": raw,
                  "game_date_hint": hint}

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
    name="fanduel",
    domain="fanduel.com",
    aliases=("sportsbook.fanduel.com",),  # explicit subdomain match
    wait_selectors=(
        "[class*='player-prop']",
        "[class*='alternate-line']",
        "[class*='selection']",
        "[data-testid*='market']",
    ),
    extra_wait_seconds=4.0,
    session_markers=SessionMarkers(
        login_wall=(
            "log in",
            "sign up",
            "create account",
            "join now",
        ),
        authenticated=(
            "spread",
            "total",
            "moneyline",
            "nba",
            "points",
            "rebounds",
        ),
        min_authenticated_hits=4,
    ),
    team_line_extractor=extract_team_lines,
)
