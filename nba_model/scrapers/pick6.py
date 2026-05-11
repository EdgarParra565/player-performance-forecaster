"""DraftKings Pick6 (DFS pickem) scraper config + player-prop parser.

Format observed on the basketball lobby (auth'd):

    "J. Brunson PG NYK @ PHI Today, 7:10 PM ... 26.5 ... Points More ... Less"
    "V. Wembanyama PF SAS @ MIN Today, 9:40 PM ... 25.5 ... Points More ... Less"

Player names are abbreviated (first-initial style, e.g. "J. Brunson").
We expand them against ``nba_active_players_ref`` so the chart's name
match works.  When an abbreviation maps to multiple active players, the
record is dropped (we'd rather miss than misattribute).
"""

from __future__ import annotations

import re
from typing import Optional

from nba_model.scrapers.base import BookScraper, SessionMarkers
from nba_model.scrapers.player_names import (
    load_active_player_names,
    resolve_player_name,
)


# Position abbrevs Pick6 uses.
_POS = r"(?:PG|SG|SF|PF|C|G|F)"
# Team abbrev (2-4 uppercase letters).
_TEAM = r"[A-Z]{2,4}"
# Pick6 row shape (post-icon-noise):
#   "<F. LastName> <POS> <TEAM> @ <TEAM> Today|Sat|Sun..., <time> ... <line> ... <Stat> More"
_ROW_RE = re.compile(
    r"(?P<player>[A-Z](?:[A-Z]?\.|[A-Za-z]{2,})\s+[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)?)"
    r"\s+" + _POS + r"\s+"
    + _TEAM + r"\s+(?:@|vs\.?)\s+" + _TEAM + r"\s+"
      r"(?:Today|Tomorrow|Mon|Tue|Wed|Thu|Fri|Sat|Sun)[A-Za-z,\s\d:]+(?:AM|PM)\s+"
      r".*?(?P<line>\d{1,3}(?:\.\d+)?)"
      r"\s+Plus\s+Icon\s+Plus\s+Icon\s+"
      r"(?P<stat>[A-Za-z][A-Za-z0-9 \-+\']{0,30}?)"
      r"\s+(?P<side>More|Less)",
    flags=re.DOTALL,
)


_ACTIVE_PLAYERS_CACHE: Optional[list[str]] = None


def _load_active_players(db_path: str = "data/database/nba_data.db") -> list[str]:
    """Cached wrapper around the shared loader."""
    global _ACTIVE_PLAYERS_CACHE
    if _ACTIVE_PLAYERS_CACHE is None:
        _ACTIVE_PLAYERS_CACHE = load_active_player_names(db_path)
    return _ACTIVE_PLAYERS_CACHE


def _expand_abbreviated_name(abbrev: str) -> Optional[str]:
    """Map ``"J. Brunson"`` → ``"Jalen Brunson"`` via the shared resolver.

    Returns ``None`` when zero or multiple active players match.  The
    resolution logic itself lives in ``scrapers/player_names.py`` so other
    scrapers can reuse the same expansion without duplicating the matrix
    of initial / surname / suffix rules.
    """
    return resolve_player_name(abbrev, _load_active_players())


def preprocess(text: str) -> str:
    """Strip Pick6 UI noise and expand abbreviated names.

    Output: " ".join(f"{full_name} {line} {stat} {side}") so the generic
    ``_CARD_PATTERNS`` in browser_prop_parser pick up the rows.
    """
    segments: list[str] = []
    seen: set[tuple] = set()
    for m in _ROW_RE.finditer(text):
        abbrev = m.group("player").strip()
        full = _expand_abbreviated_name(abbrev)
        if full is None:
            continue
        try:
            line = float(m.group("line"))
        except ValueError:
            continue
        stat = m.group("stat").strip()
        side = m.group("side").strip()
        key = (full.lower(), round(line, 3), stat.lower(), side.lower())
        if key in seen:
            continue
        seen.add(key)
        segments.append(f"{full} {line} {stat} {side}")
    return " ".join(segments)


SCRAPER = BookScraper(
    name="pick6",
    domain="pick6.draftkings.com",
    wait_selectors=(
        "[class*='player-pick']",
        "[class*='projection']",
        "[class*='pick']",
        "[data-testid*='pick']",
        "article",
    ),
    extra_wait_seconds=4.0,
    session_markers=SessionMarkers(
        login_wall=(
            "log in",
            "sign up",
            "create account",
            "verify identity",
        ),
        authenticated=(
            "more",
            "less",
            "higher",
            "lower",
            "nba",
            "points",
        ),
        min_authenticated_hits=4,
    ),
    prop_preprocess=preprocess,
    # No regex tuple: preprocess emits canonical "Name Line Stat Side" form,
    # which the generic _CARD_PATTERNS in browser_prop_parser handle.
    parser_regexes=(_ROW_RE,),  # presence signals "has player parser"
)
