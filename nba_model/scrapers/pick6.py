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
import sqlite3
from typing import Optional

from nba_model.scrapers.base import BookScraper, SessionMarkers


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
    """Read canonical active-player names from the reference table.

    Cached at module level so repeated parser calls don't re-query.
    """
    global _ACTIVE_PLAYERS_CACHE
    if _ACTIVE_PLAYERS_CACHE is not None:
        return _ACTIVE_PLAYERS_CACHE
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT player_name FROM nba_active_players_ref"
        ).fetchall()
        conn.close()
        _ACTIVE_PLAYERS_CACHE = [r[0] for r in rows if r[0]]
    except Exception:
        _ACTIVE_PLAYERS_CACHE = []
    return _ACTIVE_PLAYERS_CACHE


def _expand_abbreviated_name(abbrev: str) -> Optional[str]:
    """Map "J. Brunson" → "Jalen Brunson" via active-players reference.

    Returns None when 0 or >1 active players match (we don't guess).
    """
    parts = abbrev.strip().split()
    if len(parts) < 2:
        return None
    first = parts[0].rstrip(".")
    last = " ".join(parts[1:])
    if not first or not last:
        return None
    initial = first[0].upper()

    matches = []
    for full in _load_active_players():
        full_parts = full.split()
        if len(full_parts) < 2:
            continue
        # Match "Brunson" against last token, OR against everything after first word.
        canon_first = full_parts[0]
        canon_last = " ".join(full_parts[1:])
        # Prefer exact first-name match (e.g. "Stephen" matches abbrev "Stephen Curry"
        # written out unabbreviated by Pick6 in some places).
        if first.lower() == canon_first.lower() and last.lower() == canon_last.lower():
            return full
        # Initial + last name match — most common case.
        if (canon_first[0].upper() == initial
                and canon_last.lower() == last.lower()):
            matches.append(full)
    if len(matches) == 1:
        return matches[0]
    return None


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
