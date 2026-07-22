"""VegasInsider aggregator scraper config + cross-book odds-grid parser.

VegasInsider republishes a public NBA player-prop odds grid: real American
odds from ~11 books, one column per book, sectioned by stat. Unlike the DFS
boards (which post no price), these are REAL odds — exactly what
``betting_lines`` / ``--use-market-lines`` / cross-book line-shopping are
starved for — so the rows land in ``betting_lines`` (via
``nba_model.data.vegasinsider_odds_ingestion``) attributed to the UNDERLYING
book with ``source='vegasinsider'`` for provenance.

Observed snapshot shape (whitespace-collapsed visible text), one section per
stat::

    ... See All Rebounds Odds Time Bet365 PrizePicks BetMGM DraftKings Caesars
    FanDuel HardRock Fanatics Sleeper Underdog RiversCasino › › › ... ›
    Victor Wembanyama o12.5 -110 + o12.5 -137 + o12.5 -120 + ... o12.5 -106 +
    Karl-Anthony Towns o11.5 -130 + o12 -137 + ...

Each player row carries exactly one over-only cell per book, in the header's
book order. Cells are over-only (``o<line> <odds>``) — we never invent an
under price. This module only *parses*; storage/ID-resolution lives in the
ingestion module so the parser stays a pure, unit-testable function.
"""

from __future__ import annotations

import re
from typing import Optional

from nba_model.scrapers.base import BookScraper, SessionMarkers

# Book columns in the exact left-to-right order VegasInsider renders them.
# Position in this tuple is the ONLY thing that attributes a cell to a book,
# so it must match the header row verbatim.
BOOK_ORDER: tuple[str, ...] = (
    "bet365", "prizepicks", "betmgm", "draftkings", "caesars", "fanduel",
    "hardrock", "fanatics", "sleeper", "underdog", "riverscasino",
)

# Map the aggregator's book label → our registry's canonical book name where
# they differ. Unknown books (e.g. bet365, which has no scraper) pass through
# unchanged, as the mission requires.
AGGREGATOR_BOOK_MAP: dict[str, str] = {
    "hardrock": "hardrockbet",
    "riverscasino": "betrivers",
}

# Header row that names the 11 book columns, verbatim.
_HEADER_BOOKS = (
    "Bet365 PrizePicks BetMGM DraftKings Caesars FanDuel "
    "HardRock Fanatics Sleeper Underdog RiversCasino"
)
# The stat label sits immediately before "Odds Time <books>". Allow a leading
# digit ("3 Pointers") plus letters/spaces/hyphens.
_SECTION_HEADER_RE = re.compile(
    r"(?P<stat>(?:\d[\d \-]*)?[A-Za-z][A-Za-z \-]*?)\s+Odds Time "
    + re.escape(_HEADER_BOOKS)
)
# One over-only cell: "o<line> <american-odds>".
_CELL_RE = re.compile(r"o(\d+(?:\.\d+)?)\s+([+-]\d{2,4})")
# A player row: a Mixed-Case name (1–4 words) followed by a run of ≥2 cells.
# Player names start with an uppercase letter; cells start with lowercase
# "o<digit>", so the two never collide.
_ROW_RE = re.compile(
    r"(?P<player>[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,3})\s+"
    r"(?P<cells>(?:o\d+(?:\.\d+)?\s+[+-]\d{2,4}\s*\+?\s*){2,})"
)


def _normalize_stat(raw: str) -> Optional[str]:
    """Map a section's stat label to a canonical ``betting_lines`` stat_type.

    Returns ``None`` for a stat we don't model (row skipped). ``"pointer"``
    is checked before ``"point"`` so "3 Pointers" doesn't collapse to points.
    """
    s = (raw or "").lower()
    if "pointer" in s:
        return "three_pointers_made"
    if "rebound" in s:
        return "rebounds"
    if "assist" in s:
        return "assists"
    if "point" in s:
        return "points"
    return None


def extract_odds_rows(text: str) -> list[dict]:
    """Parse the VegasInsider odds grid into per-(player, book) over rows.

    Each dict: ``{player_name, book, stat_type, line_value, over_odds}`` — one
    per book column per player, with the book already mapped to the registry
    name. Over-only, so callers store ``under_odds=None``. Rows whose cell
    count doesn't match the 11-book header are dropped (misaligned → cannot be
    safely attributed) rather than guessed at.
    """
    if not text:
        return []
    headers = [
        (m.start(), m.end(), _normalize_stat(m.group("stat")))
        for m in _SECTION_HEADER_RE.finditer(text)
    ]
    rows: list[dict] = []
    for i, (start, end, stat) in enumerate(headers):
        if stat is None:
            continue
        body_end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        body = text[end:body_end]
        for m in _ROW_RE.finditer(body):
            cells = _CELL_RE.findall(m.group("cells"))
            if len(cells) != len(BOOK_ORDER):
                continue
            player = m.group("player").strip()
            for book, (line, odds) in zip(BOOK_ORDER, cells):
                rows.append({
                    "player_name": player,
                    "book": AGGREGATOR_BOOK_MAP.get(book, book),
                    "stat_type": stat,
                    "line_value": float(line),
                    "over_odds": int(odds),
                })
    return rows


SCRAPER = BookScraper(
    name="vegasinsider",
    domain="vegasinsider.com",
    wait_selectors=(
        "[class*='odds']",
        "[class*='matchup']",
        "[class*='line']",
        "table",
    ),
    extra_wait_seconds=2.0,
    session_markers=SessionMarkers(
        login_wall=(),
        authenticated=("nba", "odds time", "draftkings", "fanduel"),
        min_authenticated_hits=2,
    ),
)
