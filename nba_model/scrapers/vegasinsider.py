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
# digit ("3 Pointers") plus letters/spaces/hyphens. The label is length-bounded:
# an unbounded lazy run rescans from every start position (quadratic — ~13s on
# 60K chars of plain words).
_SECTION_HEADER_RE = re.compile(
    r"(?P<stat>(?:\d[\d \-]{0,10})?[A-Za-z][A-Za-z \-]{0,60}?)\s+Odds Time "
    + re.escape(_HEADER_BOOKS)
)
# Every "Odds Time" marks a section boundary, whether or not its header is
# valid — so a section whose header we reject can never bleed its rows into
# the previous section under that section's stat/books.
_ODDS_TIME_RE = re.compile(r"Odds Time")
# One over-only cell: "o<line> <american-odds>". Real American odds have
# |odds| >= 100 (3-5 digits); "-11" is a truncated cell, not a price.
_CELL_RE = re.compile(r"o(\d+(?:\.\d+)?)\s+([+-]\d{3,5})(?!\d)")
# A player row: a Mixed-Case name (1–4 words) followed by a run of ≥2 cells.
# Player names start with an uppercase letter; cells start with lowercase
# "o<digit>", so the two never collide.
_ROW_RE = re.compile(
    r"(?P<player>[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,3})\s+"
    r"(?P<cells>(?:o\d+(?:\.\d+)?\s+[+-]\d{3,5}(?!\d)\s*\+?\s*){2,})"
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
    boundaries = [m.start() for m in _ODDS_TIME_RE.finditer(text)]
    rows: list[dict] = []
    for h in _SECTION_HEADER_RE.finditer(text):
        stat = _normalize_stat(h.group("stat"))
        if stat is None:
            continue
        own_odds_time = h.end() - len(_HEADER_BOOKS) - len("Odds Time ")
        later = [b for b in boundaries if b > own_odds_time]
        body_end = later[0] if later else len(text)
        body = text[h.end():body_end]
        matches = list(_ROW_RE.finditer(body))
        if not later and matches and _ends_mid_row(body):
            matches = matches[:-1]  # capture cut the final row mid-cell
        for m in matches:
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


# ---------------------------------------------------------------------------
# MLB player-props grid (vegasinsider.com/mlb/odds/player-props/)
# ---------------------------------------------------------------------------
#
# Authored against the REAL 2026-07-22 capture (web_text_snapshots id 155,
# 60,000 chars). Same over-cell idea as the NBA grid, but the token order
# differs in ways that make the NBA extractor read 0 rows:
#
#   * every section carries its OWN book header (Strikeouts: 10 books,
#     Home Runs: 6, Total Bases: 8, Runs Batted In: 7), followed by exactly one
#     "›" per book column::
#
#         Strikeouts Odds Time Bet365 PrizePicks BetMGM DraftKings Caesars
#         FanDuel Fanatics Sleeper Underdog RiversCasino › › › › › › › › › ›
#         Reid Detmers o5.5 -165 + o6 -137 + o6.5 +105 ... o5.5 -141 +
#
#   * a cell is ``o<line> <odds>`` OR a bare ``<odds>`` with no line token
#     (the Home Runs section is mostly bare: it is the line-less yes/no
#     "to hit a home run" market, i.e. the 0.5 threshold);
#   * stray ``0`` tokens appear between cells (no price) and ``+`` tokens are
#     add-to-betslip buttons;
#   * EMPTY book cells render nothing, so most rows carry fewer cells than the
#     header has books. Position is the only book attribution, so only rows
#     whose cell count equals the section's book count are emitted — partial
#     rows are dropped, never guessed at (in the 07-22 capture: 9/35 K rows,
#     11/409 HR rows, 23/369 TB rows, 0/105 RBI rows are fully aligned).

# Section label keyword -> canonical sports/mlb.py stat key. ONLY the four
# sections observed in the real capture are mapped; any other section label
# returns None (skipped) until it has been seen in a real capture. The VI
# "Strikeouts" section lists starting pitchers (lines 3.5-7.5), so it is the
# pitcher market — hitter and pitcher stat groups stay separate.
_MLB_SECTION_STATS: tuple[tuple[str, str], ...] = (
    ("home run", "anytime_home_run"),
    ("total bases", "total_bases"),
    ("runs batted in", "rbis"),
    ("strikeout", "strikeouts_pitcher"),
)

# Sections whose line-less (bare) price is the yes/no market at the 0.5
# threshold. Anywhere else a bare cell holds its column position but is not
# emitted (its line is unknown).
_MLB_YESNO_STATS = frozenset({"anytime_home_run"})

# Book columns that hold their position (so alignment stays exact) but are
# NOT emitted: in the 07-22 capture BetMGM's MLB cells are self-inconsistent
# with every other book — total-bases "o0.5 +105" / "o4.5 +110", strikeout
# alt lines priced against the field, and anytime-HR prices ~1.5x the
# consensus. Treat as untrusted until a fresh capture shows it realigned.
MLB_UNTRUSTED_BOOKS = frozenset({"betmgm"})

MLB_PARSER_VERSION = "vi_mlb_grid_v1"

# "<stat label> Odds Time <Book> <Book> ... › › ..." — the book run ends at
# the first "›"; the arrow count must equal the book count (checked below).
_MLB_SECTION_RE = re.compile(
    r"(?P<stat>[A-Za-z][A-Za-z \-]{0,60}?)\s+(?P<ot>Odds Time)\s+"
    r"(?P<books>(?:[A-Z][A-Za-z0-9]*\s+)+?)"
    r"(?P<arrows>(?:›\s*)+)"
)
_MLB_LINE_TOKEN = re.compile(r"^o(\d+(?:\.\d+)?)$")
_MLB_ODDS_TOKEN = re.compile(r"^[+-]\d{3,5}$")
_MLB_SKIP_TOKENS = frozenset({"›", "+", "0"})


def _normalize_mlb_stat(raw: str) -> Optional[str]:
    # The label regex is leftmost-greedy over letters, so it can drag in
    # preceding page copy ("... homerun odds Strikeouts"); only the label's
    # trailing words ("See All Runs Batted In" -> "Runs Batted In") count.
    s = " ".join((raw or "").lower().split()[-4:])
    for keyword, stat_key in _MLB_SECTION_STATS:
        if keyword in s:
            return stat_key
    return None


def _tokenize_mlb_rows(body: str) -> list[tuple[str, list[tuple[Optional[float], int]]]]:
    """Split a section body into ``(player, [(line|None, odds), ...])`` rows.

    A row is closed by the next name token (or the end of ``body``); the
    caller drops the final row when the document ends mid-row
    (``_ends_mid_row``), so a cut-off row never masquerades as complete.
    """
    rows: list[tuple[str, list[tuple[Optional[float], int]]]] = []
    name: list[str] = []
    cells: list[tuple[Optional[float], int]] = []
    pending_line: Optional[float] = None
    for tok in body.split():
        if tok in _MLB_SKIP_TOKENS:
            continue
        m = _MLB_LINE_TOKEN.match(tok)
        if m:
            pending_line = float(m.group(1))
            continue
        if _MLB_ODDS_TOKEN.match(tok):
            cells.append((pending_line, int(tok)))
            pending_line = None
            continue
        if cells:
            rows.append((" ".join(name), cells))
            name, cells = [], []
        pending_line = None
        name.append(tok)
    if name and cells:
        rows.append((" ".join(name), cells))
    return rows


def _ends_mid_row(body: str) -> bool:
    """True when the text stops on a cell/line/button token (no trailing page
    copy after the grid) — i.e. a capped capture that may have cut the final
    row, possibly mid-price ("+1200" -> "+12")."""
    toks = body.split()
    if not toks:
        return False
    last = toks[-1]
    if last == "+":
        # A lone "+" is either the betslip button after a COMPLETE price
        # ("-106 +") or the sign of a price the cap cut off ("o0.5 +").
        return not (len(toks) > 1 and _MLB_ODDS_TOKEN.match(toks[-2]))
    return bool(
        _MLB_LINE_TOKEN.match(last)
        or re.match(r"^[+-]\d*$", last)  # a price, possibly cut mid-digits
    )


def extract_mlb_odds_rows(text: str) -> list[dict]:
    """Parse the VegasInsider MLB player-props grid into per-(player, book) rows.

    Each dict: ``{player_name, book, stat_type, market_shape, line_value,
    side, odds}``. Over-only (the grid posts no under/"No" prices), so
    ``side`` is always ``'over'`` (for yes/no markets: Yes -> over, same
    convention as ``mlb_props.preprocess_mlb_props``). ``market_shape`` is
    ``'yes_no'`` for anytime-HR cells and ``'over_under'`` otherwise.

    Only fully-aligned rows are emitted (see module notes above); the last row
    of the document is dropped when the text ends mid-row, because a capped
    capture can cut it mid-cell.
    """
    if not text:
        return []
    text = re.sub(r"\s+", " ", text)
    # Every "Odds Time" is a section boundary, valid header or not: a section
    # whose header is rejected yields 0 rows instead of bleeding into the
    # previous section under that section's stat and book columns.
    boundaries = [m.start() for m in _ODDS_TIME_RE.finditer(text)]
    headers = []
    for m in _MLB_SECTION_RE.finditer(text):
        books = m.group("books").split()
        arrows = m.group("arrows").count("›")
        if not books or arrows != len(books):
            continue  # header/arrow mismatch -> layout drift, don't guess
        headers.append((m.start("ot"), m.end(), _normalize_mlb_stat(m.group("stat")),
                        [b.lower() for b in books]))

    rows: list[dict] = []
    for ot_start, end, stat, books in headers:
        later = [b for b in boundaries if b > ot_start]
        is_last_section = not later
        body_end = later[0] if later else len(text)
        body = text[end:body_end]
        parsed = _tokenize_mlb_rows(body)
        if is_last_section and parsed and _ends_mid_row(body):
            parsed = parsed[:-1]  # cut off by the capture cap mid-row
        if stat is None:
            continue
        yes_no = stat in _MLB_YESNO_STATS
        for player, cells in parsed:
            if len(cells) != len(books):
                continue
            for book, (line, odds) in zip(books, cells):
                if book in MLB_UNTRUSTED_BOOKS:
                    continue
                if yes_no:
                    # Anytime HR: bare price or explicit o0.5 only.
                    if line not in (None, 0.5):
                        continue
                    line = 0.5
                elif line is None:
                    continue  # bare cell outside a yes/no market: line unknown
                rows.append({
                    "player_name": player.strip(),
                    "book": AGGREGATOR_BOOK_MAP.get(book, book),
                    "stat_type": stat,
                    "market_shape": "yes_no" if yes_no else "over_under",
                    "line_value": float(line),
                    "side": "over",
                    "odds": int(odds),
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
    # The MLB props grid overflows the 60K global cap (the 2026-07-22 capture
    # is cut mid-"Runs Batted In"); lift it so later sections are captured.
    max_text_chars=250_000,
    session_markers=SessionMarkers(
        login_wall=(),
        authenticated=("nba", "odds time", "draftkings", "fanduel"),
        min_authenticated_hits=2,
    ),
)
