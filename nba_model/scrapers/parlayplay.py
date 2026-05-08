"""ParlayPlay (DFS pickem) scraper config + player-prop parser.

ParlayPlay shows ALTERNATE lines per (player, stat).  Format observed:

    "Jalen Brunson PG - NYK NYK @ PHI 7:10 PM "
    "Less Points More 2.5x 23.5 Points 1.43x 🔥 2.21x 24.5 Points 1.47x ..."
    "Show more Points lines (17) "
    "Less Rebounds More 1.5 Rebounds 1.06x 2.26x 2.5 Rebounds 1.46x 1.5x ..."

Each (player, stat) section contains multiple alternate lines.  For the
cross-book consensus we want ONE canonical line per (player, stat) — we
pick the median, which lines up well with what other DFS books quote.
"""

from __future__ import annotations

import re
from statistics import median

from nba_model.scrapers.base import BookScraper, SessionMarkers

# Position abbrev (e.g. "PG", "SF", "PF").
_POS = r"(?:PG|SG|SF|PF|C|G|F)"
# Player header: "<Full Name> <POS> - <TEAM> <TEAM> @ <TEAM> <time> <AM|PM>"
_PLAYER_HEADER_RE = re.compile(
    r"(?P<player>[A-Z][A-Za-z'\-\.]+(?:\s+[A-Z][A-Za-z'\-\.]+){1,3})"
    r"\s+" + _POS + r"\s+-\s+[A-Z]{2,4}\s+"
    r"[A-Z]{2,4}\s+(?:@|vs\.?)\s+[A-Z]{2,4}\s+"
    r"\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)"
)
# Stat block header: "Less <Stat> More" — e.g. "Less Points More",
# "Less 3PT Made More", "Less Pts + Reb + Ast More".
_STAT_HEADER_RE = re.compile(
    r"Less\s+(?P<stat>[A-Za-z0-9][A-Za-z0-9 \-+\']{0,30}?)\s+More"
)
# Line within a stat block: any "<line>.<dec> <Stat>" mention.
# We don't try to disambiguate Less vs More multipliers — we only need the
# numeric line.  All lines for the stat section are collected and we pick
# the median.
_LINE_PATTERN = r"(?P<line>\d+(?:\.\d+)?)"


def _stat_token_re(stat: str) -> re.Pattern:
    """Build a regex that finds all line values for ``stat`` in a text block."""
    # Escape the stat label so embedded "+" / "-" don't break the regex.
    return re.compile(_LINE_PATTERN + r"\s+" + re.escape(stat) + r"\b")


def preprocess(text: str) -> str:
    """Emit canonical "Name Line Stat Over" segments for the generic parser.

    We always emit ``Over`` (mapped to ``over`` by ``_canonicalize_side``).
    The line value is the median of the alternate lines listed under each
    (player, stat) block, which approximates what one-line books like PP/UD
    quote for the same player.
    """
    segments: list[str] = []
    seen: set[tuple] = set()

    # Split text into player blocks by player-header positions.
    headers = list(_PLAYER_HEADER_RE.finditer(text))
    for idx, header in enumerate(headers):
        player = header.group("player").strip()
        block_start = header.end()
        block_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        block = text[block_start:block_end]

        # Find stat headers within this player's block.
        stat_headers = list(_STAT_HEADER_RE.finditer(block))
        for sidx, sh in enumerate(stat_headers):
            stat = sh.group("stat").strip()
            sblock_start = sh.end()
            sblock_end = (stat_headers[sidx + 1].start()
                          if sidx + 1 < len(stat_headers) else len(block))
            sblock = block[sblock_start:sblock_end]

            line_re = _stat_token_re(stat)
            lines = [
                float(m.group("line"))
                for m in line_re.finditer(sblock)
            ]
            if not lines:
                continue
            canonical_line = median(lines)
            key = (player.lower(), stat.lower(), round(canonical_line, 3))
            if key in seen:
                continue
            seen.add(key)
            segments.append(f"{player} {canonical_line} {stat} Over")

    return " ".join(segments)


SCRAPER = BookScraper(
    name="parlayplay",
    domain="parlayplay.io",
    wait_selectors=(
        "[class*='player']",
        "[class*='pick']",
        "[class*='projection']",
    ),
    extra_wait_seconds=4.0,
    session_markers=SessionMarkers(
        login_wall=("log in", "sign up", "create account"),
        authenticated=("more", "less", "nba", "points", "rebounds", "assists"),
        min_authenticated_hits=4,
    ),
    prop_preprocess=preprocess,
    parser_regexes=(_PLAYER_HEADER_RE,),
)
