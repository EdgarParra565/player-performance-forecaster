"""PrizePicks scraper config and parser."""

from __future__ import annotations

import re

from nba_model.scrapers.base import (
    BookScraper,
    SessionMarkers,
    build_pp_style_name_pattern,
)


_NAME_PAT = build_pp_style_name_pattern()

# Current PrizePicks board format (observed 2026-05):
#   "127.7K NYK - C-F Karl-Anthony Towns @ PHI Fri 7:10pm 5.5 Assists More"
#   "72.5K MIN - C-F Naz Reid vs SAS Fri 9:40pm 6.5 Rebounds More"
# i.e. <popularity>K <TEAM> - <POS> <Player> @|vs <OPP> <Day> <Time> <Line> <Stat> <Side>
_BOARD_RE = re.compile(
    r"\d+(?:\.\d+)?K\s+"                          # popularity ("127.7K")
    r"[A-Z]{2,4}\s+-\s+"                          # team abbreviation + dash
    r"[A-Z]{1,3}(?:-[A-Z]{1,3})?\s+"              # position ("C-F", "G", "F-G")
    + _NAME_PAT
    + r"\s+(?:@|vs\.?)\s+[A-Z]{2,4}\s+"           # opponent
      r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"       # day-of-week
      r"\d{1,2}:\d{2}(?:am|pm|AM|PM)\s+"          # game time
      r"(?P<line>\d{1,3}(?:\.\d+)?)\s+"
      r"(?P<stat>[A-Z0-9][A-Za-z0-9 +\-\']{1,30}?)"
      r"\s+(?P<side>More|Less|Over|Under|Higher|Lower)",
)

# Legacy primary pattern: "Player TEAM Stat Line More|Less"
# Not IGNORECASE: case sensitivity needed to exclude all-uppercase tokens.
_PRIMARY_RE = re.compile(
    _NAME_PAT
    + r"\s+[A-Z]{2,4}\s+"
      r"(?P<stat>[A-Z][A-Za-z +\']{1,30}?)"
      r"\s+(?P<line>\d{1,3}(?:\.\d+)?)"
      r"\s+(?P<side>More|Less|Over|Under|Higher|Lower)",
)

# Alternate pattern: "Player Line Stat More|Less" (line before stat).
_ALT_RE = re.compile(
    _NAME_PAT
    + r"(?:\s+[A-Z]{2,5}(?:\s+(?:@|vs\.?)\s+[A-Z]{2,5})?)?"
      r"\s+(?P<line>\d{1,3}(?:\.\d+)?)"
      r"\s+(?P<stat>[A-Za-z][A-Za-z +\']{1,30}?)"
      r"\s+(?P<side>More|Less|Over|Under|Higher|Lower)",
)


def _clean_player(raw_name: str) -> str:
    """Strip trailing team-abbreviation tokens from an extracted name.

    The PP regexes sometimes capture trailing team codes (LAL, GSW, ...).
    Returns the empty string when the candidate isn't a plausible name.
    """
    name = " ".join(str(raw_name or "").split()).strip(" -:|,;")
    parts = [part for part in name.split(" ") if part]
    if len(parts) < 2:
        return ""
    while parts and re.match(r"^[A-Z]{2,4}$", parts[-1]):
        parts = parts[:-1]
    if len(parts) < 2:
        return ""
    return " ".join(parts)


def preprocess(text: str) -> str:
    """Strip PrizePicks team/game segments so generic patterns can match.

    Handles formats like:
      'Player TEAM Stat 27.5 More Less'
      'Player TEAM @ TEAM 27.5 Stat More Less'
    """
    segments: list[str] = []
    seen: set[tuple] = set()
    for pat in (_BOARD_RE, _PRIMARY_RE, _ALT_RE):
        for match in pat.finditer(text):
            player = _clean_player(match.group("player"))
            if not player:
                continue
            line = match.group("line").strip()
            stat = match.group("stat").strip()
            side = match.group("side").strip()
            key = (player.lower(), line, stat.lower())
            if key in seen:
                continue
            seen.add(key)
            segments.append(f"{player} {line} {stat} {side}")
    return " ".join(segments) if segments else ""


def extract_text_js(page) -> str:
    """Extract PrizePicks projection-card text via targeted JS DOM traversal.

    PrizePicks uses a React SPA with CSS-module class names that change
    between deploys. This queries a broad set of structural selectors and
    returns the deduplicated visible text of matching elements.
    """
    try:
        result = page.evaluate(
            """
            () => {
                const selectors = [
                    '[class*="Projection"]',
                    '[class*="projection"]',
                    '[class*="StatLine"]',
                    '[class*="stat-line"]',
                    '[class*="PlayerCard"]',
                    '[class*="player-card"]',
                    '[class*="Pick"]',
                    '[class*="pick"]',
                    '[class*="Board"]',
                    '[class*="board"]',
                    '[data-testid*="projection"]',
                    '[data-testid*="pick"]',
                    'article',
                ];
                const texts = [];
                const seen = new Set();
                for (const sel of selectors) {
                    try {
                        document.querySelectorAll(sel).forEach(el => {
                            const t = (el.innerText || '').trim().replace(/\\s+/g, ' ');
                            if (t.length > 8 && !seen.has(t)) {
                                seen.add(t);
                                texts.push(t);
                            }
                        });
                    } catch (e) {}
                }
                return texts.join(' ');
            }
            """
        )
        return " ".join(str(result or "").split())
    except Exception:
        return ""


SCRAPER = BookScraper(
    name="prizepicks",
    domain="prizepicks.com",
    wait_selectors=(
        # Broad class-name fragment matches — PrizePicks uses CSS modules.
        "[class*='Projection']",
        "[class*='projection']",
        "[class*='ProjectionCard']",
        "[class*='StatLine']",
        "[class*='stat-line']",
        "[class*='Pick']",
        "[class*='pick']",
        "[data-testid*='projection']",
        "[data-testid*='pick']",
        # Semantic fallbacks — PP wraps each card in an article or li.
        "article",
        "ul li",
    ),
    extra_wait_seconds=8.0,  # Extra time for React SPA lazy render.
    session_markers=SessionMarkers(
        login_wall=(
            "enter your phone number",
            "enter phone number",
            "log in to prizepicks",
            "create a new account",
            "sign up for prizepicks",
            "verify your identity",
        ),
        authenticated=(
            "more",
            "less",
            "projections",
            "nba",
            "points",
            "rebounds",
            "assists",
        ),
        min_authenticated_hits=4,
    ),
    js_extractor=extract_text_js,
    prop_preprocess=preprocess,
    parser_regexes=(_BOARD_RE, _PRIMARY_RE, _ALT_RE),
)
