"""Shared types and utilities for per-book scrapers.

Each book lives in its own module (`prizepicks.py`, `underdog.py`, ...) and
exports a ``BookScraper`` instance describing:

  * the domain it matches (e.g. ``prizepicks.com``)
  * Playwright wait selectors and extra wait time used by the fetcher
  * session-state markers used to validate authenticated fetches
  * an optional JS extractor used when generic HTML/innerText extraction is weak
  * an optional preprocessor that converts the raw page text into a normalized
    ``"Player Line Stat Side"`` form so the generic ``_CARD_PATTERNS`` match it

Books with no parser (`prop_preprocess=None`, `parser_regexes=()`) act as
config-only stubs: their fetch path works (so we can collect snapshots once a
session is set up) but no prop cards are extracted yet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

# A preprocessor converts free-form page text into a sequence of normalized
# "Player Line Stat Side" segments separated by spaces, which the generic
# parser regexes in browser_prop_parser._CARD_PATTERNS can then match.
PreprocessFn = Callable[[str], str]

# A team-line extractor returns a list of dicts ready for
# DatabaseManager.insert_web_team_lines (minus snapshot_id / source_url /
# observed_at_utc / book / parser_version / record_sha256, which the
# orchestrator fills in).  Each dict carries:
#   away_team, home_team, market_type ('spread'|'total'|'moneyline'|'team_total'),
#   side ('home'|'away'|'over'|'under'), team (optional),
#   line_value (float|None), odds_american (int|None), raw_text (str)
TeamLineExtractor = Callable[[str], list]

# A page-context JS extractor returns supplemental visible text from the page
# DOM via page.evaluate.  Used when the generic HTML / innerText extraction
# misses content (e.g. CSS-module React SPAs).
JSExtractorFn = Callable[..., str]


@dataclass(frozen=True)
class SessionMarkers:
    """Substrings used to classify a page as authenticated or login-walled."""

    login_wall: tuple[str, ...] = ()
    authenticated: tuple[str, ...] = ()
    min_authenticated_hits: int = 3


@dataclass(frozen=True)
class BookScraper:
    """Per-book scraper configuration and parsing hooks.

    A book is the unit of customization: domain matching, page-load behaviour,
    auth detection, and prop-card extraction all key off this object.
    """

    name: str
    domain: str
    wait_selectors: tuple[str, ...] = ()
    extra_wait_seconds: float = 0.0
    session_markers: SessionMarkers = field(default_factory=SessionMarkers)
    js_extractor: Optional[JSExtractorFn] = None
    prop_preprocess: Optional[PreprocessFn] = None
    parser_regexes: tuple[re.Pattern, ...] = ()
    # Optional game-level extractor returning fully-formed team-line dicts.
    team_line_extractor: Optional[TeamLineExtractor] = None
    # Extra domains that resolve to this book (rebrands, regional variants).
    aliases: tuple[str, ...] = ()

    def matches_host(self, host: str) -> bool:
        """Return True when ``host`` equals or is a subdomain of any known domain."""
        host = (host or "").lower()
        for candidate in (self.domain,) + self.aliases:
            if host == candidate or host.endswith(f".{candidate}"):
                return True
        return False


# ----- Shared regex building blocks used by multiple books -----------------

# Words that should never appear at the start of a player-name token.  Shared
# across PP/UD pre-processors and the generic _CARD_PATTERNS in
# browser_prop_parser so UI labels don't get captured as player names.
NAME_STOP_WORDS: frozenset[str] = frozenset(
    {
        "and",
        "nba",
        "nfl",
        "mlb",
        "pick",
        "pickem",
        "higher",
        "lower",
        "over",
        "under",
        "more",
        "less",
        "loading",
        "login",
        "signin",
        "sign",
        "edt",
        "est",
        "cdt",
        "cst",
        "pdt",
        "pst",
        "popular",
        "featured",
        "champions",
        "drafts",
        "live",
        "results",
        "rankings",
        "mobile",
        "web",
        "coming",
        "soon",
        "download",
        "app",
        "standard",
        "flex",
        "play",
        "boost",
        # sportsbook UI labels that appear before prop cards
        "projections",
        "projection",
        "board",
        "slate",
        "lineup",
        "entry",
    }
)


def build_pp_style_name_pattern() -> str:
    """Build a case-sensitive name pattern for PP/UD-style "Mixed Case" tokens.

    Each word must contain at least one lowercase letter (excludes all-caps
    team abbreviations).  A negative lookahead prevents stop words like
    "More" / "Less" / "Projections" from starting a name token, even though
    they technically have lowercase letters.  Compiled WITHOUT IGNORECASE so
    [a-z] is genuinely case-sensitive.
    """
    stop_lookahead = (
        "(?!"
        + "|".join(
            re.escape(word[0].upper() + word[1:]) + r"\b"
            for word in sorted(NAME_STOP_WORDS, key=len, reverse=True)
            if word
        )
        + ")"
    )
    name_word = stop_lookahead + r"[A-Z][A-Za-z.\'\-]*[a-z][A-Za-z.\'\-]*"
    return r"(?P<player>" + name_word + r"(?:\s+" + name_word + r"){1,3}" + r")"
