"""Underdog Fantasy scraper config and parser."""

from __future__ import annotations

import re

from nba_model.scrapers.base import BookScraper, SessionMarkers


# Mixed-case word pattern: each word must contain at least one lowercase
# letter, so all-uppercase team abbreviations are excluded from name capture.
_NAME_PAT = (
    r"(?P<player>"
    r"[A-Z][A-Za-z.\'\-]*[a-z][A-Za-z.\'\-]*"
    r"(?:\s+[A-Z][A-Za-z.\'\-]*[a-z][A-Za-z.\'\-]*){1,3}"
    r")"
)

# Underdog UI shape: "Player TEAM @ TEAM - <gametime> 11.5 Rebounds Higher 1.06x"
_PROP_RE = re.compile(
    _NAME_PAT
    + r"\s+[A-Z]{2,4}\s+(?:@|vs\.?)\s+[A-Z]{2,4}"
      r"\s+-\s+.*?(?:EDT|EST|CDT|CST|PDT|PST|PM|AM)\s+"
      r"(?P<line>\d{1,3}(?:\.\d+)?)"
      r"\s+(?P<stat>[A-Za-z][A-Za-z +\']{1,30}?)"
      r"\s+(?P<side>Higher|Lower|Over|Under)",
    flags=re.IGNORECASE,
)


def preprocess(text: str) -> str:
    """Strip Underdog game-info segments so generic patterns can match.

    Converts 'Player TEAM @ TEAM - TIME 11.5 Rebounds Higher 1.06x'
    into     'Player 11.5 Rebounds Higher'
    """
    segments: list[str] = []
    for match in _PROP_RE.finditer(text):
        player = match.group("player").strip()
        line = match.group("line").strip()
        stat = match.group("stat").strip()
        side = match.group("side").strip()
        segments.append(f"{player} {line} {stat} {side}")
    return " ".join(segments) if segments else ""


SCRAPER = BookScraper(
    name="underdog",
    domain="underdogfantasy.com",
    # 2026 rebrand: app.underdogsports.com is now the primary host.
    aliases=("underdogsports.com",),
    wait_selectors=(
        "[class*='pick-em']",
        "[class*='player-pick']",
        "[class*='stat-line']",
        "[data-testid*='pick']",
        "[class*='higher-lower']",
    ),
    extra_wait_seconds=5.0,
    session_markers=SessionMarkers(
        login_wall=(
            "sign in to continue",
            "enter your email",
            "forgot password",
        ),
        authenticated=(
            "higher",
            "lower",
            "nba",
            "pick",
            "points",
        ),
        min_authenticated_hits=3,
    ),
    prop_preprocess=preprocess,
    parser_regexes=(_PROP_RE,),
)
