"""FanDuel scraper config (sportsbook + DFS).

The DFS site (``www.fanduel.com``) lists tournaments only — no player or
team prop lines.  The sportsbook lives on a different host
(``sportsbook.fanduel.com``).  We register both via ``aliases`` so either
tab is picked up by the registry.

Sportsbook page format (basketball lobby) is similar to DraftKings:

    "NY Knicks @ PHI 76ers Today 7:10pm +1.5 -108 O 213.5 -110 +102 "
    "-1.5 -112 U 213.5 -110 -122"

Parser is added once an authenticated FanDuel sportsbook snapshot is on
disk.  For now the scraper has no team-line extractor — fetch path runs
and stores snapshots, but parsing is a follow-up.
"""

from __future__ import annotations

from nba_model.scrapers.base import BookScraper, SessionMarkers


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
)
