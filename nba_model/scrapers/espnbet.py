"""ESPN BET sportsbook scraper config (stub).

Penn-owned sportsbook with ESPN branding. Format expected to be similar
to BetMGM / Caesars: spread / total / moneyline per game.
"""

from __future__ import annotations

from nba_model.scrapers.base import BookScraper, SessionMarkers


SCRAPER = BookScraper(
    name="espnbet",
    domain="espnbet.com",
    wait_selectors=(
        "[class*='market']",
        "[class*='outcome']",
        "[class*='option']",
        "[data-testid*='market']",
    ),
    extra_wait_seconds=4.0,
    session_markers=SessionMarkers(
        login_wall=("log in", "sign up", "create account", "join"),
        authenticated=("spread", "total", "moneyline", "nba", "points", "rebounds"),
        min_authenticated_hits=4,
    ),
)
