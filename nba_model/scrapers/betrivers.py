"""BetRivers sportsbook scraper config (stub).

Traditional sportsbook with American odds. Parser hooks omitted; add once
authenticated sample text is available.
"""

from __future__ import annotations

from nba_model.scrapers.base import BookScraper, SessionMarkers


SCRAPER = BookScraper(
    name="betrivers",
    domain="betrivers.com",
    wait_selectors=(
        "[class*='market']",
        "[class*='outcome']",
        "[class*='selection']",
        "[data-testid*='odd']",
    ),
    extra_wait_seconds=4.0,
    session_markers=SessionMarkers(
        login_wall=(
            "log in",
            "sign up",
            "register",
            "create account",
        ),
        authenticated=(
            "over",
            "under",
            "points",
            "rebounds",
            "assists",
            "nba",
        ),
        min_authenticated_hits=4,
    ),
)
