"""Fanatics Sportsbook scraper config (stub).

Traditional sportsbook with American odds. Format expected to be similar
to BetMGM / Caesars: spread / total / moneyline per game.
"""

from __future__ import annotations

from nba_model.scrapers.base import BookScraper, SessionMarkers


SCRAPER = BookScraper(
    name="fanatics",
    domain="fanatics.com",
    aliases=("sportsbook.fanatics.com", "betfanatics.com"),
    wait_selectors=(
        "[class*='market']",
        "[class*='outcome']",
        "[class*='selection']",
        "[data-testid*='market']",
    ),
    extra_wait_seconds=4.0,
    session_markers=SessionMarkers(
        login_wall=("log in", "sign up", "create account"),
        authenticated=("spread", "total", "moneyline", "nba", "today", "tomorrow"),
        min_authenticated_hits=4,
    ),
)
