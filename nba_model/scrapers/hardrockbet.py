"""Hard Rock Bet sportsbook scraper config (stub).

Hard Rock Hotel & Casino's sportsbook product. Format expected to be
similar to other US sportsbooks: spread / total / moneyline per game.
"""

from __future__ import annotations

from nba_model.scrapers.base import BookScraper, SessionMarkers


SCRAPER = BookScraper(
    name="hardrockbet",
    domain="hardrock.bet",
    aliases=("hardrockcasino.com", "app.hardrock.bet"),
    wait_selectors=(
        "[class*='market']",
        "[class*='outcome']",
        "[class*='selection']",
        "[data-testid*='market']",
    ),
    extra_wait_seconds=4.0,
    session_markers=SessionMarkers(
        login_wall=("log in", "sign up", "create account"),
        authenticated=("spread", "total", "moneyline", "nba", "today"),
        min_authenticated_hits=4,
    ),
)
