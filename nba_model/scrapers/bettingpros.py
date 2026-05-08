"""BettingPros aggregator scraper config (stub).

Free-tier site with NBA player prop projections plus consensus lines from
multiple sportsbooks. Useful complement to direct book scraping when
sessions for individual books aren't available.
"""

from __future__ import annotations

from nba_model.scrapers.base import BookScraper, SessionMarkers


SCRAPER = BookScraper(
    name="bettingpros",
    domain="bettingpros.com",
    wait_selectors=(
        "[class*='player']",
        "[class*='prop']",
        "[class*='projection']",
        "[class*='line']",
    ),
    extra_wait_seconds=3.0,
    session_markers=SessionMarkers(
        login_wall=(),
        authenticated=("nba", "points", "rebounds", "assists", "projection", "consensus"),
        min_authenticated_hits=3,
    ),
)
