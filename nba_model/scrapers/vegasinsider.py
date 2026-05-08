"""VegasInsider aggregator scraper config (stub).

Public aggregator with cross-book NBA odds tables. No login required.
"""

from __future__ import annotations

from nba_model.scrapers.base import BookScraper, SessionMarkers


SCRAPER = BookScraper(
    name="vegasinsider",
    domain="vegasinsider.com",
    wait_selectors=(
        "[class*='odds']",
        "[class*='matchup']",
        "[class*='line']",
        "table",
    ),
    extra_wait_seconds=2.0,
    session_markers=SessionMarkers(
        login_wall=(),
        authenticated=("nba", "spread", "total", "moneyline", "odds"),
        min_authenticated_hits=2,
    ),
)
