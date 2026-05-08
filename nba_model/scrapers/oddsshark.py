"""OddsShark aggregator scraper config (stub).

Public aggregator that publishes consensus NBA odds from multiple books.
No login required, but Cloudflare may block headless traffic. Use the
CDP path (real Chrome) if direct fetch fails.

Note: OddsShark consensus is one step removed from raw book lines. Useful
as a fallback when individual books aren't accessible.
"""

from __future__ import annotations

from nba_model.scrapers.base import BookScraper, SessionMarkers


SCRAPER = BookScraper(
    name="oddsshark",
    domain="oddsshark.com",
    wait_selectors=(
        "[class*='matchup']",
        "[class*='odds']",
        "[class*='consensus']",
        "table",
    ),
    extra_wait_seconds=2.0,
    session_markers=SessionMarkers(
        login_wall=(),  # public site
        authenticated=("nba", "spread", "total", "moneyline", "consensus"),
        min_authenticated_hits=2,
    ),
)
