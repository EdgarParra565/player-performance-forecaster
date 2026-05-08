"""Sleeper Picks (DFS pickem) scraper config (stub).

Sleeper's pickem product lives at sleeper.com/picks. Same Higher/Lower
shape as PrizePicks. Parser added once an authenticated NBA snapshot is
captured.
"""

from __future__ import annotations

from nba_model.scrapers.base import BookScraper, SessionMarkers


SCRAPER = BookScraper(
    name="sleeper",
    domain="sleeper.com",
    wait_selectors=(
        "[class*='player-pick']",
        "[class*='pick-card']",
        "[class*='projection']",
        "[data-testid*='pick']",
    ),
    extra_wait_seconds=4.0,
    session_markers=SessionMarkers(
        login_wall=(
            "log in",
            "sign up",
            "create account",
        ),
        authenticated=(
            "more",
            "less",
            "higher",
            "lower",
            "nba",
            "points",
        ),
        min_authenticated_hits=4,
    ),
)
