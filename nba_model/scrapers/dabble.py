"""Dabble (DFS pickem) scraper config (stub).

Australian-origin pickem app. Same Higher/Lower shape as PrizePicks.
"""

from __future__ import annotations

from nba_model.scrapers.base import BookScraper, SessionMarkers


SCRAPER = BookScraper(
    name="dabble",
    domain="dabblesports.com",
    aliases=("dabble.com",),
    wait_selectors=(
        "[class*='player-pick']",
        "[class*='projection']",
        "[class*='pick-card']",
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
