"""Fliff scraper config (stub).

Fliff is a sweepstakes/social-pickem book whose prop UI is similar to
PrizePicks (More/Less). Fetch path is wired up; parser is left for later
once an authenticated text sample is captured.
"""

from __future__ import annotations

from nba_model.scrapers.base import BookScraper, SessionMarkers


SCRAPER = BookScraper(
    name="fliff",
    domain="fliff.com",
    wait_selectors=(
        "[class*='player-prop']",
        "[class*='pick']",
        "[class*='prop']",
        "[data-testid*='prop']",
        "[data-testid*='pick']",
        "article",
    ),
    extra_wait_seconds=4.0,
    session_markers=SessionMarkers(
        login_wall=(
            "log in",
            "sign up",
            "create account",
            "verify your phone",
        ),
        authenticated=(
            "more",
            "less",
            "nba",
            "points",
            "rebounds",
            "assists",
        ),
        min_authenticated_hits=4,
    ),
)
