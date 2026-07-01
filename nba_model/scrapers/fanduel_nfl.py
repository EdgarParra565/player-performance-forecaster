"""FanDuel NFL scraper config (player props) — sport='nfl' (WS6 scaffolding).

Separate from the NBA FanDuel config (``fanduel.py``): same domain, different
sport, resolved via the per-(book, sport) registry. Player props first, using
the shared NFL prop preprocessor.

TODO(real-capture): authored against a representative fixture, not a live
authenticated FanDuel NFL snapshot — re-validate via the Chrome :9222 host.
"""

from __future__ import annotations

from nba_model.scrapers.base import BookScraper, SessionMarkers
from nba_model.scrapers.nfl_props import _ROW_RE, preprocess_nfl_props


SCRAPER = BookScraper(
    name="fanduel",
    domain="fanduel.com",
    sport="nfl",
    aliases=("sportsbook.fanduel.com",),
    wait_selectors=(
        "[class*='player-prop']",
        "[class*='selection']",
        "[data-testid*='market']",
    ),
    extra_wait_seconds=4.0,
    session_markers=SessionMarkers(
        login_wall=("log in", "sign up", "create account", "join now"),
        authenticated=(
            "passing yards",
            "rushing yards",
            "receiving yards",
            "receptions",
            "nfl",
            "over",
        ),
        min_authenticated_hits=4,
    ),
    prop_preprocess=preprocess_nfl_props,
    parser_regexes=(_ROW_RE,),  # presence signals "has player parser"
)
