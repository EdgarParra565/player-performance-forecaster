"""DraftKings MLB scraper config (player props) — sport='mlb'.

Separate from the NBA DraftKings config (``draftkings.py``): same domain,
different sport, resolved via the per-(book, sport) registry. Player props
first, using the shared MLB prop preprocessor (handles hitter + pitcher
markets and both over/under and yes/no shapes).

Parsed lines are joined to games/players via ``mlb_results_ingestion`` (MLB
Stats API personId + schedule).

TEAM LINES: uses the same shared ``mlb_team_lines.extract_team_lines`` that was
validated against a live FanDuel MLB capture, so the parser code is proven.
DK live validation is still BLOCKED: on 2026-06-28 the DK MLB lobby fetch
returned only the category-nav shell (the game grid is lazy-loaded JS that the
snapshot didn't capture) — needs an interactive scroll / longer wait or a
per-game URL. Not an auth or parser problem.

TODO(real-capture): player-prop preprocessor still fixture-based (props live on
per-game pages, not the lobby).
"""

from __future__ import annotations

from nba_model.scrapers.base import BookScraper, SessionMarkers
from nba_model.scrapers.mlb_props import preprocess_mlb_props
from nba_model.scrapers.mlb_team_lines import extract_team_lines


SCRAPER = BookScraper(
    name="draftkings",
    domain="draftkings.com",
    sport="mlb",
    wait_selectors=(
        "[class*='sportsbook-outcome']",
        "[class*='player-prop']",
        "[data-testid*='outcome']",
    ),
    extra_wait_seconds=4.0,
    session_markers=SessionMarkers(
        login_wall=("sign up", "log in", "create account", "verify identity"),
        authenticated=(
            "total bases",
            "strikeouts",
            "hits",
            "home run",
            "mlb",
            "over",
        ),
        min_authenticated_hits=4,
    ),
    prop_preprocess=preprocess_mlb_props,
    team_line_extractor=extract_team_lines,
)
