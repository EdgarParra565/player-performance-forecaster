"""FanDuel MLB scraper config (player props) — sport='mlb'.

Separate from the NBA FanDuel config (``fanduel.py``): same domain, different
sport, resolved via the per-(book, sport) registry. Player props first, using
the shared MLB prop preprocessor (hitter + pitcher markets, over/under +
yes/no shapes).

Parsed lines are joined to games/players via ``mlb_results_ingestion`` (MLB
Stats API personId + schedule).

TEAM LINES: VALIDATED against a LIVE FanDuel MLB capture (2026-06-28, CDP) —
``team_line_extractor`` (shared ``mlb_team_lines.extract_team_lines``) parsed
18 games / 168 rows into ``web_team_lines`` (sport='mlb'); see
``test_mlb_team_lines.py`` (fixture is a real captured slice).

TODO(real-capture): the player-prop preprocessor (``preprocess_mlb_props``) is
still fixture-based — sportsbook player props live on per-game pages, not the
league lobby captured here; re-validate it against a live prop-page snapshot.
"""

from __future__ import annotations

from nba_model.scrapers.base import BookScraper, SessionMarkers
from nba_model.scrapers.mlb_props import preprocess_mlb_props
from nba_model.scrapers.mlb_team_lines import extract_team_lines


SCRAPER = BookScraper(
    name="fanduel",
    domain="fanduel.com",
    sport="mlb",
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
