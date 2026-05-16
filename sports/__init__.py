"""Multi-sport registry.

This package is the seam between the sport-agnostic infrastructure
(distribution fitting, Monte Carlo, EV math, Stripe, Streamlit shell,
auth, subscriptions, input validation) and the sport-specific configuration
(stat types, team codes, plausible-line ranges, scrapers, season formats).

Status as of 2026-05-16:
- **nba**:    fully implemented; the original product.
- **nfl, mlb, nhl, soccer**: stubs only.  Their `Sport` config is in place
  so the UI can list them in the sport-selector and so future PRs know
  exactly where to plug in real data ingestion + scrapers.

How to add a new sport:
  1. Drop a module in `nba_model/sports/<sport>.py` exporting `SPORT`.
  2. List the new module here in `SPORTS` so it shows in the registry.
  3. Wire ingestion: a thin shim that pulls game-by-game stats into the
     existing `game_logs` table (or a sport-specific table if the schema
     diverges).
  4. Wire scrapers: per-book modules under `nba_model/scrapers/<book>/`
     keyed by `Sport.key` so PrizePicks (NBA) and PrizePicks (NFL) are
     separate files.

See `docs/MULTI_SPORT_PLAN.md` for the full rollout plan, per-sport open
questions, and where each piece slots into existing modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Sport:
    """Sport-specific configuration the agnostic layers consume.

    Anything that varies between sports goes here. Anything that doesn't
    (distribution fitting, EV math, the Stripe layer, etc.) stays in the
    agnostic packages and reads `Sport` values when it needs to.

    Required fields:
        key:           short slug used in URLs, env vars, DB tags ("nba")
        display_name:  human-friendly label for the UI ("NBA")
        status:        "live" | "stub" | "planned"
        stat_types:    canonical stat names for player props
        stat_line_ranges: per-stat plausible (lo, hi) range for the
                       data-poisoning validator in `input_validation`
        team_codes:    canonical short codes (e.g. NBA: LAL, NFL: KC,
                       MLB: NYY, NHL: BOS, soccer: ARS).
        season_format: human-readable description of the season format
        primary_books: list of books we expect lines for in this sport
        notes:         freeform planning notes (open questions, blockers)
    """
    key: str
    display_name: str
    status: str
    stat_types: tuple[str, ...] = ()
    stat_line_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    team_codes: tuple[str, ...] = ()
    season_format: str = ""
    primary_books: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    sub_leagues: tuple["Sport", ...] = ()  # only used by soccer

    @property
    def is_live(self) -> bool:
        return self.status == "live"


# Each module exports its own SPORT object. Importing here so call sites
# only need `from sports import SPORTS, get_sport`.
from sports import nba as _nba
from sports import nfl as _nfl
from sports import mlb as _mlb
from sports import nhl as _nhl
from sports import soccer as _soccer


SPORTS: tuple[Sport, ...] = (
    _nba.SPORT,
    _nfl.SPORT,
    _mlb.SPORT,
    _nhl.SPORT,
    _soccer.SPORT,
)

SPORT_BY_KEY: dict[str, Sport] = {s.key: s for s in SPORTS}


def get_sport(key: str) -> Sport:
    """Look up a sport by its short key. Defaults to NBA if unknown."""
    return SPORT_BY_KEY.get((key or "").strip().lower(), _nba.SPORT)


def live_sports() -> list[Sport]:
    return [s for s in SPORTS if s.is_live]


def stub_sports() -> list[Sport]:
    return [s for s in SPORTS if s.status == "stub"]
