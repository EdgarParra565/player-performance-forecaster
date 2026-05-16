"""NFL sport config.

Status: STUB.  Stat type list + line ranges + team codes are filled in so
the UI can show "NFL (coming soon)" in the sport-selector and so the future
ingestion + scraper PRs know exactly what schema to target.

Next steps to make NFL live (rough order):
  1. Pick an ingest source.  `nfl_data_py` is the obvious choice
     (`pip install nfl_data_py`); it wraps `nflverse-data` Parquet drops
     and is the equivalent of `nba_api` for the NFL world.
  2. Schema decision: extend `game_logs` with a `sport` column, or split
     into per-sport tables.  Recommendation: add `sport TEXT NOT NULL
     DEFAULT 'nba'` + a composite key change.  Cleaner than N parallel
     schemas.
  3. Wire per-book NFL scrapers under `nba_model/scrapers/<book>/nfl.py`.
     Most books split their player-prop pages by sport, so the existing
     `BookScraper` registry just needs sport-keyed entries.
  4. Three big NFL-specific quirks to watch for:
     - Weekly cadence (vs nightly) - rolling-mean window math needs
       different defaults (5-game rolling is very different in NFL).
     - Position-dependent stats (QB vs RB vs WR project entirely
       differently - one model won't fit all positions).
     - Bye weeks - the game-by-game series isn't continuous.
"""

from sports import Sport

SPORT = Sport(
    key="nfl",
    display_name="NFL",
    status="stub",
    stat_types=(
        # QB
        "passing_yards", "passing_touchdowns", "passing_completions",
        "passing_attempts", "interceptions",
        # RB / WR / TE
        "rushing_yards", "rushing_attempts", "rushing_touchdowns",
        "receiving_yards", "receptions", "receiving_touchdowns",
        # Combined / anytime markets
        "anytime_touchdown_scorer", "longest_reception", "longest_rush",
        # Kicker
        "kicking_points", "field_goals_made",
    ),
    stat_line_ranges={
        "passing_yards":          (0.0, 600.0),
        "passing_touchdowns":     (0.0,   8.0),
        "passing_completions":    (0.0,  60.0),
        "passing_attempts":       (0.0,  80.0),
        "interceptions":          (0.0,   8.0),
        "rushing_yards":          (0.0, 350.0),
        "rushing_attempts":       (0.0,  45.0),
        "rushing_touchdowns":     (0.0,   5.0),
        "receiving_yards":        (0.0, 300.0),
        "receptions":             (0.0,  25.0),
        "receiving_touchdowns":   (0.0,   5.0),
        "longest_reception":      (0.0, 100.0),
        "longest_rush":           (0.0, 100.0),
        "kicking_points":         (0.0,  25.0),
        "field_goals_made":       (0.0,   8.0),
    },
    team_codes=(
        "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
        "DAL", "DEN", "DET", "GB",  "HOU", "IND", "JAX", "KC",
        "LV",  "LAC", "LAR", "MIA", "MIN", "NE",  "NO",  "NYG",
        "NYJ", "PHI", "PIT", "SEA", "SF",  "TB",  "TEN", "WAS",
    ),
    season_format="YYYY (single calendar year, e.g. 2026 = the 2026-27 season)",
    primary_books=(
        "fanduel", "draftkings", "betmgm", "caesars", "betrivers",
        "fanatics", "prizepicks", "underdog", "pick6",
    ),
    notes=(
        "Use nfl_data_py for ingest (wraps nflverse Parquet).",
        "Weekly cadence -- rolling-mean defaults differ from NBA.",
        "Stats are position-dependent; one model won't fit QB + WR + K.",
        "Add bye-week filter so the rolling mean isn't biased.",
        "anytime_touchdown_scorer is a yes/no market -- Bernoulli, not normal.",
    ),
)
