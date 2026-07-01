"""MLB sport config.

Status: BETA — MLB is the first non-NBA sport with a real data layer.

Implemented (data/scrapers layer):
  1. Ingest: the OFFICIAL MLB STATS API (https://statsapi.mlb.com/api/v1) is
     the primary baseline-data source — free, no key, no auth (the MLB analog
     to nba_api). See `nba_model/data/mlb_results_ingestion.py`
     (schedule -> boxscore -> per-player game logs). `pybaseball` is kept as an
     OPTIONAL, guarded Statcast/park-factor supplement only.
  2. Schema: MLB box scores diverge too much from the NBA-shaped `game_logs`,
     so MLB lives in a dedicated long-format `mlb_game_logs` table (one row per
     player/game/stat, `value` tagged by stat_type, `sport='mlb'`). This keeps
     MLB rows out of every NBA query by construction.
  3. MLB-specific realities honored:
     - Pitcher vs batter are separate populations + markets — see
       `scrapers/mlb_props.stat_group()` ('hitting' | 'pitching' | 'combined').
     - Park factors: `nba_model/data/mlb_park_factors.py` (light lookup, TODO
       to validate against real data).
  4. Lines come in BOTH yes/no (hit a HR -> anytime_home_run) and over/under
     (over 1.5 hits) forms — `preprocess_mlb_props` handles both.

Scrapers (props first): `scrapers/draftkings_mlb.py`, `scrapers/fanduel_mlb.py`
(sport='mlb'), resolvable via `get_scraper_for_book_sport(book, 'mlb')`.

Remaining for fully-"live" status: a sport-filtered Streamlit rendering path
(the web view currently reads the NBA-shaped tables only) — that's web-layer
work. MLB is intentionally kept OUT of the NBA cross-book consensus path.
"""

from sports import Sport

SPORT = Sport(
    key="mlb",
    display_name="MLB",
    status="beta",
    stat_types=(
        # Hitter
        "hits", "total_bases", "home_runs", "rbis", "runs_scored",
        "stolen_bases", "walks", "strikeouts_batter", "singles",
        # Pitcher
        "strikeouts_pitcher", "earned_runs", "outs_recorded",
        "hits_allowed", "walks_allowed", "wins", "pitcher_record",
        # Combined / anytime
        "anytime_home_run", "first_run_scorer",
    ),
    stat_line_ranges={
        "hits":                 (0.0,   6.0),
        "total_bases":          (0.0,  12.0),
        "home_runs":            (0.0,   4.0),
        "rbis":                 (0.0,   8.0),
        "runs_scored":          (0.0,   5.0),
        "stolen_bases":         (0.0,   4.0),
        "walks":                (0.0,   5.0),
        "strikeouts_batter":    (0.0,   5.0),
        "singles":              (0.0,   5.0),
        "strikeouts_pitcher":   (0.0,  18.0),
        "earned_runs":          (0.0,  10.0),
        "outs_recorded":        (0.0,  27.0),
        "hits_allowed":         (0.0,  15.0),
        "walks_allowed":        (0.0,   8.0),
    },
    team_codes=(
        "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE",
        "COL", "DET", "HOU", "KC",  "LAA", "LAD", "MIA", "MIL",
        "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SD",  "SF",
        "SEA", "STL", "TB",  "TEX", "TOR", "WSH",
    ),
    season_format="YYYY (single calendar year, regular season Mar-Sep + playoffs)",
    primary_books=(
        "fanduel", "draftkings", "betmgm", "caesars", "betrivers",
        "fanatics", "prizepicks", "underdog",
    ),
    notes=(
        "Primary ingest: official MLB Stats API (statsapi.mlb.com); pybaseball "
        "is an optional, guarded Statcast/park-factor supplement only.",
        "Pitcher vs batter need separate stat-type lists + separate models.",
        "Park-factor adjustment is the MLB analog to NBA defense rating.",
        "Some markets are yes/no (anytime HR) -- Bernoulli, not normal.",
        "Right-handed vs left-handed split is a meaningful feature for hitters.",
    ),
)
