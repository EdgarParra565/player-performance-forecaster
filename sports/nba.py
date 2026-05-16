"""NBA sport config.

Reflects what's already implemented across the codebase - this module is the
single source of truth so future refactors can stop reading hardcoded NBA
constants from `input_validation.py`, `player_charts.py`, etc.
"""

from sports import Sport

SPORT = Sport(
    key="nba",
    display_name="NBA",
    status="live",
    stat_types=(
        "points",
        "assists",
        "rebounds",
        "pra",
        "three_pointers_made",
        "field_goals_made",
        "minutes",
    ),
    stat_line_ranges={
        # Mirrors `nba_model.web.input_validation.STAT_LINE_RANGES`.
        "points":              (0.0, 100.0),
        "assists":             (0.0,  35.0),
        "rebounds":            (0.0,  40.0),
        "pra":                 (0.0, 150.0),
        "ra":                  (0.0,  60.0),
        "three_pointers_made": (0.0,  20.0),
        "field_goals_made":    (0.0,  35.0),
        "minutes":             (0.0,  60.0),
    },
    team_codes=(
        "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN",
        "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA",
        "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX",
        "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
    ),
    season_format="YYYY-YY (e.g. 2025-26 = Oct 2025 – Jun 2026)",
    primary_books=(
        "fanduel", "draftkings", "betmgm", "caesars", "betrivers",
        "bovada", "betonline.ag", "fanatics", "prizepicks", "underdog",
    ),
)
