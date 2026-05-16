"""MLB sport config.

Status: STUB.

Next steps:
  1. Ingest: `pybaseball` is the standard wrapper for FanGraphs + Statcast.
     `pybaseball.statcast_batter`, `statcast_pitcher`, `schedule_and_record`
     cover most of what we need.
  2. Schema: same `sport='mlb'` extension to `game_logs`. MLB's box-score
     diverges enough (PA / AB / BB / SO / HR vs basketball-shaped stats)
     that a dedicated `mlb_batter_logs` + `mlb_pitcher_logs` pair is
     probably cleaner long-term, but for an MVP we can squeeze key stats
     into a generic `value` column tagged by stat_type.
  3. Two BIG MLB-specific issues:
     - Pitcher vs batter are entirely separate populations + markets.
       Two distinct stat-type lists, two distinct rolling-window contexts.
     - Park factors matter a LOT (Coors Field is +12% runs, Petco is -8%).
       We'd want a `park_factor` adjustment analogous to NBA's defense
       adjustment.
  4. Lines come in BOTH yes/no (hit a HR) and over/under (over 1.5 hits)
     forms. The probe panel needs both modes.
"""

from sports import Sport

SPORT = Sport(
    key="mlb",
    display_name="MLB",
    status="stub",
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
        "Use pybaseball for ingest (Statcast + FanGraphs wrapper).",
        "Pitcher vs batter need separate stat-type lists + separate models.",
        "Park-factor adjustment is the MLB analog to NBA defense rating.",
        "Some markets are yes/no (anytime HR) -- Bernoulli, not normal.",
        "Right-handed vs left-handed split is a meaningful feature for hitters.",
    ),
)
