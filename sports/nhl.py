"""NHL sport config.

Status: STUB.

Next steps:
  1. Ingest: `nhl-api-py` (community wrapper of the official NHL API) is
     the path of least resistance. Alternative: `hockey_scraper` for
     play-by-play.
  2. Schema: `sport='nhl'` extension to `game_logs`. Hockey box-score
     fits the basketball shape better than baseball does (skaters all
     score / assist / shoot, like NBA players score / assist / rebound).
  3. NHL-specific quirks:
     - Goalies vs skaters are separate populations (like MLB pitcher / batter)
       but the gap is less extreme - most goalie markets are wins / saves /
       goals-against.
     - Shootouts: do they count toward stats? Books treat them differently
       across markets, codify the rule once.
     - Plus/minus is a notoriously noisy stat, low priority.
"""

from sports import Sport

SPORT = Sport(
    key="nhl",
    display_name="NHL",
    status="stub",
    stat_types=(
        # Skater
        "goals", "assists_nhl", "points_nhl", "shots_on_goal",
        "anytime_goal", "first_goal", "blocked_shots", "hits",
        "power_play_points",
        # Goalie
        "saves", "goals_against", "shots_faced", "goalie_win",
    ),
    stat_line_ranges={
        "goals":              (0.0,  5.0),
        "assists_nhl":        (0.0,  5.0),
        "points_nhl":         (0.0,  6.0),  # goals + assists
        "shots_on_goal":      (0.0, 12.0),
        "blocked_shots":      (0.0,  8.0),
        "hits":               (0.0, 12.0),
        "power_play_points":  (0.0,  4.0),
        "saves":              (0.0, 55.0),
        "goals_against":      (0.0,  8.0),
        "shots_faced":        (0.0, 60.0),
    },
    team_codes=(
        "ANA", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ",
        "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH",
        "NJD", "NYI", "NYR", "OTT", "PHI", "PIT", "SJS", "SEA",
        "STL", "TBL", "TOR", "VAN", "VGK", "WSH", "WPG", "UTA",
    ),
    season_format="YYYY-YY (e.g. 2025-26 = Oct 2025 – Jun 2026)",
    primary_books=(
        "fanduel", "draftkings", "betmgm", "caesars", "betrivers",
        "fanatics", "prizepicks", "underdog",
    ),
    notes=(
        "Use nhl-api-py for ingest.",
        "Goalies are a separate stat-type list -- keep them out of the skater "
        "browse view by default.",
        "Decide once how shootout goals count across markets; books differ.",
        "anytime_goal / first_goal are yes/no -- Bernoulli not normal.",
    ),
)
