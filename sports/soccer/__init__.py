"""Soccer sport config.

Status: STUB.

Soccer is structurally different from NBA / NFL / MLB / NHL: there's no
single league.  We model it as a parent `Sport` (`key="soccer"`) holding a
tuple of sub-league `Sport` objects so the UI can show a two-level picker
(`Soccer -> Premier League`).

Rollout order (per the user's priority list):
  1. Premier League ("epl")
  2. La Liga ("laliga")
  3. Serie A ("seriea")
  4. Bundesliga ("bundesliga")
  5. Ligue 1 ("ligue1")
  6. UEFA Champions League ("ucl")
  7. Future: Copa Libertadores, Copa Sudamericana

Cross-cutting soccer notes:
  - Markets are three-way (home win / draw / away win) instead of NBA's
    two-way; the EV math needs to handle it. `nba_model/web/parlay_compare`
    assumes binary outcomes - will need extending.
  - Player-prop markets exist (anytime goal, shots on target, cards,
    corners) but are far thinner than NBA prop markets at most US books.
    Pinnacle / Bet365 are the deeper sources, neither US-friendly.
  - Cup competitions (UCL, Copa Lib) have two-legged ties + aggregate
    scoring + away-goals rules historically.  The schema needs to model
    "tie" as a higher-level entity than "match".
  - Season formats split: European leagues are Aug-May ("2025-26"),
    Copa Libertadores is calendar-year ("2026").
  - Promotion / relegation churns team membership annually; storing
    league-team mapping by season is essential (NBA / NFL / MLB / NHL
    don't have this problem).
"""

from sports import Sport


def _common_player_stats() -> tuple[str, ...]:
    # Shared across all top-flight soccer leagues. Cup competitions add
    # nothing distinct.
    return (
        "shots", "shots_on_target", "anytime_goalscorer",
        "first_goalscorer", "last_goalscorer", "to_score_2_or_more",
        "assists_soccer", "fouls_committed", "tackles", "passes_completed",
        "yellow_cards", "red_cards", "saves_keeper",
    )


def _common_line_ranges() -> dict[str, tuple[float, float]]:
    return {
        "shots":              (0.0,  10.0),
        "shots_on_target":    (0.0,   6.0),
        "assists_soccer":     (0.0,   4.0),
        "fouls_committed":    (0.0,   8.0),
        "tackles":            (0.0,  10.0),
        "passes_completed":   (0.0, 200.0),
        "saves_keeper":       (0.0,  15.0),
    }


EPL = Sport(
    key="epl",
    display_name="Premier League",
    status="stub",
    stat_types=_common_player_stats(),
    stat_line_ranges=_common_line_ranges(),
    team_codes=(
        "ARS", "AVL", "BOU", "BRE", "BHA", "CHE", "CRY", "EVE",
        "FUL", "LIV", "MCI", "MUN", "NEW", "NFO", "TOT", "WHU",
        "WOL",  # promotion/relegation will rotate the rest.
    ),
    season_format="YYYY-YY (Aug YYYY – May YYYY+1)",
    primary_books=(
        "fanduel", "draftkings", "betmgm", "caesars", "betrivers",
        "fanatics", "bet365",  # bet365 is the depth source globally.
    ),
    notes=(
        "Promotion/relegation: team_codes is a current-season snapshot.",
        "Cup overlap: same player can have EPL + FA Cup + UCL lines on the "
        "same gameweek; markets are distinct.",
    ),
)

LALIGA = Sport(
    key="laliga",
    display_name="La Liga",
    status="stub",
    stat_types=_common_player_stats(),
    stat_line_ranges=_common_line_ranges(),
    team_codes=(
        "RMA", "BAR", "ATM", "SEV", "VAL", "VLL", "VIL", "RBE", "RSO",
        "ATH", "GET", "ESP", "MLL", "ALA", "RAY", "OSA", "CEL", "GIR",
        # promotion/relegation snapshot
    ),
    season_format="YYYY-YY (Aug YYYY – May YYYY+1)",
    primary_books=("fanduel", "draftkings", "betmgm", "caesars", "bet365"),
    notes=(
        "Most US books carry top-of-table only; depth markets are bet365 / "
        "Pinnacle.",
    ),
)

SERIE_A = Sport(
    key="seriea",
    display_name="Serie A",
    status="stub",
    stat_types=_common_player_stats(),
    stat_line_ranges=_common_line_ranges(),
    team_codes=(
        "JUV", "INT", "MIL", "NAP", "ROM", "LAZ", "ATA", "FIO", "BOL",
        "TOR", "UDI", "GEN", "VER", "EMP", "CAG", "PAR", "VEN", "COM",
        # promotion/relegation snapshot
    ),
    season_format="YYYY-YY (Aug YYYY – May YYYY+1)",
    primary_books=("fanduel", "draftkings", "betmgm", "bet365"),
)

BUNDESLIGA = Sport(
    key="bundesliga",
    display_name="Bundesliga",
    status="stub",
    stat_types=_common_player_stats(),
    stat_line_ranges=_common_line_ranges(),
    team_codes=(
        "BAY", "BVB", "B04", "RBL", "EIN", "VFB", "BMG", "FRE", "WOB",
        "HOF", "MAI", "WER", "AUG", "STP", "FCH", "HEI", "UNI", "KIE",
        # promotion/relegation snapshot
    ),
    season_format="YYYY-YY (Aug YYYY – May YYYY+1)",
    primary_books=("fanduel", "draftkings", "betmgm", "bet365"),
)

LIGUE_1 = Sport(
    key="ligue1",
    display_name="Ligue 1",
    status="stub",
    stat_types=_common_player_stats(),
    stat_line_ranges=_common_line_ranges(),
    team_codes=(
        "PSG", "MAR", "MON", "LIL", "NIC", "REN", "LYO", "LEN", "STR",
        "NAN", "TOU", "HAV", "ANG", "AUX", "MTP", "REM", "BRE", "STE",
        # promotion/relegation snapshot
    ),
    season_format="YYYY-YY (Aug YYYY – May YYYY+1)",
    primary_books=("fanduel", "draftkings", "betmgm", "bet365"),
)

UCL = Sport(
    key="ucl",
    display_name="UEFA Champions League",
    status="stub",
    stat_types=_common_player_stats(),
    stat_line_ranges=_common_line_ranges(),
    # Team membership rotates yearly based on each domestic league's results.
    # Keep this empty so we don't bake in a 2025-26 snapshot.
    team_codes=(),
    season_format=(
        "YYYY-YY (Sep YYYY group stage – May YYYY+1 final)"
    ),
    primary_books=("fanduel", "draftkings", "betmgm", "bet365"),
    notes=(
        "Two-legged knockout ties (aggregate score, no away-goals since 2021).",
        "Team membership rotates yearly; track by (season, team) -> league.",
        "Player props in cup competitions are thinner than league markets.",
    ),
)

# ---- Future: South American competitions -----------------------------------
# These are intentionally NOT added to SPORTS yet — they're documented here
# as a planned addition.  When they ship, the entries to register are:
#
# COPA_LIBERTADORES = Sport(
#     key="copa_libertadores",
#     display_name="CONMEBOL Libertadores",
#     status="stub",
#     season_format="YYYY (calendar-year, Feb – Nov)",
#     notes=(
#         "Calendar-year season (vs Euro Aug-May).",
#         "Two-legged knockouts since 2017 final reform.",
#         "Books with depth coverage: bet365, Pinnacle, regional Brazilian "
#         "books (Betano, Stake, Sportingbet).",
#     ),
# )
# COPA_SUDAMERICANA = Sport(
#     key="copa_sudamericana",
#     display_name="CONMEBOL Sudamericana",
#     status="stub",
#     # Same season + book caveats as Libertadores.
# )

SOCCER_SUB_LEAGUES: tuple[Sport, ...] = (
    EPL, LALIGA, SERIE_A, BUNDESLIGA, LIGUE_1, UCL,
)

# Parent entry the registry sees; sub-leagues nested under it.
SPORT = Sport(
    key="soccer",
    display_name="Soccer",
    status="stub",
    season_format="varies by competition - see sub-leagues",
    primary_books=("fanduel", "draftkings", "betmgm", "bet365"),
    notes=(
        "Three-way moneyline market (W/D/L) breaks the binary-outcome "
        "assumption in parlay_compare.py - extend before going live.",
        "Player-prop depth in US books is thin; bet365 / Pinnacle have more.",
        "Promotion/relegation: team_codes per sub-league is a snapshot, not "
        "an enum. Plan a `season_team_membership` table.",
        "Future South American expansion: Copa Libertadores + Copa "
        "Sudamericana. Calendar-year seasons (Feb-Nov). Book coverage is "
        "regional (bet365, Pinnacle, Betano).",
    ),
    sub_leagues=SOCCER_SUB_LEAGUES,
)
