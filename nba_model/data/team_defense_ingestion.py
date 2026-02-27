"""Populate and validate team_defense table from NBA API team stats."""

import argparse
import logging
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams as static_teams

from nba_model.data.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)
DEFAULT_SEASON = "2024-25"
DEFAULT_DB_PATH = "data/database/nba_data.db"


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return first matching column name from candidate list."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _team_mappings():
    """Build static mappings for team id/name -> abbreviation."""
    teams_data = static_teams.get_teams()
    by_id = {}
    by_name = {}
    for team in teams_data:
        team_id = team.get("id")
        abbr = str(team.get("abbreviation", "")).upper()
        full_name = str(team.get("full_name", "")).strip().lower()
        nickname = str(team.get("nickname", "")).strip().lower()
        if team_id is not None and abbr:
            by_id[int(team_id)] = abbr
        if full_name and abbr:
            by_name[full_name] = abbr
        if nickname and abbr:
            by_name[nickname] = abbr
    return by_id, by_name


def _expected_team_abbrevs() -> set[str]:
    """Return expected NBA team abbreviation set from static metadata."""
    teams_data = static_teams.get_teams()
    return {
        str(team.get("abbreviation", "")).strip().upper()
        for team in teams_data
        if str(team.get("abbreviation", "")).strip()
    }


def _resolve_team_abbrev_series(df: pd.DataFrame) -> pd.Series:
    """
    Resolve team abbreviation from available API columns.

    Supports TEAM_ABBREVIATION/TEAM_ABBREV directly, then TEAM_ID, then TEAM_NAME.
    """
    team_col = _find_col(df, ["TEAM_ABBREVIATION", "TEAM_ABBREV"])
    if team_col:
        return df[team_col].astype(str).str.upper()

    by_id, by_name = _team_mappings()

    team_id_col = _find_col(df, ["TEAM_ID"])
    if team_id_col:
        mapped = pd.to_numeric(df[team_id_col], errors="coerce").map(
            lambda value: by_id.get(int(value)) if pd.notna(value) else None
        )
        if mapped.notna().any():
            return mapped

    team_name_col = _find_col(df, ["TEAM_NAME", "TEAM"])
    if team_name_col:
        mapped = df[team_name_col].astype(str).str.strip().str.lower().map(by_name)
        if mapped.notna().any():
            return mapped

    raise KeyError(f"Unable to resolve team abbreviation from columns: {list(df.columns)}")


def fetch_team_defense_df(season: str = DEFAULT_SEASON) -> pd.DataFrame:
    """
    Fetch NBA team defensive metrics for a season.

    Returns dataframe containing team abbreviation and key defensive fields.
    """
    endpoint = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Defense",
        per_mode_detailed="PerGame",
    )
    df = endpoint.get_data_frames()[0]
    if df.empty:
        raise ValueError("No team defense data returned from NBA API.")

    team_abbrev_series = _resolve_team_abbrev_series(df)
    def_col = _find_col(df, ["DEF_RATING", "DEFRTG", "DEF_RTG"])
    opp_col = _find_col(df, ["OPP_PTS", "OPP_PTS_PG", "OPP_PPG"])
    pace_col = _find_col(df, ["PACE"])

    if not def_col:
        raise KeyError(f"Expected DEF_RATING column missing in API response: {list(df.columns)}")

    rows = pd.DataFrame(
        {
            "team_abbrev": team_abbrev_series.astype(str).str.upper(),
            "season": season,
            "def_rating": pd.to_numeric(df[def_col], errors="coerce"),
            "opp_ppg": pd.to_numeric(df[opp_col], errors="coerce") if opp_col else None,
            "pace": pd.to_numeric(df[pace_col], errors="coerce") if pace_col else None,
        }
    )
    rows = rows.dropna(subset=["team_abbrev", "def_rating"]).reset_index(drop=True)
    return rows


def populate_team_defense(season: str = DEFAULT_SEASON, db_path: str = DEFAULT_DB_PATH) -> int:
    """Fetch and upsert team defense rows into SQLite."""
    rows = fetch_team_defense_df(season=season)
    records = rows.to_dict(orient="records")
    with DatabaseManager(db_path=db_path) as db:
        db.insert_team_defense_records(records)
    logger.info(f"Populated team_defense for season={season} with {len(records)} rows")
    return len(records)


def build_team_defense_validation_report(
    season: Optional[str] = DEFAULT_SEASON,
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """
    Build lightweight integrity report for team_defense coverage.

    Report fields include row count, unique teams present, missing teams and
    latest update timestamp for the requested season.
    """
    where_clause = "WHERE season = ?" if season else ""
    params = (season,) if season else ()

    with DatabaseManager(db_path=db_path) as db:
        row_count = db.conn.execute(
            f"SELECT COUNT(*) FROM team_defense {where_clause}",
            params,
        ).fetchone()[0]
        latest_updated = db.conn.execute(
            f"SELECT MAX(last_updated) FROM team_defense {where_clause}",
            params,
        ).fetchone()[0]
        rows = db.conn.execute(
            f"SELECT team_abbrev FROM team_defense {where_clause}",
            params,
        ).fetchall()

    loaded_teams = {
        str(row[0]).strip().upper()
        for row in rows
        if row and row[0] and str(row[0]).strip()
    }
    expected_teams = _expected_team_abbrevs()
    missing_teams = sorted(expected_teams - loaded_teams)
    unexpected_teams = sorted(loaded_teams - expected_teams)

    return {
        "season": season,
        "row_count": int(row_count or 0),
        "team_count": len(loaded_teams),
        "expected_team_count": len(expected_teams),
        "missing_teams": missing_teams,
        "unexpected_teams": unexpected_teams,
        "latest_updated": latest_updated,
        "is_complete": (
            len(loaded_teams) == len(expected_teams)
            and not missing_teams
            and not unexpected_teams
        ),
    }


def print_team_defense_validation_report(report: dict) -> None:
    """Print a concise validation report for team_defense coverage."""
    season_label = report.get("season") or "all seasons"
    print("\nTeam defense validation")
    print(f"Season: {season_label}")
    print(f"Rows: {report.get('row_count', 0)}")
    print(
        "Teams present: "
        f"{report.get('team_count', 0)}/{report.get('expected_team_count', 0)}"
    )
    print(f"Latest update: {report.get('latest_updated') or 'N/A'}")
    missing_teams = report.get("missing_teams") or []
    unexpected_teams = report.get("unexpected_teams") or []
    print(f"Missing teams: {', '.join(missing_teams) if missing_teams else 'none'}")
    print(f"Unexpected teams: {', '.join(unexpected_teams) if unexpected_teams else 'none'}")
    print(f"Coverage complete: {'yes' if report.get('is_complete') else 'no'}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Populate team_defense table from NBA API.")
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip ingestion and only run validation report.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip post-ingestion validation report.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit with code 1 when validation finds missing/unexpected teams.",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    if not args.validate_only:
        count = populate_team_defense(season=args.season, db_path=args.db_path)
        print(f"Inserted/updated team_defense rows: {count}")

    if args.skip_validation:
        return

    report = build_team_defense_validation_report(season=args.season, db_path=args.db_path)
    print_team_defense_validation_report(report)

    if args.fail_on_missing and (report["missing_teams"] or report["unexpected_teams"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
