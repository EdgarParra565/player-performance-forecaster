"""Bulk NBA results ingestion via nba_api.

Two passes per season:
  1. **Games**: ``leaguegamefinder.LeagueGameFinder`` returns one row per
     team-game (so two rows per game). We upsert all rows into ``games``.
  2. **Player game logs**: ``playergamelogs.PlayerGameLogs`` returns the
     full league's player game-log frame in a single request — much faster
     than the per-player ``playergamelog.PlayerGameLog`` endpoint that the
     legacy DataLoader uses.

Each season is fetched independently with retries; the orchestrator stops
at the first season where the API returns nothing (current season starts).

CLI:
    python -m nba_model.data.nba_results_ingestion --seasons 2024-25 2023-24
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from typing import Iterable, Optional

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playergamelogs

from nba_model.data.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

DEFAULT_SEASONS = ("2025-26", "2024-25", "2023-24")
DEFAULT_SEASON_TYPES = ("Regular Season", "Playoffs")
RATE_LIMIT_SECONDS = 0.6
DEFAULT_MAX_ATTEMPTS = 3


def _retry(callable_, attempts: int = DEFAULT_MAX_ATTEMPTS):
    """Run ``callable_()`` with exponential backoff between failures.

    nba_api throttles aggressively; brief retries handle the common
    ``ReadTimeout``/``HTTPError`` cases without surfacing to the caller.
    """
    last: Optional[Exception] = None
    for i in range(1, attempts + 1):
        try:
            return callable_()
        except Exception as exc:  # nba_api raises a variety of error types
            last = exc
            wait = 1.5 ** i
            logger.warning(
                "nba_api call failed (attempt %s/%s): %s — retrying in %.1fs",
                i, attempts, exc, wait,
            )
            time.sleep(wait)
    raise last  # type: ignore[misc]


def _parse_matchup(matchup: str) -> tuple[Optional[str], str]:
    """Extract opponent abbrev + home/away marker from an nba_api MATCHUP cell.

    Examples: "NYK vs. PHI" → ("PHI", "home"); "NYK @ PHI" → ("PHI", "away").
    """
    if not matchup:
        return (None, "")
    text = str(matchup).strip()
    if " vs. " in text or " vs " in text:
        sep = " vs. " if " vs. " in text else " vs "
        _, opp = text.split(sep, 1)
        return (opp.strip().upper(), "home")
    if " @ " in text:
        _, opp = text.split(" @ ", 1)
        return (opp.strip().upper(), "away")
    return (None, "")


def fetch_games_for_season(season: str) -> pd.DataFrame:
    """Fetch one season of team-game rows via leaguegamefinder."""
    logger.info("Fetching games for season %s …", season)
    time.sleep(RATE_LIMIT_SECONDS)
    finder = _retry(lambda: leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",  # NBA
    ))
    df = finder.get_data_frames()[0]
    df["SEASON"] = season
    return df


def transform_games_frame(df: pd.DataFrame) -> list[dict]:
    """Convert nba_api leaguegamefinder rows → our ``games`` schema dicts."""
    if df is None or df.empty:
        return []
    out: list[dict] = []
    for row in df.itertuples(index=False):
        as_dict = row._asdict() if hasattr(row, "_asdict") else dict(row)
        opp, home_away = _parse_matchup(as_dict.get("MATCHUP"))
        # leaguegamefinder doesn't return SEASON_TYPE directly — the type
        # is encoded in the leading digit of SEASON_ID (e.g. "22024" =
        # Regular Season 2024-25, "42024" = Playoffs 2024-25).
        sid = str(as_dict.get("SEASON_ID") or "")
        season_type_map = {
            "1": "Pre Season", "2": "Regular Season",
            "4": "Playoffs",   "5": "Play-In", "6": "All-Star",
        }
        season_type = season_type_map.get(sid[:1], "Regular Season")
        out.append({
            "game_id":         as_dict.get("GAME_ID"),
            "season":          as_dict.get("SEASON"),
            "season_type":     season_type,
            "game_date":       as_dict.get("GAME_DATE"),
            "team_id":         as_dict.get("TEAM_ID"),
            "team_abbrev":     as_dict.get("TEAM_ABBREVIATION"),
            "team_name":       as_dict.get("TEAM_NAME"),
            "matchup":         as_dict.get("MATCHUP"),
            "home_away":       home_away,
            "opponent_abbrev": opp,
            "result":          as_dict.get("WL"),
            "pts":             as_dict.get("PTS"),
            # opp_pts isn't exposed by leaguegamefinder; we'll fill it in
            # downstream by joining the two team rows for the same game.
            "opp_pts":         None,
            "plus_minus":      as_dict.get("PLUS_MINUS"),
            "fg_pct":          as_dict.get("FG_PCT"),
            "fg3_pct":         as_dict.get("FG3_PCT"),
            "ft_pct":          as_dict.get("FT_PCT"),
            "rebounds":        as_dict.get("REB"),
            "assists":         as_dict.get("AST"),
            "steals":          as_dict.get("STL"),
            "blocks":          as_dict.get("BLK"),
            "turnovers":       as_dict.get("TOV"),
        })
    return out


def _backfill_opp_pts(records: list[dict]) -> list[dict]:
    """Fill ``opp_pts`` for each row by looking up the other team in the game."""
    by_game: dict[str, dict[int, dict]] = {}
    for r in records:
        gid = r.get("game_id")
        if not gid:
            continue
        by_game.setdefault(gid, {})[int(r.get("team_id") or 0)] = r
    for gid, teams in by_game.items():
        if len(teams) != 2:
            continue
        team_a, team_b = list(teams.values())
        team_a["opp_pts"] = team_b.get("pts")
        team_b["opp_pts"] = team_a.get("pts")
    return records


def ingest_games(
    seasons: Iterable[str] = DEFAULT_SEASONS,
    db_path: str = "data/database/nba_data.db",
) -> dict:
    """Pull game-level results for ``seasons`` and upsert into ``games``."""
    summary = {"seasons": [], "total_attempted": 0, "total_inserted": 0}
    with DatabaseManager(db_path=db_path) as db:
        for season in seasons:
            try:
                frame = fetch_games_for_season(season)
            except Exception as exc:
                summary["seasons"].append(
                    {"season": season, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            records = _backfill_opp_pts(transform_games_frame(frame))
            res = db.insert_games(records)
            summary["seasons"].append({
                "season": season,
                "rows_pulled": len(records),
                "inserted": res["inserted"],
                "attempted": res["attempted"],
            })
            summary["total_attempted"] += res["attempted"]
            summary["total_inserted"] += res["inserted"]
    return summary


# ----- Player game logs ----------------------------------------------------


def fetch_player_game_logs_for_season(season: str) -> pd.DataFrame:
    """Fetch one season of league-wide player game logs."""
    logger.info("Fetching player game logs for season %s …", season)
    time.sleep(RATE_LIMIT_SECONDS)
    res = _retry(lambda: playergamelogs.PlayerGameLogs(season_nullable=season))
    return res.get_data_frames()[0]


def transform_player_logs_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map the playergamelogs frame into ``game_logs`` table columns."""
    if df is None or df.empty:
        return pd.DataFrame()
    rename = {
        "PLAYER_ID":   "player_id",
        "GAME_ID":     "game_id",
        "GAME_DATE":   "game_date",
        "SEASON_YEAR": "season",
        "MATCHUP":     "matchup",
        "WL":          "result",
        "MIN":         "minutes",
        "PTS":         "points",
        "FGM":         "fgm",
        "FGA":         "fga",
        "FG_PCT":      "fg_pct",
        "FG3M":        "fg3m",
        "FG3A":        "fg3a",
        "FG3_PCT":     "fg3_pct",
        "FTM":         "ftm",
        "FTA":         "fta",
        "FT_PCT":      "ft_pct",
        "OREB":        "oreb",
        "DREB":        "dreb",
        "REB":         "rebounds",
        "AST":         "assists",
        "STL":         "steals",
        "BLK":         "blocks",
        "TOV":         "turnovers",
        "PLUS_MINUS":  "plus_minus",
    }
    keep = {k: v for k, v in rename.items() if k in df.columns}
    out = df.rename(columns=keep).copy()
    out = out[[v for v in keep.values() if v in out.columns]]
    if "matchup" in out.columns:
        out["home_away"] = out["matchup"].apply(
            lambda m: "home" if isinstance(m, str) and " vs. " in m else "away"
        )
    return out


def ingest_player_game_logs(
    seasons: Iterable[str] = DEFAULT_SEASONS,
    db_path: str = "data/database/nba_data.db",
) -> dict:
    """Pull league-wide player game logs and insert into ``game_logs``."""
    summary = {"seasons": [], "total_inserted": 0, "total_attempted": 0}
    with DatabaseManager(db_path=db_path) as db:
        for season in seasons:
            try:
                df = fetch_player_game_logs_for_season(season)
            except Exception as exc:
                summary["seasons"].append(
                    {"season": season, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            mapped = transform_player_logs_frame(df)
            attempted = int(len(mapped))
            inserted = 0
            if attempted > 0:
                inserted = int(db.insert_game_logs(mapped))
            summary["seasons"].append({
                "season": season, "attempted": attempted, "inserted": inserted,
            })
            summary["total_attempted"] += attempted
            summary["total_inserted"] += inserted
    return summary


def ingest_all(
    seasons: Iterable[str] = DEFAULT_SEASONS,
    db_path: str = "data/database/nba_data.db",
    skip_player_logs: bool = False,
) -> dict:
    """Convenience wrapper: ingest games + (optionally) player game logs."""
    games_summary = ingest_games(seasons=seasons, db_path=db_path)
    if skip_player_logs:
        return {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "games": games_summary,
            "player_logs": {"skipped": True},
        }
    logs_summary = ingest_player_game_logs(seasons=seasons, db_path=db_path)
    return {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "games": games_summary,
        "player_logs": logs_summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Bulk-ingest NBA games + player game logs from nba_api.",
    )
    parser.add_argument("--db-path", default="data/database/nba_data.db")
    parser.add_argument(
        "--seasons", nargs="+", default=list(DEFAULT_SEASONS),
        help="Seasons to ingest (e.g. 2024-25 2023-24).",
    )
    parser.add_argument(
        "--skip-player-logs", action="store_true",
        help="Only ingest games (faster; useful for quick refreshes).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    summary = ingest_all(
        seasons=args.seasons,
        db_path=args.db_path,
        skip_player_logs=args.skip_player_logs,
    )
    print("Ingest summary:")
    print(f"  games:")
    for s in summary.get("games", {}).get("seasons", []):
        print(f"    - {s}")
    if not summary.get("player_logs", {}).get("skipped"):
        print(f"  player_logs:")
        for s in summary.get("player_logs", {}).get("seasons", []):
            print(f"    - {s}")


if __name__ == "__main__":
    main()
