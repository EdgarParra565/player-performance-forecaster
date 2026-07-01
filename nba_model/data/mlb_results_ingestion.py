"""MLB results ingestion via the official MLB Stats API.

Primary source: ``https://statsapi.mlb.com/api/v1`` — the MLB.com / Gameday
backend. Free, no API key, no auth. This is the MLB analog to how the NBA
side uses ``nba_api`` / stats.nba.com, and we prefer it over scraping for
player game logs (the book scrapers in ``scrapers/*_mlb.py`` supply the
*lines*; this module supplies the *games + players* those lines attach to).

Design (mirrors ``nfl_results_ingestion.py``):
  * HTTP lives in thin ``fetch_*`` functions (using ``requests``, already a
    dependency — bundles certifi so TLS works where bare urllib doesn't).
  * The ``transform_*`` functions are PURE: they take already-parsed JSON and
    return normalized rows. They are unit-tested against committed JSON
    fixtures and never touch the network.
  * Game logs are written to the dedicated ``mlb_game_logs`` table (long
    format, ``sport='mlb'``) so MLB rows never leak into NBA queries.

Optional supplement: ``pybaseball`` (Statcast / park factors) is imported
lazily and guarded — absent in this env, so it must never be required to
import this module or run the primary path.

CLI:
    .venv/bin/python3 -m nba_model.data.mlb_results_ingestion \\
        --start-date 2026-06-28 --end-date 2026-06-28
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
from typing import Optional

import requests

from nba_model.data.database.db_manager import DatabaseManager

logger = logging.getLogger("nba_model.mlb_results_ingestion")

API_BASE = "https://statsapi.mlb.com/api/v1"
DEFAULT_TIMEOUT = 20
SPORT_ID_MLB = 1

# MLB Stats API boxscore batting field -> canonical hitter stat key.
# ('singles' is derived; see _hitter_stats_from_batting.)
_BATTING_FIELD_MAP: dict[str, str] = {
    "hits": "hits",
    "totalBases": "total_bases",
    "homeRuns": "home_runs",
    "rbi": "rbis",
    "runs": "runs_scored",
    "stolenBases": "stolen_bases",
    "baseOnBalls": "walks",
    "strikeOuts": "strikeouts_batter",
}
# MLB Stats API boxscore pitching field -> canonical pitcher stat key.
_PITCHING_FIELD_MAP: dict[str, str] = {
    "strikeOuts": "strikeouts_pitcher",
    "earnedRuns": "earned_runs",
    "outs": "outs_recorded",
    "hits": "hits_allowed",
    "baseOnBalls": "walks_allowed",
    "wins": "wins",
}


# --------------------------------------------------------------------------
# HTTP layer (network) — thin wrappers, not exercised in tests.
# --------------------------------------------------------------------------
def _get_json(url: str, params: Optional[dict] = None, timeout: int = DEFAULT_TIMEOUT) -> dict:
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_schedule(start_date: str, end_date: str) -> dict:
    """Raw schedule JSON for an inclusive date range (YYYY-MM-DD)."""
    return _get_json(
        f"{API_BASE}/schedule",
        params={"sportId": SPORT_ID_MLB, "startDate": start_date, "endDate": end_date},
    )


def fetch_boxscore(game_pk: int) -> dict:
    """Raw boxscore JSON for one game."""
    return _get_json(f"{API_BASE}/game/{int(game_pk)}/boxscore")


def fetch_player_game_log(person_id: int, group: str = "hitting", season: Optional[int] = None) -> dict:
    """Raw per-player gameLog JSON (group='hitting' or 'pitching')."""
    params = {"stats": "gameLog", "group": group}
    if season is not None:
        params["season"] = int(season)
    return _get_json(f"{API_BASE}/people/{int(person_id)}/stats", params=params)


def fetch_team_roster(team_id: int) -> dict:
    """Raw roster JSON for a team."""
    return _get_json(f"{API_BASE}/teams/{int(team_id)}/roster")


# --------------------------------------------------------------------------
# Pure transforms (no network) — fixture-tested.
# --------------------------------------------------------------------------
def transform_schedule(schedule_json: dict) -> list[dict]:
    """Flatten schedule JSON → one dict per game (gamePk + teams + date)."""
    out: list[dict] = []
    for date_block in (schedule_json or {}).get("dates", []) or []:
        for game in date_block.get("games", []) or []:
            teams = game.get("teams", {})
            home = teams.get("home", {}).get("team", {})
            away = teams.get("away", {}).get("team", {})
            game_pk = game.get("gamePk")
            if game_pk is None:
                continue
            out.append({
                "game_pk": int(game_pk),
                "game_date": game.get("officialDate") or date_block.get("date"),
                "season": int(game["season"]) if str(game.get("season") or "").isdigit() else None,
                "home_team_id": home.get("id"),
                "home_team_name": home.get("name"),
                "away_team_id": away.get("id"),
                "away_team_name": away.get("name"),
                "status": (game.get("status", {}) or {}).get("abstractGameState"),
            })
    return out


def _singles_from_batting(b: dict) -> Optional[float]:
    """singles = hits - doubles - triples - HR (None when hits absent)."""
    if b.get("hits") is None:
        return None
    try:
        return float(
            int(b.get("hits", 0))
            - int(b.get("doubles", 0))
            - int(b.get("triples", 0))
            - int(b.get("homeRuns", 0))
        )
    except (TypeError, ValueError):
        return None


def _rows_from_player(
    player: dict, group: str, field_map: dict, game_pk, game_date, season, team, opponent
) -> list[dict]:
    stats = (player.get("stats", {}) or {}).get(group, {}) or {}
    if not stats:
        return []
    person = player.get("person", {}) or {}
    pid = person.get("id")
    if pid is None:
        return []
    base = {
        "sport": "mlb",
        "player_id": int(pid),
        "player_name": person.get("fullName"),
        "team": team,
        "opponent": opponent,
        "game_pk": int(game_pk) if game_pk is not None else None,
        "game_date": game_date,
        "season": season,
        "player_group": group,
    }
    rows: list[dict] = []
    for field, key in field_map.items():
        val = stats.get(field)
        if val is None:
            continue
        try:
            rows.append({**base, "stat_type": key, "value": float(val)})
        except (TypeError, ValueError):
            continue
    # Derived hitter stat: singles.
    if group == "batting":
        singles = _singles_from_batting(stats)
        if singles is not None:
            rows.append({**base, "stat_type": "singles", "value": singles})
    return rows


def transform_boxscore_to_player_logs(
    boxscore_json: dict,
    game_pk: Optional[int] = None,
    game_date: Optional[str] = None,
    season: Optional[int] = None,
) -> list[dict]:
    """Boxscore JSON → long-format per-player-per-stat rows (sport='mlb').

    Emits hitter rows (``player_group='hitting'``) and pitcher rows
    (``player_group='pitching'``) keyed to canonical ``sports/mlb.py`` stats.
    ``player_group`` is normalized to 'hitting'/'pitching' even though the API
    nests them under 'batting'/'pitching'.
    """
    out: list[dict] = []
    teams = (boxscore_json or {}).get("teams", {}) or {}
    home = teams.get("home", {}) or {}
    away = teams.get("away", {}) or {}
    home_abbr = (home.get("team", {}) or {}).get("abbreviation")
    away_abbr = (away.get("team", {}) or {}).get("abbreviation")

    for side, team_abbr, opp_abbr in (
        (home, home_abbr, away_abbr),
        (away, away_abbr, home_abbr),
    ):
        for player in (side.get("players", {}) or {}).values():
            batting = _rows_from_player(
                player, "batting", _BATTING_FIELD_MAP,
                game_pk, game_date, season, team_abbr, opp_abbr,
            )
            for row in batting:
                row["player_group"] = "hitting"
            pitching = _rows_from_player(
                player, "pitching", _PITCHING_FIELD_MAP,
                game_pk, game_date, season, team_abbr, opp_abbr,
            )
            out.extend(batting)
            out.extend(pitching)
    return out


# --------------------------------------------------------------------------
# Orchestration (network) — pulls a date range and persists to mlb_game_logs.
# --------------------------------------------------------------------------
def ingest_date_range(
    start_date: str,
    end_date: str,
    db_path: str = "data/database/nba_data.db",
) -> dict:
    """Fetch schedule → boxscores for a date range, persist to ``mlb_game_logs``."""
    schedule = transform_schedule(fetch_schedule(start_date, end_date))
    all_rows: list[dict] = []
    for game in schedule:
        if game.get("status") and game["status"] not in ("Final", "Live"):
            continue
        try:
            box = fetch_boxscore(game["game_pk"])
        except requests.RequestException as exc:  # noqa: BLE001
            logger.warning("Boxscore fetch failed for %s: %s", game["game_pk"], exc)
            continue
        rows = transform_boxscore_to_player_logs(
            box, game_pk=game["game_pk"],
            game_date=game.get("game_date"), season=game.get("season"),
        )
        all_rows.extend(rows)

    with DatabaseManager(db_path=db_path) as db:
        summary = db.insert_mlb_game_logs(all_rows)

    return {
        "sport": "mlb",
        "games_considered": len(schedule),
        "rows_built": len(all_rows),
        "inserted": int(summary.get("inserted", 0)),
        "duplicates_ignored": int(summary.get("duplicates_ignored", 0)),
    }


# --------------------------------------------------------------------------
# Optional pybaseball supplement — guarded like the NFL dep.
# --------------------------------------------------------------------------
def pybaseball_available() -> bool:
    """True when the optional ``pybaseball`` supplement can be imported."""
    return importlib.util.find_spec("pybaseball") is not None


def _import_pybaseball():
    if not pybaseball_available():
        raise RuntimeError(
            "pybaseball is not installed (optional Statcast/park-factor "
            "supplement). Install on a live host: "
            "`.venv/bin/python3 -m pip install pybaseball`. The MLB Stats API "
            "path (this module's fetch_*/transform_*) is the primary source "
            "and needs no extra deps."
        )
    return importlib.import_module("pybaseball")


def fetch_statcast_batter(start_dt: str, end_dt: str, player_id: int):
    """Optional Statcast pull (needs pybaseball)."""
    pb = _import_pybaseball()
    return pb.statcast_batter(start_dt, end_dt, player_id)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest MLB player game logs from the MLB Stats API.",
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--db-path", default="data/database/nba_data.db")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args()
    summary = ingest_date_range(args.start_date, args.end_date, db_path=args.db_path)
    print("MLB ingestion summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
