"""Daily ETL runner for game logs, team defense, and odds ingestion."""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from nba_model.data.data_loader import DataLoader
from typing import Callable, Optional

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.data.team_defense_ingestion import (
    build_team_defense_validation_report,
    populate_team_defense,
)
from nba_model.model.odds_ingestion import (
    DEFAULT_REQUEST_RETRIES,
    DEFAULT_REQUEST_RETRY_BACKOFF,
    DEFAULT_REQUEST_RETRY_DELAY_SECONDS,
    DEFAULT_REQUEST_TIMEOUT,
    PLAYER_PROP_MARKETS,
    _default_api_key,
    fetch_and_store_betting_lines,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/database/nba_data.db"
DEFAULT_REPORT_DIR = "nba_model/data/artifacts"
DEFAULT_LOG_DIR = "nba_model/data/logs"
DEFAULT_PLAYERS = [
    "LeBron James",
    "Stephen Curry",
    "Nikola Jokic",
    "Luka Doncic",
    "Jayson Tatum",
]


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def configure_logging(log_dir: str = DEFAULT_LOG_DIR, level: int = logging.INFO) -> str:
    """
    Configure basic ETL logging to console + file and return log file path.
    Safe to call multiple times; only the first call configures handlers.
    """
    if logging.getLogger().handlers:
        # Assume logging already configured by caller (tests or application).
        return ""

    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"daily_etl_{ts}.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logger.info("Daily ETL logging initialized",
                extra={"log_path": str(log_path)})
    return str(log_path)


def _default_nba_season(reference_time: Optional[datetime] = None) -> str:
    """Infer NBA season string from current date."""
    ts = reference_time or datetime.now(timezone.utc)
    start_year = ts.year if ts.month >= 10 else ts.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _serialize_error(exc: Exception) -> dict:
    """Serialize exception into a compact dict for reporting."""
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }


def run_with_retry(
    step_name: str,
    func: Callable[[], dict],
    retries: int = 2,
    retry_delay_seconds: float = 1.0,
    retry_backoff: float = 2.0,
) -> dict:
    """
    Execute callable with retry policy and structured reporting.

    Returns a dict with status, attempt count, elapsed time, and result/error data.
    """
    retries = max(0, int(retries))
    base_delay = max(0.0, float(retry_delay_seconds))
    backoff = max(1.0, float(retry_backoff))

    last_error = None
    attempt = 0
    started = time.perf_counter()

    while attempt <= retries:
        attempt += 1
        try:
            result = func() or {}
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return {
                "status": "success",
                "step": step_name,
                "attempts": attempt,
                "elapsed_ms": elapsed_ms,
                "result": result,
            }
        except Exception as exc:  # pragma: no cover - handled by tests via behavior checks
            last_error = exc
            if attempt > retries:
                break
            delay = base_delay * (backoff ** (attempt - 1))
            logger.warning(
                "%s attempt %s/%s failed: %s. Retrying in %.2fs.",
                step_name,
                attempt,
                retries + 1,
                exc,
                delay,
            )
            if delay > 0:
                time.sleep(delay)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "status": "failed",
        "step": step_name,
        "attempts": attempt,
        "elapsed_ms": elapsed_ms,
        "error": (
            _serialize_error(last_error)
            if last_error
            else {"type": "UnknownError", "message": ""}
        ),
    }


def _list_players_from_db(db_path: str) -> list[str]:
    """Read known player names from players table."""
    with DatabaseManager(db_path=db_path) as db:
        rows = db.conn.execute(
            """
            SELECT name
            FROM players
            WHERE name IS NOT NULL AND trim(name) <> ''
            ORDER BY last_updated DESC, name ASC
            """
        ).fetchall()
    return [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]


def resolve_players(
    explicit_players: Optional[list[str]] = None,
    include_db_players: bool = True,
    db_path: str = DEFAULT_DB_PATH,
    player_limit: Optional[int] = None,
) -> list[str]:
    """Build ordered deduplicated player list for daily game-log refresh."""
    ordered = []
    seen = set()

    for name in explicit_players or []:
        player_name = str(name).strip()
        if player_name and player_name not in seen:
            ordered.append(player_name)
            seen.add(player_name)

    if include_db_players:
        for name in _list_players_from_db(db_path=db_path):
            if name not in seen:
                ordered.append(name)
                seen.add(name)

    if not ordered:
        for name in DEFAULT_PLAYERS:
            if name not in seen:
                ordered.append(name)
                seen.add(name)

    if player_limit is not None:
        return ordered[: max(0, int(player_limit))]
    return ordered


def _refresh_game_logs_for_players(
    players: list[str],
    db_path: str,
    n_games: int,
    force_refresh: bool,
    retries: int,
    retry_delay_seconds: float,
    retry_backoff: float,
) -> dict:
    """Refresh player game logs and return per-player status summary."""
    loader = DataLoader(db_path=db_path)
    player_results = []
    success_count = 0

    try:
        for player_name in players:
            def _load_player():
                df = loader.load_player_data(
                    player_name=player_name,
                    n_games=int(n_games),
                    force_refresh=bool(force_refresh),
                )
                return {
                    "rows_loaded": int(len(df)),
                }

            player_step = run_with_retry(
                step_name=f"game_logs:{player_name}",
                func=_load_player,
                retries=retries,
                retry_delay_seconds=retry_delay_seconds,
                retry_backoff=retry_backoff,
            )
            if player_step["status"] == "success":
                success_count += 1
            player_results.append(
                {
                    "player_name": player_name,
                    "status": player_step["status"],
                    "attempts": player_step["attempts"],
                    "elapsed_ms": player_step["elapsed_ms"],
                    "rows_loaded": player_step.get("result", {}).get("rows_loaded", 0),
                    "error": player_step.get("error"),
                }
            )
    finally:
        close = getattr(loader, "db", None)
        close_method = getattr(close, "close", None)
        if callable(close_method):
            close_method()

    failed_count = len(player_results) - success_count
    status = "success"
    if success_count == 0:
        status = "failed"
    elif failed_count > 0:
        status = "partial_success"

    return {
        "status": status,
        "requested_players": len(players),
        "refreshed_players": success_count,
        "failed_players": failed_count,
        "players": player_results,
    }


def _run_team_defense_step(season: str, db_path: str) -> dict:
    """Populate and validate team defense table."""
    rows_upserted = populate_team_defense(season=season, db_path=db_path)
    validation = build_team_defense_validation_report(
        season=season, db_path=db_path)
    return {
        "rows_upserted": int(rows_upserted),
        "validation": validation,
    }


def _run_odds_step(
    db_path: str,
    odds_api_key: Optional[str],
    sport: str,
    regions: str,
    markets: Optional[list[str]],
    bookmakers: Optional[list[str]],
    max_events: Optional[int],
    sleep_seconds: float,
    request_timeout: int,
    request_retries: int,
    request_retry_delay_seconds: float,
    request_retry_backoff: float,
) -> dict:
    """Fetch/store odds and return ingestion summary."""
    key = odds_api_key or _default_api_key()
    if not key:
        return {
            "status": "skipped",
            "reason": "No odds API key configured.",
        }

    summary = fetch_and_store_betting_lines(
        api_key=key,
        sport=sport,
        regions=regions,
        markets=markets,
        bookmakers=bookmakers,
        db_path=db_path,
        max_events=max_events,
        sleep_seconds=sleep_seconds,
        request_timeout=request_timeout,
        request_retries=request_retries,
        request_retry_delay_seconds=request_retry_delay_seconds,
        request_retry_backoff=request_retry_backoff,
    )
    summary["status"] = "success"
    return summary


def _write_report(report: dict, report_dir: str = DEFAULT_REPORT_DIR) -> str:
    """Persist ETL report to timestamped JSON artifact."""
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"daily_etl_report_{ts}.json"
    path.write_text(json.dumps(report, indent=2,
                    sort_keys=True), encoding="utf-8")
    return str(path)


def run_daily_etl(
    *,
    players: Optional[list[str]] = None,
    include_db_players: bool = True,
    player_limit: Optional[int] = None,
    skip_game_logs: bool = False,
    skip_team_defense: bool = False,
    skip_odds: bool = False,
    game_log_games: int = 120,
    game_log_force_refresh: bool = True,
    season: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
    odds_api_key: Optional[str] = None,
    sport: str = "basketball_nba",
    regions: str = "us",
    markets: Optional[list[str]] = None,
    bookmakers: Optional[list[str]] = None,
    max_events: Optional[int] = None,
    sleep_seconds: float = 0.15,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    request_retries: int = DEFAULT_REQUEST_RETRIES,
    request_retry_delay_seconds: float = DEFAULT_REQUEST_RETRY_DELAY_SECONDS,
    request_retry_backoff: float = DEFAULT_REQUEST_RETRY_BACKOFF,
    retries: int = 2,
    retry_delay_seconds: float = 1.0,
    retry_backoff: float = 2.0,
    report_dir: str = DEFAULT_REPORT_DIR,
    write_report: bool = True,
) -> dict:
    """Run end-to-end daily ETL flow and return structured report."""
    season = season or _default_nba_season()
    selected_players = resolve_players(
        explicit_players=players,
        include_db_players=include_db_players,
        db_path=db_path,
        player_limit=player_limit,
    )

    started_at = _utc_now_iso()
    run_start = time.perf_counter()
    steps = {}

    if skip_game_logs:
        steps["game_logs"] = {"status": "skipped",
                              "reason": "Step disabled by flag."}
    else:
        def _game_logs_step():
            return _refresh_game_logs_for_players(
                players=selected_players,
                db_path=db_path,
                n_games=game_log_games,
                force_refresh=game_log_force_refresh,
                retries=retries,
                retry_delay_seconds=retry_delay_seconds,
                retry_backoff=retry_backoff,
            )

        game_logs_step = run_with_retry(
            step_name="game_logs",
            func=_game_logs_step,
            retries=max(0, retries),
            retry_delay_seconds=retry_delay_seconds,
            retry_backoff=retry_backoff,
        )
        # Promote internal partial/full status to top-level step status.
        if game_logs_step["status"] == "success":
            internal_status = game_logs_step.get("result", {}).get("status")
            if internal_status in {"partial_success", "failed"}:
                game_logs_step["status"] = internal_status
        steps["game_logs"] = game_logs_step

    if skip_team_defense:
        steps["team_defense"] = {"status": "skipped",
                                 "reason": "Step disabled by flag."}
    else:
        team_defense_step = run_with_retry(
            step_name="team_defense",
            func=lambda: _run_team_defense_step(
                season=season, db_path=db_path),
            retries=max(0, retries),
            retry_delay_seconds=retry_delay_seconds,
            retry_backoff=retry_backoff,
        )
        steps["team_defense"] = team_defense_step

    if skip_odds:
        steps["odds"] = {"status": "skipped",
                         "reason": "Step disabled by flag."}
    else:
        odds_step = run_with_retry(
            step_name="odds",
            func=lambda: _run_odds_step(
                db_path=db_path,
                odds_api_key=odds_api_key,
                sport=sport,
                regions=regions,
                markets=markets,
                bookmakers=bookmakers,
                max_events=max_events,
                sleep_seconds=sleep_seconds,
                request_timeout=request_timeout,
                request_retries=request_retries,
                request_retry_delay_seconds=request_retry_delay_seconds,
                request_retry_backoff=request_retry_backoff,
            ),
            retries=max(0, retries),
            retry_delay_seconds=retry_delay_seconds,
            retry_backoff=retry_backoff,
        )
        # Allow explicit skip payload from odds step.
        if odds_step["status"] == "success":
            if odds_step.get("result", {}).get("status") == "skipped":
                odds_step["status"] = "skipped"
        steps["odds"] = odds_step

    step_statuses = {name: payload.get("status")
                     for name, payload in steps.items()}
    has_failures = any(status == "failed" for status in step_statuses.values())
    has_partial = any(
        status == "partial_success" for status in step_statuses.values())

    if has_failures:
        overall_status = "failed"
    elif has_partial:
        overall_status = "partial_success"
    else:
        overall_status = "success"

    finished_at = _utc_now_iso()
    total_elapsed_ms = int((time.perf_counter() - run_start) * 1000)

    report = {
        "status": overall_status,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "elapsed_ms": total_elapsed_ms,
        "season": season,
        "db_path": db_path,
        "players_selected": selected_players,
        "steps": steps,
    }

    if write_report:
        report["report_path"] = _write_report(report, report_dir=report_dir)

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run daily ETL for NBA data pipeline.")
    parser.add_argument("--players", nargs="*", default=None,
                        help="Optional explicit player names.")
    parser.add_argument(
        "--skip-db-players",
        action="store_true",
        help="Do not add players currently stored in the local players table.",
    )
    parser.add_argument("--player-limit", type=int, default=None,
                        help="Optional cap on total players refreshed.")
    parser.add_argument("--skip-game-logs", action="store_true")
    parser.add_argument("--skip-team-defense", action="store_true")
    parser.add_argument("--skip-odds", action="store_true")
    parser.add_argument("--game-log-games", type=int, default=120)
    parser.add_argument(
        "--use-cache-for-game-logs",
        action="store_true",
        help="Use cached data when available instead of force-refreshing NBA API data.",
    )
    parser.add_argument("--season", default=None,
                        help="NBA season string (e.g., 2024-25). Defaults from current date.")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--odds-api-key", default=None)
    parser.add_argument("--sport", default="basketball_nba")
    parser.add_argument("--regions", default="us")
    parser.add_argument("--bookmakers", nargs="*", default=None)
    parser.add_argument(
        "--markets",
        nargs="*",
        default=list(PLAYER_PROP_MARKETS.keys()),
        help="Odds API market keys to ingest.",
    )
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--request-timeout", type=int,
                        default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--request-retries", type=int,
                        default=DEFAULT_REQUEST_RETRIES)
    parser.add_argument(
        "--request-retry-delay-seconds",
        type=float,
        default=DEFAULT_REQUEST_RETRY_DELAY_SECONDS,
    )
    parser.add_argument("--request-retry-backoff", type=float,
                        default=DEFAULT_REQUEST_RETRY_BACKOFF)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code when any step is failed or partial_success.",
    )
    parser.add_argument("--no-write-report", action="store_true")
    return parser


def main() -> None:
    """CLI entry point: run daily ETL and write report."""
    log_path = configure_logging()
    args = _build_parser().parse_args()
    report = run_daily_etl(
        players=args.players,
        include_db_players=not args.skip_db_players,
        player_limit=args.player_limit,
        skip_game_logs=args.skip_game_logs,
        skip_team_defense=args.skip_team_defense,
        skip_odds=args.skip_odds,
        game_log_games=args.game_log_games,
        game_log_force_refresh=not args.use_cache_for_game_logs,
        season=args.season,
        db_path=args.db_path,
        odds_api_key=args.odds_api_key,
        sport=args.sport,
        regions=args.regions,
        markets=args.markets,
        bookmakers=args.bookmakers,
        max_events=args.max_events,
        sleep_seconds=args.sleep_seconds,
        request_timeout=args.request_timeout,
        request_retries=args.request_retries,
        request_retry_delay_seconds=args.request_retry_delay_seconds,
        request_retry_backoff=args.request_retry_backoff,
        retries=args.retries,
        retry_delay_seconds=args.retry_delay_seconds,
        retry_backoff=args.retry_backoff,
        report_dir=args.report_dir,
        write_report=not args.no_write_report,
    )

    print("Daily ETL summary:")
    print(f"- status: {report['status']}")
    print(f"- started_at_utc: {report['started_at_utc']}")
    print(f"- finished_at_utc: {report['finished_at_utc']}")
    print(f"- elapsed_ms: {report['elapsed_ms']}")
    print(f"- season: {report['season']}")
    print(f"- players_selected: {len(report['players_selected'])}")
    if report.get("report_path"):
        print(f"- report_path: {report['report_path']}")
    if log_path:
        print(f"- log_path: {log_path}")

    for step_name, payload in report["steps"].items():
        print(f"- step[{step_name}]: {payload.get('status')}")

    if args.strict and report["status"] in {"failed", "partial_success"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
