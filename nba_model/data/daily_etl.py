"""Daily ETL runner for game logs, team defense, odds, and reverse-engineering."""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from nba_model.data.data_loader import DataLoader
from typing import Callable, Optional

from nba_api.stats.static import players as nba_players

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.data.team_defense_ingestion import (
    build_team_defense_validation_report,
    populate_team_defense,
)
from nba_model.evaluation.market_reverse_engineering import (
    run_market_reverse_engineering,
    run_market_reverse_engineering_continuous,
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


def _list_active_player_names() -> list[str]:
    """Read active NBA players from nba_api's static catalog."""
    active_players = nba_players.get_active_players()
    names = sorted(
        {
            str(player.get("full_name", "")).strip()
            for player in active_players
            if str(player.get("full_name", "")).strip()
        }
    )
    return names


def _player_names_with_existing_game_logs(
    player_names: list[str],
    db_path: str,
) -> tuple[set[str], set[str], set[str]]:
    """
    Return names that already have at least one local game_log row.

    Returns:
        tuple[set[str], set[str], set[str]]:
            - names_with_logs: resolvable names with at least one game_log row
            - unresolved_names: names that could not be resolved to an NBA player id
            - names_without_logs: resolvable names with zero local game_log rows
    """
    if not player_names:
        return set(), set(), set()

    with DatabaseManager(db_path=db_path) as db:
        rows = db.conn.execute(
            """
            SELECT DISTINCT player_id
            FROM game_logs
            WHERE player_id IS NOT NULL
            """
        ).fetchall()

    player_ids_with_logs = {
        int(row[0]) for row in rows if row and row[0] is not None
    }
    if not player_ids_with_logs:
        # Bootstrap state: no local logs exist yet, so we cannot classify names as
        # "zero-game". Mark all as unresolved so caller keeps them.
        unresolved_names = {
            str(name).strip() for name in player_names if str(name).strip()
        }
        return set(), unresolved_names, set()

    names_with_logs: set[str] = set()
    unresolved_names: set[str] = set()
    names_without_logs: set[str] = set()
    for name in player_names:
        normalized_name = str(name).strip()
        candidates = nba_players.find_players_by_full_name(normalized_name)
        if not candidates:
            unresolved_names.add(normalized_name)
            continue
        player_id = candidates[0].get("id")
        try:
            pid = int(player_id)
        except (TypeError, ValueError):
            unresolved_names.add(normalized_name)
            continue
        if pid in player_ids_with_logs:
            names_with_logs.add(normalized_name)
        else:
            names_without_logs.add(normalized_name)

    return names_with_logs, unresolved_names, names_without_logs


def _build_player_selection(
    explicit_players: Optional[list[str]] = None,
    include_db_players: bool = True,
    db_path: str = DEFAULT_DB_PATH,
    player_limit: Optional[int] = None,
    all_db_players: bool = False,
    min_players: Optional[int] = None,
    skip_zero_game_players: bool = False,
) -> tuple[list[str], dict]:
    """Build selected players and source breakdown metadata."""
    ordered = []
    source_tags = []
    seen = set()

    for name in explicit_players or []:
        player_name = str(name).strip()
        if player_name and player_name not in seen:
            ordered.append(player_name)
            source_tags.append("explicit")
            seen.add(player_name)

    if include_db_players or all_db_players:
        for name in _list_players_from_db(db_path=db_path):
            if name not in seen:
                ordered.append(name)
                source_tags.append("db")
                seen.add(name)

    if not ordered and not all_db_players:
        for name in DEFAULT_PLAYERS:
            if name not in seen:
                ordered.append(name)
                source_tags.append("default_seed")
                seen.add(name)

    min_count = None
    if min_players is not None:
        min_count = max(0, int(min_players))

    if min_count is not None and len(ordered) < min_count:
        for name in _list_active_player_names():
            if name not in seen:
                ordered.append(name)
                source_tags.append("min_topup")
                seen.add(name)
            if len(ordered) >= min_count:
                break

    skipped_zero_game_count = 0
    skipped_zero_game_examples: list[str] = []
    unresolved_for_zero_game_check_count = 0
    resolved_without_logs_count = 0
    if skip_zero_game_players and ordered:
        names_with_logs, unresolved_names, names_without_logs = _player_names_with_existing_game_logs(
            ordered,
            db_path=db_path,
        )
        unresolved_for_zero_game_check_count = len(unresolved_names)
        resolved_without_logs_count = len(names_without_logs)

        filtered_names = []
        filtered_tags = []
        for name, tag in zip(ordered, source_tags):
            keep = (
                tag == "explicit"
                or name in names_with_logs
                or name in unresolved_names
            )
            if keep:
                filtered_names.append(name)
                filtered_tags.append(tag)
            else:
                skipped_zero_game_count += 1
                if len(skipped_zero_game_examples) < 15:
                    skipped_zero_game_examples.append(name)

        ordered = filtered_names
        source_tags = filtered_tags

    total_before_limit = len(ordered)
    selected_tags = list(source_tags)
    if player_limit is not None:
        limit = max(0, int(player_limit))
        ordered = ordered[:limit]
        selected_tags = selected_tags[:limit]

    summary = {
        "explicit_count": int(sum(1 for tag in selected_tags if tag == "explicit")),
        "db_count": int(sum(1 for tag in selected_tags if tag == "db")),
        "default_seed_count": int(sum(1 for tag in selected_tags if tag == "default_seed")),
        "min_topup_count": int(sum(1 for tag in selected_tags if tag == "min_topup")),
        "total_before_limit": int(total_before_limit),
        "total_selected": int(len(ordered)),
        "player_limit": (None if player_limit is None else int(max(0, int(player_limit)))),
        "limit_applied": bool(
            player_limit is not None and len(ordered) < total_before_limit
        ),
        "skip_zero_game_players": bool(skip_zero_game_players),
        "zero_game_skipped_count": int(skipped_zero_game_count),
        "zero_game_skipped_examples": skipped_zero_game_examples,
        "zero_game_resolved_count": int(resolved_without_logs_count),
        "zero_game_unresolved_count": int(unresolved_for_zero_game_check_count),
    }
    return ordered, summary


def resolve_players(
    explicit_players: Optional[list[str]] = None,
    include_db_players: bool = True,
    db_path: str = DEFAULT_DB_PATH,
    player_limit: Optional[int] = None,
    all_db_players: bool = False,
    min_players: Optional[int] = None,
    skip_zero_game_players: bool = False,
) -> list[str]:
    """Build ordered deduplicated player list for daily game-log refresh."""
    selected, _ = _build_player_selection(
        explicit_players=explicit_players,
        include_db_players=include_db_players,
        db_path=db_path,
        player_limit=player_limit,
        all_db_players=all_db_players,
        min_players=min_players,
        skip_zero_game_players=skip_zero_game_players,
    )
    return selected


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


def _run_reverse_engineering_step(
    db_path: str,
    source: str,
    start_date: Optional[str],
    end_date: Optional[str],
    stat_types: Optional[list[str]],
    books: Optional[list[str]],
    min_player_segment_rows: int,
    include_market_segments: bool,
    min_market_segment_rows: int,
    output_prefix: str,
    continuous: bool,
    poll_seconds: float,
    max_runs: Optional[int],
    max_wait_minutes: Optional[float],
    min_inferred_rows: int,
    min_book_stat_groups: int,
    min_player_segment_groups: int,
    require_stability_runs: int,
    stability_tolerance: float,
    min_group_rows_for_stability: int,
) -> dict:
    """Run reverse-engineering as either single-pass or continuous mode."""
    if continuous:
        return run_market_reverse_engineering_continuous(
            db_path=db_path,
            source=source,
            start_date=start_date,
            end_date=end_date,
            stat_types=stat_types,
            books=books,
            min_player_segment_rows=min_player_segment_rows,
            include_market_segments=include_market_segments,
            min_market_segment_rows=min_market_segment_rows,
            output_prefix=output_prefix,
            poll_seconds=poll_seconds,
            max_runs=max_runs,
            max_wait_minutes=max_wait_minutes,
            min_inferred_rows=min_inferred_rows,
            min_book_stat_groups=min_book_stat_groups,
            min_player_segment_groups=min_player_segment_groups,
            require_stability_runs=require_stability_runs,
            stability_tolerance=stability_tolerance,
            min_group_rows_for_stability=min_group_rows_for_stability,
        )

    return run_market_reverse_engineering(
        db_path=db_path,
        source=source,
        start_date=start_date,
        end_date=end_date,
        stat_types=stat_types,
        books=books,
        min_player_segment_rows=min_player_segment_rows,
        include_market_segments=include_market_segments,
        min_market_segment_rows=min_market_segment_rows,
        output_prefix=output_prefix,
    )


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
    all_db_players: bool = False,
    min_players: Optional[int] = None,
    skip_zero_game_players: bool = False,
    skip_game_logs: bool = False,
    skip_team_defense: bool = False,
    skip_odds: bool = False,
    skip_reverse_engineering: bool = False,
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
    reverse_engineering_continuous: bool = True,
    reverse_engineering_source: str = "both",
    reverse_engineering_start_date: Optional[str] = None,
    reverse_engineering_end_date: Optional[str] = None,
    reverse_engineering_stat_types: Optional[list[str]] = None,
    reverse_engineering_books: Optional[list[str]] = None,
    reverse_engineering_min_player_segment_rows: int = 8,
    reverse_engineering_include_market_segments: bool = True,
    reverse_engineering_min_market_segment_rows: int = 15,
    reverse_engineering_output_prefix: str = "market_reverse_engineering",
    reverse_engineering_poll_seconds: float = 300.0,
    reverse_engineering_max_runs: Optional[int] = None,
    reverse_engineering_max_wait_minutes: Optional[float] = None,
    reverse_engineering_min_inferred_rows: int = 25,
    reverse_engineering_min_book_stat_groups: int = 2,
    reverse_engineering_min_player_segment_groups: int = 5,
    reverse_engineering_require_stability_runs: int = 2,
    reverse_engineering_stability_tolerance: float = 0.10,
    reverse_engineering_min_group_rows_for_stability: int = 10,
    report_dir: str = DEFAULT_REPORT_DIR,
    write_report: bool = True,
) -> dict:
    """Run end-to-end daily ETL flow and return structured report."""
    season = season or _default_nba_season()
    selected_players, player_selection_summary = _build_player_selection(
        explicit_players=players,
        include_db_players=include_db_players,
        db_path=db_path,
        player_limit=player_limit,
        all_db_players=all_db_players,
        min_players=min_players,
        skip_zero_game_players=skip_zero_game_players,
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

    odds_status = steps.get("odds", {}).get("status")
    if skip_reverse_engineering:
        steps["reverse_engineering"] = {
            "status": "skipped",
            "reason": "Step disabled by flag.",
        }
    elif odds_status not in {"success", "partial_success"}:
        steps["reverse_engineering"] = {
            "status": "skipped",
            "reason": "Requires odds step to be success/partial_success.",
            "odds_status": odds_status,
        }
    else:
        reverse_step = run_with_retry(
            step_name="reverse_engineering",
            func=lambda: _run_reverse_engineering_step(
                db_path=db_path,
                source=reverse_engineering_source,
                start_date=reverse_engineering_start_date,
                end_date=reverse_engineering_end_date,
                stat_types=reverse_engineering_stat_types,
                books=reverse_engineering_books,
                min_player_segment_rows=reverse_engineering_min_player_segment_rows,
                include_market_segments=reverse_engineering_include_market_segments,
                min_market_segment_rows=reverse_engineering_min_market_segment_rows,
                output_prefix=reverse_engineering_output_prefix,
                continuous=reverse_engineering_continuous,
                poll_seconds=reverse_engineering_poll_seconds,
                max_runs=reverse_engineering_max_runs,
                max_wait_minutes=reverse_engineering_max_wait_minutes,
                min_inferred_rows=reverse_engineering_min_inferred_rows,
                min_book_stat_groups=reverse_engineering_min_book_stat_groups,
                min_player_segment_groups=reverse_engineering_min_player_segment_groups,
                require_stability_runs=reverse_engineering_require_stability_runs,
                stability_tolerance=reverse_engineering_stability_tolerance,
                min_group_rows_for_stability=(
                    reverse_engineering_min_group_rows_for_stability
                ),
            ),
            retries=max(0, retries),
            retry_delay_seconds=retry_delay_seconds,
            retry_backoff=retry_backoff,
        )
        if reverse_step["status"] == "success":
            reverse_status = str(
                reverse_step.get("result", {}).get("status", "")
            ).strip().lower()
            if reverse_status in {"max_runs_reached", "max_wait_reached", "waiting"}:
                reverse_step["status"] = "partial_success"
            elif reverse_status and reverse_status not in {"ready", "success"}:
                reverse_step["status"] = "failed"
        steps["reverse_engineering"] = reverse_step

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
        "player_selection_summary": player_selection_summary,
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
    parser.add_argument(
        "--all-db-players",
        action="store_true",
        help="Use all players currently in local players table (overrides --skip-db-players).",
    )
    parser.add_argument(
        "--min-players",
        type=int,
        default=None,
        help="Ensure at least this many players by expanding from active NBA player catalog.",
    )
    parser.add_argument(
        "--skip-zero-game-players",
        action="store_true",
        help=(
            "Skip non-explicit players with no local game_logs history. "
            "Useful after large discovery runs to avoid repeated zero-row fetches."
        ),
    )
    parser.add_argument("--player-limit", type=int, default=None,
                        help="Optional cap on total players refreshed.")
    parser.add_argument("--skip-game-logs", action="store_true")
    parser.add_argument("--skip-team-defense", action="store_true")
    parser.add_argument("--skip-odds", action="store_true")
    parser.add_argument("--skip-reverse-engineering", action="store_true")
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
    parser.add_argument(
        "--reverse-engineering-single-pass",
        action="store_true",
        help="Run reverse-engineering once instead of continuous mode.",
    )
    parser.add_argument(
        "--reverse-engineering-source",
        choices=["both", "lines", "snapshots"],
        default="both",
    )
    parser.add_argument("--reverse-engineering-start-date", default=None)
    parser.add_argument("--reverse-engineering-end-date", default=None)
    parser.add_argument("--reverse-engineering-stat-types", nargs="*", default=None)
    parser.add_argument("--reverse-engineering-books", nargs="*", default=None)
    parser.add_argument("--reverse-engineering-min-player-segment-rows", type=int, default=8)
    parser.add_argument("--reverse-engineering-skip-market-segments", action="store_true")
    parser.add_argument("--reverse-engineering-min-market-segment-rows", type=int, default=15)
    parser.add_argument("--reverse-engineering-output-prefix",
                        default="market_reverse_engineering")
    parser.add_argument("--reverse-engineering-poll-seconds", type=float, default=300.0)
    parser.add_argument("--reverse-engineering-max-runs", type=int, default=0)
    parser.add_argument("--reverse-engineering-max-wait-minutes", type=float, default=0.0)
    parser.add_argument("--reverse-engineering-min-inferred-rows", type=int, default=25)
    parser.add_argument("--reverse-engineering-min-book-stat-groups", type=int, default=2)
    parser.add_argument("--reverse-engineering-min-player-segment-groups",
                        type=int, default=5)
    parser.add_argument("--reverse-engineering-require-stability-runs",
                        type=int, default=2)
    parser.add_argument("--reverse-engineering-stability-tolerance",
                        type=float, default=0.10)
    parser.add_argument("--reverse-engineering-min-group-rows-for-stability",
                        type=int, default=10)
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
        include_db_players=(args.all_db_players or not args.skip_db_players),
        player_limit=args.player_limit,
        all_db_players=args.all_db_players,
        min_players=args.min_players,
        skip_zero_game_players=args.skip_zero_game_players,
        skip_game_logs=args.skip_game_logs,
        skip_team_defense=args.skip_team_defense,
        skip_odds=args.skip_odds,
        skip_reverse_engineering=args.skip_reverse_engineering,
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
        reverse_engineering_continuous=not args.reverse_engineering_single_pass,
        reverse_engineering_source=args.reverse_engineering_source,
        reverse_engineering_start_date=args.reverse_engineering_start_date,
        reverse_engineering_end_date=args.reverse_engineering_end_date,
        reverse_engineering_stat_types=args.reverse_engineering_stat_types,
        reverse_engineering_books=args.reverse_engineering_books,
        reverse_engineering_min_player_segment_rows=(
            args.reverse_engineering_min_player_segment_rows
        ),
        reverse_engineering_include_market_segments=(
            not args.reverse_engineering_skip_market_segments
        ),
        reverse_engineering_min_market_segment_rows=(
            args.reverse_engineering_min_market_segment_rows
        ),
        reverse_engineering_output_prefix=args.reverse_engineering_output_prefix,
        reverse_engineering_poll_seconds=args.reverse_engineering_poll_seconds,
        reverse_engineering_max_runs=(
            args.reverse_engineering_max_runs
            if args.reverse_engineering_max_runs > 0
            else None
        ),
        reverse_engineering_max_wait_minutes=(
            args.reverse_engineering_max_wait_minutes
            if args.reverse_engineering_max_wait_minutes > 0
            else None
        ),
        reverse_engineering_min_inferred_rows=(
            args.reverse_engineering_min_inferred_rows
        ),
        reverse_engineering_min_book_stat_groups=(
            args.reverse_engineering_min_book_stat_groups
        ),
        reverse_engineering_min_player_segment_groups=(
            args.reverse_engineering_min_player_segment_groups
        ),
        reverse_engineering_require_stability_runs=(
            args.reverse_engineering_require_stability_runs
        ),
        reverse_engineering_stability_tolerance=(
            args.reverse_engineering_stability_tolerance
        ),
        reverse_engineering_min_group_rows_for_stability=(
            args.reverse_engineering_min_group_rows_for_stability
        ),
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
    player_summary = report.get("player_selection_summary", {})
    if player_summary:
        print(
            "- players_by_source: "
            f"explicit={player_summary.get('explicit_count', 0)}, "
            f"db={player_summary.get('db_count', 0)}, "
            f"default_seed={player_summary.get('default_seed_count', 0)}, "
            f"min_topup={player_summary.get('min_topup_count', 0)}"
        )
        if player_summary.get("skip_zero_game_players"):
            print(
                "- players_zero_game_skipped: "
                f"{player_summary.get('zero_game_skipped_count', 0)}"
            )
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
