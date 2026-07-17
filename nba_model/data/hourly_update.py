"""Hourly NBA ETL runner.

Designed to fire every hour on the dev Mac via launchd / cron. The full
sportsbook scraping path needs a residential Chrome on
``--remote-debugging-port=9222`` to bypass Cloudflare / PerimeterX /
DataDome TLS fingerprinting, so this runner is intentionally local-host
only — it will not run in a container or in GitHub Actions.

Pipeline (in order, each step writes to the JSON report):
    1. Preflight: Playwright importable + Chrome CDP reachable on :9222
    2. Web-text ingestion (CDP) over data/config/web_text_urls.txt
    3. Browser prop parser (prizepicks / underdog / pick6 / parlayplay)
    4. Team-line parser (betmgm / caesars / draftkings / bovada / kalshi)
    5. Lightweight nba_api refresh (recent games + recent player logs)
    6. Re-derive team_priors (single-pass reverse engineering)
    7. Settle prediction outcomes (idempotent backfill)
    8. Write timestamped JSON report under nba_model/data/artifacts/hourly/

Idempotency / overlap safety:
    Acquires an fcntl flock on a lockfile so two overlapping runs can't
    stomp each other (e.g. an 02:00 run that hangs past 03:00). The
    second invocation logs a clear "already running" message and exits
    with code 75 (EX_TEMPFAIL) — launchd is configured to keep retrying
    on the regular interval.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback

try:  # POSIX (the dev Mac / launchd host) — preferred path.
    import fcntl
except ImportError:  # pragma: no cover - Windows dev box has no fcntl.
    fcntl = None
    import msvcrt
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("nba_model.hourly_update")

DEFAULT_DB_PATH = "data/database/nba_data.db"
DEFAULT_URLS_FILE = "data/config/web_text_urls.txt"
DEFAULT_REPORT_DIR = "nba_model/data/artifacts/hourly"
DEFAULT_LOG_DIR = "nba_model/data/logs"
DEFAULT_LOCKFILE = "/tmp/nba_hourly_update.lock"
DEFAULT_CHROME_PORT = 9222

EXIT_OK = 0
EXIT_LOCKED = 75            # EX_TEMPFAIL — try again next interval
EXIT_PREFLIGHT_FAILED = 78  # EX_CONFIG — fix Chrome / venv before retrying
EXIT_STEP_FAILED = 1        # at least one ETL step raised


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _try_lock(fh) -> bool:
    """Non-blocking exclusive lock. Returns True if acquired, False if held."""
    if fcntl is not None:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False
    # Windows fallback: lock a single byte at offset 0 (mandatory, per-handle).
    try:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        return True
    except OSError:
        return False


def _release_lock(fh) -> None:
    if fcntl is not None:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        return
    try:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass


@contextmanager
def _acquire_lock(lockfile: str):
    """Non-blocking advisory lock. Yields True if acquired, False if already held."""
    path = Path(lockfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    # "r+" avoids truncating a file whose byte range another handle has locked
    # (Windows refuses the truncate); touch first so the file always exists.
    path.touch(exist_ok=True)
    fh = open(path, "r+", encoding="utf-8")
    try:
        if not _try_lock(fh):
            yield False
            return
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(f"pid={os.getpid()} started={_utc_now_iso()}\n")
            fh.flush()
        except OSError:
            pass
        try:
            yield True
        finally:
            _release_lock(fh)
    finally:
        fh.close()


def _configure_logging(log_dir: str) -> str:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"hourly_update_{ts}.log"
    handlers = [logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    return str(log_path)


def _write_report(report: dict, report_dir: str) -> str:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out / f"hourly_update_{ts}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _record_step(report: dict, name: str, fn, *args, **kwargs) -> bool:
    """Run a step, catch exceptions, attach result to report, return success."""
    started = time.monotonic()
    try:
        result = fn(*args, **kwargs)
        report["steps"][name] = {
            "ok": True,
            "duration_s": round(time.monotonic() - started, 2),
            "result": result,
        }
        return True
    except Exception as exc:  # noqa: BLE001 — surfacing every step failure
        report["steps"][name] = {
            "ok": False,
            "duration_s": round(time.monotonic() - started, 2),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        logger.error("step '%s' failed: %s", name, exc)
        report["failed_steps"].append(name)
        return False


def _run_preflight(
    chrome_port: int,
    chrome_host: str,
    require_playwright: bool,
) -> dict:
    """Verify Chrome + Playwright are wired correctly. Raises on failure."""
    from nba_model.model.web_text_ingestion import (
        check_chrome_cdp_reachable,
        playwright_is_available,
    )

    playwright_ok = playwright_is_available()
    if require_playwright and not playwright_ok:
        raise RuntimeError(
            "Playwright is not importable in this interpreter. "
            "The hourly runner MUST use the project venv: "
            "run as `.venv/bin/python3 -m nba_model.data.hourly_update`. "
            "Install browsers with `.venv/bin/python3 -m playwright install chromium`."
        )

    chrome = check_chrome_cdp_reachable(chrome_port, host=chrome_host)
    if not chrome["ok"]:
        raise RuntimeError(
            f"{chrome['error']}. Start Chrome with:\n"
            f"  open -na 'Google Chrome' --args "
            f"--remote-debugging-port={chrome_port} "
            "--user-data-dir=/tmp/pp-chrome-profile\n"
            "Then log in to PrizePicks/Underdog/DK/BetMGM/Caesars in that "
            "window before the next hourly tick."
        )

    return {
        "playwright_available": playwright_ok,
        "chrome": chrome,
    }


def _run_web_text(
    db_path: str,
    urls_file: str,
    chrome_port: int,
    browser_auth_state_file: Optional[str],
) -> dict:
    from nba_model.model.web_text_ingestion import (
        fetch_and_store_web_text,
        load_urls_from_file,
    )

    urls = load_urls_from_file(urls_file)
    if not urls:
        return {"urls": 0, "skipped": "empty url list"}
    summary = fetch_and_store_web_text(
        urls=urls,
        db_path=db_path,
        # Hourly cadence — never block on the daily 24h window.
        min_hours_between_polls=0.0,
        force_poll=True,
        browser_auth_state_file=browser_auth_state_file,
        browser_user_data_dir=None,
        chrome_debug_port=chrome_port,
    )
    summary["urls"] = len(urls)
    return summary


def _run_browser_prop_parser(db_path: str) -> dict:
    from nba_model.model.browser_prop_parser import parse_and_store_web_prop_cards

    return parse_and_store_web_prop_cards(
        db_path=db_path,
        source_urls=None,            # any recent snapshot
        max_snapshots_per_url=1,
        max_total_snapshots=20,
        min_parse_confidence=0.2,
    )


def _run_team_line_parser(db_path: str) -> dict:
    from nba_model.model.team_line_parser import parse_and_store_web_team_lines

    return parse_and_store_web_team_lines(
        db_path=db_path,
        max_snapshots_per_url=1,
        max_total_snapshots=20,
    )


def _run_game_log_refresh(db_path: str, max_players: int) -> dict:
    """Refresh recent NBA game logs for tracked players.

    Touches the lightweight nba_api endpoint; rate-limited to avoid the
    20-req-per-minute soft cap.
    """
    from nba_model.data.data_loader import DataLoader
    from nba_model.data.database.db_manager import DatabaseManager

    db = DatabaseManager(db_path=db_path)
    # Players that already have history in the DB — same set the daily ETL uses.
    # game_logs is keyed by player_id (no name column), so resolve the name via
    # the scraper-synced active-players ref, falling back to the players table.
    rows = db.conn.execute(
        """
        SELECT DISTINCT COALESCE(r.player_name, p.name) AS name
        FROM game_logs gl
        LEFT JOIN nba_active_players_ref r ON r.player_id = gl.player_id
        LEFT JOIN players p ON p.player_id = gl.player_id
        WHERE COALESCE(r.player_name, p.name) IS NOT NULL
        ORDER BY name
        LIMIT ?
        """,
        (int(max_players),),
    ).fetchall()
    loader = DataLoader()
    refreshed = 0
    failed: list[dict] = []
    for (name,) in rows:
        try:
            loader.load_player_data(name, n_games=10, force_refresh=True)
            refreshed += 1
            time.sleep(0.6)
        except Exception as exc:  # noqa: BLE001 — single-player failures shouldn't kill the run
            failed.append({"player": name, "error": str(exc)})
    return {"players_refreshed": refreshed, "failures": failed}


def _run_players_table_sync(db_path: str) -> dict:
    """Keep ``players.team`` populated from the latest game_logs row.

    Cheap to run (one CTE + an UPDATE), and it's the only step that catches
    mid-season trades + draft activations between the daily ETL passes.
    """
    from nba_model.data.database.db_manager import DatabaseManager

    db = DatabaseManager(db_path=db_path)
    return db.sync_players_table()


def _run_reverse_engineering(db_path: str) -> dict:
    """Single-pass team-priors derivation from the latest cross-book consensus."""
    from nba_model.model.team_line_reverse_engineering import (
        derive_team_priors_from_consensus,
    )

    return derive_team_priors_from_consensus(
        since_hours=6.0,  # hourly run — only look at the recent window
        min_books=2,
        db_path=db_path,
    )


def _run_outcome_settlement(db_path: str) -> dict:
    from nba_model.data.database.db_manager import DatabaseManager

    db = DatabaseManager(db_path=db_path)
    return db.backfill_predictions_outcomes()


def _run_prediction_recompute(db_path: str) -> dict:
    """Re-score every distinct (player, stat_type) currently quoted in
    ``betting_lines`` for today's game date against the latest model — so the
    deployed model is genuinely "updated every hour."

    Returns counts; per-prediction errors are aggregated, not raised.
    """
    from nba_model.data.database.db_manager import DatabaseManager
    from nba_model.model.prop_board import (
        DEFAULT_N_GAMES,
        DEFAULT_ROLLING_WINDOW,
        _build_board_lines,
        _build_player_history,
        _fetch_betting_lines_for_game,
    )

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = _fetch_betting_lines_for_game(
        db_path=db_path,
        game_date=today,
        stat_types=["points", "assists", "rebounds", "pra"],
        home_team=None,
        away_team=None,
        books=None,
    )
    if rows.empty:
        return {"scored": 0, "target_date": today, "skipped": "no betting_lines for today"}

    histories: dict = {}
    history_failures: list[dict] = []
    for player_name in sorted(rows["player_name"].unique()):
        try:
            histories[player_name] = _build_player_history(
                player_name=player_name,
                n_games=DEFAULT_N_GAMES,
                rolling_window=DEFAULT_ROLLING_WINDOW,
                db_path=db_path,
            )
        except Exception as exc:  # noqa: BLE001 — single-player failure shouldn't kill the run
            history_failures.append({"player": player_name, "error": str(exc)})

    # Blend the cross-book team priors (pace + implied team total) for the
    # whole slate so the hourly-refreshed projections share the market signal.
    with DatabaseManager(db_path=db_path) as db:
        team_priors_map = db.get_team_prior_inputs_map()

    board_lines = _build_board_lines(
        rows=rows,
        player_histories=histories,
        rolling_window=DEFAULT_ROLLING_WINDOW,
        team_priors=team_priors_map,
    )

    # Persist the board lines as predictions so the predictions table widens
    # beyond points (points/assists/rebounds/pra) and the deployed model's
    # predicted_mean is actually stored. Idempotent for today's slate:
    # re-delete this (player, date, stat) before inserting so repeated hourly
    # ticks don't pile up duplicate rows.
    name_to_id: dict = {}
    for r in rows.itertuples(index=False):
        name_to_id.setdefault(str(r.player_name), int(r.player_id))
    persisted = _persist_predictions(db_path, board_lines, name_to_id, today)

    return {
        "scored": len(board_lines),
        "predictions_persisted": persisted,
        "target_date": today,
        "history_failures": history_failures,
        "players": len(histories),
        "teams_with_priors": len(team_priors_map),
    }


def _run_bet_log_settlement(db_path: str, calibration_source: str = "bet_log") -> dict:
    """Settle pending ``bet_log`` rows and refresh the calibration artifact.

    Flag-gated (default OFF) paper-trading maintenance: grade any bet_log picks
    whose games have landed (filling ``clv_delta`` from line snapshots when
    present), then regenerate the reliability/Brier report so the calibration
    artifact stays current. No execution — measurement only."""
    from nba_model.data.database.db_manager import DatabaseManager
    from nba_model.evaluation.calibration_report import run_calibration_report

    with DatabaseManager(db_path=db_path) as db:
        settle = db.settle_bet_log()
    calibration = run_calibration_report(db_path=db_path, source=calibration_source)
    return {"settle": settle, "calibration": calibration}


def _persist_predictions(db_path, board_lines, name_to_id, target_date) -> int:
    """Write board lines into the predictions table (idempotent per slate)."""
    if not board_lines:
        return 0
    from nba_model.data.database.db_manager import DatabaseManager
    persisted = 0
    with DatabaseManager(db_path=db_path) as db:
        for line in board_lines:
            pid = name_to_id.get(line.player_name)
            if pid is None:
                continue
            db.conn.execute(
                "DELETE FROM predictions WHERE player_id = ? "
                "AND date(game_date) = date(?) AND lower(stat_type) = lower(?)",
                (int(pid), target_date, str(line.stat_type)),
            )
            db.insert_prediction({
                "player_id": int(pid),
                "game_date": target_date,
                "stat_type": str(line.stat_type),
                "predicted_mean": float(line.mu),
                "predicted_std": float(line.sigma),
                "prob_over": float(line.prob_over),
                "line_value": float(line.line_value),
                "book_odds": line.over_odds,
                "expected_value": line.ev_over,
            })
            persisted += 1
    return persisted


def run_hourly_update(
    *,
    db_path: str = DEFAULT_DB_PATH,
    urls_file: str = DEFAULT_URLS_FILE,
    report_dir: str = DEFAULT_REPORT_DIR,
    chrome_port: int = DEFAULT_CHROME_PORT,
    chrome_host: str = "127.0.0.1",
    browser_auth_state_file: Optional[str] = None,
    max_players: int = 25,
    require_playwright: bool = True,
    skip_recompute: bool = False,
    settle_bet_log: bool = False,
    alert_webhook_url: Optional[str] = None,
) -> dict:
    """Execute the hourly pipeline and return the JSON report dict."""
    from nba_model.data.etl_alerts import build_alert, maybe_send_alert
    report = {
        "started_at": _utc_now_iso(),
        "ended_at": None,
        "ok": False,
        "steps": {},
        "failed_steps": [],
        "exit_code": EXIT_OK,
    }

    try:
        preflight = _run_preflight(chrome_port, chrome_host, require_playwright)
        report["steps"]["preflight"] = {"ok": True, "result": preflight}
    except Exception as exc:  # noqa: BLE001 — preflight failure is fatal
        report["steps"]["preflight"] = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        report["failed_steps"].append("preflight")
        report["ended_at"] = _utc_now_iso()
        report["exit_code"] = EXIT_PREFLIGHT_FAILED
        report["alert"] = build_alert(report)
        report["report_path"] = _write_report(report, report_dir)
        maybe_send_alert(report, alert_webhook_url)
        logger.error("preflight failed: %s", exc)
        return report

    # Steps run in order; later steps depend on earlier outputs landing in DB.
    _record_step(
        report, "web_text",
        _run_web_text, db_path, urls_file, chrome_port, browser_auth_state_file,
    )
    _record_step(report, "browser_prop_parser", _run_browser_prop_parser, db_path)
    _record_step(report, "team_line_parser", _run_team_line_parser, db_path)
    _record_step(report, "game_log_refresh", _run_game_log_refresh, db_path, max_players)
    _record_step(report, "players_table_sync", _run_players_table_sync, db_path)
    _record_step(report, "reverse_engineering", _run_reverse_engineering, db_path)
    _record_step(report, "outcome_settlement", _run_outcome_settlement, db_path)
    if not skip_recompute:
        _record_step(report, "prediction_recompute", _run_prediction_recompute, db_path)
    # Optional paper-trading maintenance (default OFF): settle bet_log +
    # refresh the calibration artifact. Kept out of the default hourly path so
    # it can't slow or destabilize the core ETL until explicitly enabled.
    if settle_bet_log:
        _record_step(report, "bet_log_settlement", _run_bet_log_settlement, db_path)

    report["ended_at"] = _utc_now_iso()
    report["ok"] = not report["failed_steps"]
    report["exit_code"] = EXIT_OK if report["ok"] else EXIT_STEP_FAILED
    report["alert"] = build_alert(report)
    report["report_path"] = _write_report(report, report_dir)
    maybe_send_alert(report, alert_webhook_url)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hourly NBA ETL runner")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--urls-file", default=DEFAULT_URLS_FILE)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--lockfile", default=DEFAULT_LOCKFILE)
    parser.add_argument("--chrome-port", type=int, default=DEFAULT_CHROME_PORT)
    parser.add_argument("--chrome-host", default="127.0.0.1")
    parser.add_argument("--browser-auth-state-file", default=None)
    parser.add_argument("--max-players", type=int, default=25)
    parser.add_argument(
        "--no-require-playwright",
        action="store_true",
        help="Skip the Playwright preflight (useful for unit tests; do NOT use in prod).",
    )
    parser.add_argument("--skip-recompute", action="store_true")
    parser.add_argument(
        "--settle-bet-log", action="store_true",
        help="Also settle pending bet_log rows + refresh the calibration "
             "artifact (paper-trading maintenance; default OFF).",
    )
    parser.add_argument(
        "--alert-webhook-url", default=None,
        help="POST a JSON alert here when the run fails or partially fails.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    log_path = _configure_logging(args.log_dir)
    logger.info("Hourly update starting (log=%s, pid=%s)", log_path, os.getpid())

    with _acquire_lock(args.lockfile) as acquired:
        if not acquired:
            logger.error("Another hourly run already holds the lock at %s", args.lockfile)
            return EXIT_LOCKED
        report = run_hourly_update(
            db_path=args.db_path,
            urls_file=args.urls_file,
            report_dir=args.report_dir,
            chrome_port=args.chrome_port,
            chrome_host=args.chrome_host,
            browser_auth_state_file=args.browser_auth_state_file,
            max_players=args.max_players,
            require_playwright=not args.no_require_playwright,
            skip_recompute=args.skip_recompute,
            settle_bet_log=args.settle_bet_log,
            alert_webhook_url=args.alert_webhook_url,
        )

    logger.info(
        "Hourly update done. ok=%s failed_steps=%s report=%s",
        report["ok"], report["failed_steps"], report.get("report_path"),
    )
    return int(report["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
