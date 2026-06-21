"""Publish the ETL'd ``nba_data.db`` to git for the read-only cloud deploy.

Implements ``docs/DEPLOYMENT.md`` §14 step 2: on a clean hourly run the
always-on scraping host commits the refreshed database to ``main`` (or a
configured data branch) and pushes; Streamlit Cloud redeploys on push and
serves the new snapshot.

Safety rails (the reason this is a module and not a one-line ``git commit``):
    * Refuse to publish a database another process is mid-write on — a
      locked DB means the ETL is still running, and a half-written page
      would ship a corrupt snapshot.
    * Take a consistent online ``.backup`` copy before touching git, so a
      bad push is always recoverable.
    * Log row counts for the headline tables so the launchd log shows the
      snapshot actually grew (a flat count run-over-run is a silent-failure
      smell).
    * ``--dry-run`` prints exactly what would happen and skips every
      mutating git command.

The bash wrapper ``scripts/publish_db.sh`` resolves the project venv and
re-execs this module so cron / launchd can't pick up system python.
"""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

DEFAULT_DB_PATH = "data/database/nba_data.db"
DEFAULT_BACKUP_DIR = "data/database/backups"
DEFAULT_BRANCH = "main"
DEFAULT_REMOTE = "origin"

# Tables whose row counts we surface in the publish log. Missing tables are
# skipped (older DBs / partial schemas) rather than erroring.
REPORTED_TABLES: tuple[str, ...] = (
    "game_logs",
    "games",
    "betting_lines",
    "betting_line_snapshots",
    "web_prop_cards",
    "web_team_lines",
    "predictions",
    "team_priors",
)

EXIT_OK = 0
EXIT_NO_DB = 2
EXIT_LOCKED = 3
EXIT_GIT_FAILED = 4
EXIT_NOTHING_TO_PUBLISH = 0  # not an error — just nothing changed


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def db_is_locked(db_path: str, timeout_seconds: float = 1.0) -> bool:
    """True if another connection holds a write lock on the database.

    Attempts a ``BEGIN IMMEDIATE`` (acquires a RESERVED lock) with a short
    busy timeout. ``sqlite3.OperationalError: database is locked`` means an
    ETL writer is active and we must not publish.
    """
    conn = sqlite3.connect(db_path, timeout=timeout_seconds)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.rollback()
        return False
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc).lower() or "busy" in str(exc).lower():
            return True
        raise
    finally:
        conn.close()


def count_rows(db_path: str, tables: Sequence[str] = REPORTED_TABLES) -> dict:
    """Row count per table; tables absent from the schema are omitted."""
    counts: dict[str, int] = {}
    conn = sqlite3.connect(db_path)
    try:
        present = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for table in tables:
            if table not in present:
                continue
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            counts[table] = int(row[0]) if row else 0
    finally:
        conn.close()
    return counts


def backup_db(db_path: str, backup_dir: str) -> str:
    """Take a consistent online ``.backup`` snapshot. Returns the backup path."""
    out_dir = Path(backup_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = out_dir / f"nba_data_{stamp}.db"
    src = sqlite3.connect(db_path)
    try:
        dst = sqlite3.connect(str(dest))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    return str(dest)


def _run_git(args: Sequence[str], cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def run_publish(
    db_path: str = DEFAULT_DB_PATH,
    *,
    branch: str = DEFAULT_BRANCH,
    remote: str = DEFAULT_REMOTE,
    backup_dir: str = DEFAULT_BACKUP_DIR,
    repo_root: Optional[str] = None,
    dry_run: bool = False,
    message: Optional[str] = None,
    git_runner: Callable[[Sequence[str], str], subprocess.CompletedProcess] = _run_git,
    log: Callable[[str], None] = print,
) -> dict:
    """Publish the DB to git. Returns a report dict; never raises on a clean
    no-op or a held lock (caller maps ``exit_code`` to the process exit)."""
    report: dict = {
        "db_path": db_path,
        "dry_run": bool(dry_run),
        "timestamp": _utc_stamp(),
        "row_counts": {},
        "backup_path": None,
        "committed": False,
        "pushed": False,
        "exit_code": EXIT_OK,
        "message": None,
    }

    db = Path(db_path)
    if not db.exists():
        report["exit_code"] = EXIT_NO_DB
        report["message"] = f"database not found: {db_path}"
        log(f"FATAL: {report['message']}")
        return report

    if db_is_locked(db_path):
        report["exit_code"] = EXIT_LOCKED
        report["message"] = (
            f"{db_path} is locked — an ETL writer is still active. "
            "Refusing to publish a half-written snapshot; retry next tick."
        )
        log(f"SKIP: {report['message']}")
        return report

    report["row_counts"] = count_rows(db_path)
    for table, n in report["row_counts"].items():
        log(f"  {table}: {n} rows")

    # Default: the project root is three levels up from data/database/<db>.
    root = repo_root or str(db.resolve().parent.parent.parent)

    commit_msg = message or f"etl: db snapshot {report['timestamp']}"

    if dry_run:
        report["backup_path"] = "(dry-run: backup skipped)"
        report["message"] = (
            f"dry-run: would back up, `git add -f {db_path}`, "
            f"commit '{commit_msg}', and push to {remote}/{branch}"
        )
        log(report["message"])
        return report

    report["backup_path"] = backup_db(db_path, backup_dir)
    log(f"  backup: {report['backup_path']}")

    # Stage the DB; bail cleanly if nothing actually changed. ``-f`` is needed
    # because data/database/*.db is gitignored — publishing it is deliberate.
    add = git_runner(["add", "-f", "--", db_path], root)
    if add.returncode != 0:
        report["exit_code"] = EXIT_GIT_FAILED
        report["message"] = f"git add failed: {add.stderr.strip()}"
        log(f"FATAL: {report['message']}")
        return report

    diff = git_runner(["diff", "--cached", "--quiet", "--", db_path], root)
    if diff.returncode == 0:
        report["message"] = "no DB changes to publish (snapshot unchanged)"
        report["exit_code"] = EXIT_NOTHING_TO_PUBLISH
        log(report["message"])
        return report

    commit = git_runner(["commit", "-m", commit_msg, "--", db_path], root)
    if commit.returncode != 0:
        report["exit_code"] = EXIT_GIT_FAILED
        report["message"] = f"git commit failed: {commit.stderr.strip()}"
        log(f"FATAL: {report['message']}")
        return report
    report["committed"] = True
    log(f"  committed: {commit_msg}")

    push = git_runner(["push", remote, branch], root)
    if push.returncode != 0:
        report["exit_code"] = EXIT_GIT_FAILED
        report["message"] = f"git push failed: {push.stderr.strip()}"
        log(f"FATAL: {report['message']}")
        return report
    report["pushed"] = True
    report["message"] = f"published to {remote}/{branch}"
    log(f"  pushed to {remote}/{branch}")
    return report


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Publish nba_data.db to git for the cloud deploy (DEPLOYMENT.md §14).",
    )
    p.add_argument("--db", default=DEFAULT_DB_PATH, help="path to the SQLite DB")
    p.add_argument("--branch", default=DEFAULT_BRANCH, help="branch to push")
    p.add_argument("--remote", default=DEFAULT_REMOTE, help="git remote")
    p.add_argument("--backup-dir", default=DEFAULT_BACKUP_DIR,
                   help="directory for the pre-publish online backup")
    p.add_argument("--repo-root", default=None,
                   help="git repo root (defaults to the DB's project root)")
    p.add_argument("--message", default=None, help="override the commit message")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would happen; skip all mutating git commands")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    report = run_publish(
        db_path=args.db,
        branch=args.branch,
        remote=args.remote,
        backup_dir=args.backup_dir,
        repo_root=args.repo_root,
        dry_run=args.dry_run,
        message=args.message,
    )
    return int(report["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
