"""Tests for the nightly DB publish hook (DEPLOYMENT.md §14).

The git commands are injected via a fake ``git_runner`` so the suite never
touches the real repo or the network; the SQLite-facing helpers (lock check,
row counts, online backup) run against throwaway temp databases.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from nba_model.data import publish_db


def _make_db(path: str, rows: int = 3) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE game_logs (id INTEGER PRIMARY KEY, v INTEGER)")
        conn.execute("CREATE TABLE betting_lines (id INTEGER PRIMARY KEY)")
        conn.executemany(
            "INSERT INTO game_logs (v) VALUES (?)", [(i,) for i in range(rows)]
        )
        conn.commit()
    finally:
        conn.close()


class FakeGit:
    """Records git invocations and returns scripted return codes."""

    def __init__(self, returns=None):
        # returns: dict subcommand -> returncode (default 0); diff defaults to
        # returncode 1 (= staged changes present) so a commit proceeds.
        self.returns = returns or {}
        self.calls = []

    def __call__(self, args, cwd):
        self.calls.append(list(args))
        sub = args[0]
        rc = self.returns.get(sub, 1 if sub == "diff" else 0)

        class _CP:
            returncode = rc
            stdout = ""
            stderr = "boom" if rc not in (0, 1) else ""
        return _CP()


class LockCheckTests(unittest.TestCase):
    def test_unlocked_db_reports_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "x.db")
            _make_db(db)
            self.assertFalse(publish_db.db_is_locked(db))

    def test_locked_db_reports_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "x.db")
            _make_db(db)
            holder = sqlite3.connect(db)
            try:
                holder.execute("BEGIN IMMEDIATE")  # hold a RESERVED lock
                holder.execute("INSERT INTO game_logs (v) VALUES (99)")
                self.assertTrue(publish_db.db_is_locked(db, timeout_seconds=0.2))
            finally:
                holder.rollback()
                holder.close()


class RowCountTests(unittest.TestCase):
    def test_counts_present_tables_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "x.db")
            _make_db(db, rows=5)
            counts = publish_db.count_rows(db)
        self.assertEqual(counts["game_logs"], 5)
        self.assertEqual(counts["betting_lines"], 0)
        # Tables not in this skeleton DB are simply omitted.
        self.assertNotIn("predictions", counts)


class BackupTests(unittest.TestCase):
    def test_backup_creates_readable_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "x.db")
            _make_db(db, rows=7)
            dest = publish_db.backup_db(db, str(Path(tmp) / "backups"))
            self.assertTrue(Path(dest).exists())
            conn = sqlite3.connect(dest)
            try:
                n = conn.execute("SELECT COUNT(*) FROM game_logs").fetchone()[0]
            finally:
                conn.close()
        self.assertEqual(n, 7)


class RunPublishTests(unittest.TestCase):
    def test_missing_db_exits_no_db(self):
        report = publish_db.run_publish(
            db_path="/nonexistent/nope.db", git_runner=FakeGit(), log=lambda *_: None,
        )
        self.assertEqual(report["exit_code"], publish_db.EXIT_NO_DB)
        self.assertFalse(report["committed"])

    def test_locked_db_skips_without_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "x.db")
            _make_db(db)
            holder = sqlite3.connect(db)
            git = FakeGit()
            try:
                holder.execute("BEGIN IMMEDIATE")
                holder.execute("INSERT INTO game_logs (v) VALUES (1)")
                report = publish_db.run_publish(
                    db_path=db, repo_root=tmp, git_runner=git,
                    log=lambda *_: None,
                )
            finally:
                holder.rollback()
                holder.close()
        self.assertEqual(report["exit_code"], publish_db.EXIT_LOCKED)
        self.assertEqual(git.calls, [])  # never touched git

    def test_dry_run_skips_all_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "x.db")
            _make_db(db)
            git = FakeGit()
            report = publish_db.run_publish(
                db_path=db, repo_root=tmp, dry_run=True, git_runner=git,
                log=lambda *_: None,
            )
        self.assertEqual(report["exit_code"], publish_db.EXIT_OK)
        self.assertFalse(report["committed"])
        self.assertEqual(git.calls, [])
        self.assertIn("game_logs", report["row_counts"])

    def test_happy_path_commits_and_pushes(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "x.db")
            _make_db(db)
            git = FakeGit()  # diff -> rc 1 (changes staged) → commit + push
            report = publish_db.run_publish(
                db_path=db, repo_root=tmp, branch="main", remote="origin",
                backup_dir=str(Path(tmp) / "backups"), git_runner=git,
                log=lambda *_: None,
            )
            # Backup lives under tmp, so check it before the dir is torn down.
            self.assertTrue(Path(report["backup_path"]).exists())
        self.assertEqual(report["exit_code"], publish_db.EXIT_OK)
        self.assertTrue(report["committed"])
        self.assertTrue(report["pushed"])
        subcommands = [c[0] for c in git.calls]
        self.assertEqual(subcommands, ["add", "diff", "commit", "push"])

    def test_no_changes_is_clean_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "x.db")
            _make_db(db)
            git = FakeGit(returns={"diff": 0})  # nothing staged
            report = publish_db.run_publish(
                db_path=db, repo_root=tmp,
                backup_dir=str(Path(tmp) / "backups"), git_runner=git,
                log=lambda *_: None,
            )
        self.assertEqual(report["exit_code"], publish_db.EXIT_OK)
        self.assertFalse(report["committed"])
        subcommands = [c[0] for c in git.calls]
        self.assertEqual(subcommands, ["add", "diff"])  # stopped after diff

    def test_push_failure_surfaces_git_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "x.db")
            _make_db(db)
            git = FakeGit(returns={"push": 128})
            report = publish_db.run_publish(
                db_path=db, repo_root=tmp,
                backup_dir=str(Path(tmp) / "backups"), git_runner=git,
                log=lambda *_: None,
            )
        self.assertEqual(report["exit_code"], publish_db.EXIT_GIT_FAILED)
        self.assertTrue(report["committed"])
        self.assertFalse(report["pushed"])


if __name__ == "__main__":
    unittest.main()
