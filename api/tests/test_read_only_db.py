"""The API serves the mounted file READ-ONLY, one connection per request
(api/db.py over DatabaseManager(path, read_only=True))."""
import hashlib
import os
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import db
from api.tests.test_api_phase2 import LEBRON_ID, _seed
from nba_model.data.database import db_manager as dbm
from nba_model.visualization import player_charts as pc

ENDPOINTS = (
    "/api/health",
    "/api/meta",
    "/api/slate/kpis",
    "/api/slate/recent-games?n=3",
    "/api/slate/edges?model_mode=chart_mean",
    "/api/players/search?q=leb",
    f"/api/players/{LEBRON_ID}?stat=points",
    f"/api/players/{LEBRON_ID}/line-movement?stat=points",
    "/api/cross-book",
    "/api/teams/LAL/chart",
    "/api/paper-trades",
    "/api/paper-trades/calibration?source=predictions",
)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class _OpenCounter:
    """Records every real DatabaseManager construction (path, read_only)."""

    def __init__(self):
        self.calls = []
        self._orig = dbm.DatabaseManager.__init__

    def __enter__(self):
        counter = self

        def wrapped(inst, db_path="data/database/nba_data.db", read_only=False):
            counter.calls.append((str(Path(db_path).resolve()), bool(read_only)))
            return counter._orig(inst, db_path, read_only=read_only)

        self._p = patch.object(dbm.DatabaseManager, "__init__", wrapped)
        self._p.start()
        return self

    def __exit__(self, *exc):
        self._p.stop()


class ReadOnlyServingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.mount = root / "mount"
        self.mount.mkdir()
        self.src = self.mount / "nba_data.db"
        _seed(str(self.src))  # writer open: fully migrated file, like the live DB
        self.env = patch.dict(os.environ, {"NBA_DB_PATH": str(self.src)})
        self.env.start()
        from api.main import app
        self.client = TestClient(app)

    def tearDown(self):
        self._ro(False)
        self.env.stop()
        self._tmp.cleanup()

    def _ro(self, ro: bool):
        if self.src.exists():
            os.chmod(self.src, stat.S_IRUSR | stat.S_IRGRP | (0 if ro else stat.S_IWUSR))
        os.chmod(self.mount, stat.S_IRUSR | stat.S_IXUSR | (0 if ro else stat.S_IWUSR))

    def test_one_read_only_open_per_request_on_the_mount_itself(self):
        for path in ENDPOINTS:
            with _OpenCounter() as c:
                r = self.client.get(path)
            self.assertEqual(r.status_code, 200, (path, r.text[:200]))
            self.assertEqual(c.calls, [(str(self.src.resolve()), True)], path)

    def test_parlay_post_also_single_open(self):
        body = {"legs": [{"player_id": LEBRON_ID, "stat": "points", "line": 18.5, "side": "over"},
                         {"player_id": LEBRON_ID, "stat": "rebounds", "line": 7.5, "side": "over"}]}
        with _OpenCounter() as c:
            self.client.post("/api/parlay/price", json=body)
        self.assertEqual(c.calls, [(str(self.src.resolve()), True)])

    def test_read_only_mount_is_served_and_never_written(self):
        self._ro(True)
        before = (_sha(self.src), self.src.stat().st_mtime_ns)
        for path in ENDPOINTS:
            self.assertEqual(self.client.get(path).status_code, 200, path)
        self.assertEqual((_sha(self.src), self.src.stat().st_mtime_ns), before)
        self.assertEqual([p.name for p in self.mount.iterdir()], ["nba_data.db"])  # no journals
        self.assertFalse(Path("/tmp/nba-flagship/nba_data.db").exists())  # no scratch copy

    def test_writes_through_the_shared_connection_fail(self):
        with db.read_only_session(str(self.src)):
            borrowed = pc.DatabaseManager(db_path=str(self.src))
            self.assertEqual(db.current_path(), self.src.resolve())
            with self.assertRaises(sqlite3.OperationalError):
                borrowed.conn.execute("CREATE TABLE should_not_exist (x INT)")
            borrowed.close()  # no-op: the session still works afterwards
            self.assertGreater(borrowed.conn.execute("SELECT COUNT(*) FROM game_logs").fetchone()[0], 0)
        with sqlite3.connect(self.src) as conn:
            names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
        self.assertNotIn("should_not_exist", names)

    def test_outside_a_request_the_name_is_the_normal_writer(self):
        other = Path(self._tmp.name) / "etl.db"
        mgr = pc.DatabaseManager(db_path=str(other))
        try:
            self.assertIsInstance(mgr, dbm.DatabaseManager)
            self.assertFalse(mgr.read_only)
        finally:
            mgr.close()
        self.assertTrue(other.is_file())  # writer creates + migrates, as the ETL expects


class NotReadyStatesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        from api.main import app
        self.client = TestClient(app)

    def tearDown(self):
        os.environ.pop("NBA_DB_PATH", None)
        self._tmp.cleanup()

    def _use(self, p: Path):
        os.environ["NBA_DB_PATH"] = str(p)

    def test_unmigrated_file_is_its_own_code_and_is_not_migrated(self):
        p = self.root / "old.db"
        _seed(str(p))
        with sqlite3.connect(p) as conn:  # simulate a file from before the 2026-10-05 migration
            conn.execute("ALTER TABLE web_team_lines DROP COLUMN game_date")
        before = _sha(p)
        self._use(p)
        h = self.client.get("/api/health")
        self.assertEqual(h.status_code, 503)
        self.assertEqual((h.json()["db_state"], h.json()["code"]), ("unmigrated", "db_unmigrated"))
        r = self.client.get("/api/slate/kpis")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["code"], "db_unmigrated")
        self.assertIn("needs migrating", r.json()["detail"])
        self.assertNotIn(str(self.root), r.text + h.text)  # no paths leak
        self.assertEqual(_sha(p), before)  # nothing papered over / migrated

    def test_missing_and_empty_stay_not_mounted_and_create_nothing(self):
        mount = self.root / "data"
        mount.mkdir()
        self._use(mount / "nba_data.db")
        self.assertEqual(self.client.get("/api/meta").json()["code"], "db_not_mounted")
        self.assertEqual(list(mount.iterdir()), [])
        empty = mount / "nba_data.db"
        empty.write_bytes(b"")
        self.assertEqual(self.client.get("/api/health").json()["code"], "db_not_mounted")
        self.assertEqual(empty.stat().st_size, 0)

    def test_foreign_sqlite_file_is_invalid(self):
        p = self.root / "foreign.db"
        with sqlite3.connect(p) as conn:
            conn.execute("CREATE TABLE recipes (name TEXT)")
        self._use(p)
        r = self.client.get("/api/meta")
        self.assertEqual((r.status_code, r.json()["code"]), (503, "db_invalid"))

    def test_classify(self):
        E = dbm.DatabaseNotReadyError
        self.assertEqual(db.classify(E("database file not found: /x")), "not_mounted")
        self.assertEqual(db.classify(E("database file is empty: /x")), "not_mounted")
        self.assertEqual(db.classify(E("not a readable SQLite database: /x: boom")), "invalid")
        self.assertEqual(db.classify(E(
            "database at /x is incomplete (missing columns: web_team_lines.game_date) — open it")), "unmigrated")
        self.assertEqual(db.classify(E(
            "database at /x is incomplete (missing tables: bet_log) — open it")), "unmigrated")
        self.assertEqual(db.classify(E(
            "database at /x is incomplete (missing tables: game_logs, games; missing columns: a.b)")), "invalid")


if __name__ == "__main__":
    unittest.main()
