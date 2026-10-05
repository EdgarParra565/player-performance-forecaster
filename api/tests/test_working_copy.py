"""Read-only source DB (e.g. ./data/database:/data:ro): the API serves a
working copy so the data layer's open-time migrations never write the host
file (api/working_copy.py)."""
import hashlib
import os
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import working_copy
from api.tests.test_api_phase2 import _seed


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class WorkingCopyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.mount = root / "mount"
        self.mount.mkdir()
        self.src = self.mount / "nba_data.db"
        _seed(str(self.src))
        self.cache = root / "cache"
        self.env = patch.dict(os.environ, {"NBA_DB_PATH": str(self.src),
                                           "API_DB_CACHE_DIR": str(self.cache),
                                           "API_DB_WORKING_COPY": "auto"})
        self.env.start()
        working_copy.reset()
        from api.main import app
        self.client = TestClient(app)

    def tearDown(self):
        self._make_ro(False)
        self.env.stop()
        working_copy.reset()
        self._tmp.cleanup()

    def _make_ro(self, ro: bool):
        file_mode = stat.S_IRUSR | stat.S_IRGRP | (0 if ro else stat.S_IWUSR)
        dir_mode = stat.S_IRUSR | stat.S_IXUSR | (0 if ro else stat.S_IWUSR)
        if self.src.exists():
            os.chmod(self.src, file_mode)
        os.chmod(self.mount, dir_mode)

    def test_read_only_source_is_served_via_copy_and_never_written(self):
        self._make_ro(True)
        before = _sha(self.src)
        r = self.client.get("/api/slate/kpis")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertGreater(r.json()["players_tracked"], 0)
        h = self.client.get("/api/health")
        self.assertEqual(h.status_code, 200)
        self.assertEqual(h.json()["db_state"], "ok")
        self.assertEqual(_sha(self.src), before)            # host file untouched
        self.assertTrue((self.cache / "nba_data.db").is_file())
        self.assertEqual(sorted(p.name for p in self.mount.iterdir()), ["nba_data.db"])  # no journals

    def test_copy_refreshes_when_source_changes(self):
        self._make_ro(True)
        n1 = self.client.get("/api/slate/kpis").json()["players_tracked"]
        self._make_ro(False)
        with sqlite3.connect(self.src) as conn:
            conn.execute("DELETE FROM game_logs")
        os.utime(self.src, (1, 1))  # force a distinct mtime signature
        self._make_ro(True)
        n2 = self.client.get("/api/slate/kpis").json()["players_tracked"]
        self.assertGreater(n1, 0)
        self.assertEqual(n2, 0)

    def test_writable_source_is_used_directly(self):
        self.assertEqual(working_copy.serving_path(str(self.src)), str(self.src))
        self.client.get("/api/slate/kpis")
        self.assertFalse(self.cache.exists())

    def test_mode_overrides(self):
        with patch.dict(os.environ, {"API_DB_WORKING_COPY": "always"}):
            self.assertEqual(working_copy.serving_path(str(self.src)), str(self.cache / "nba_data.db"))
        self._make_ro(True)
        with patch.dict(os.environ, {"API_DB_WORKING_COPY": "never"}):
            self.assertEqual(working_copy.serving_path(str(self.src)), str(self.src))


if __name__ == "__main__":
    unittest.main()
