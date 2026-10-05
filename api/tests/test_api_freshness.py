"""Freshness uses last-SEEN, not last-CHANGED (notes.txt handoff).

web_prop_cards is change-only: ``observed_at_utc`` moves only when a line
changes, while unchanged re-sights bump ``last_seen_at_utc``. A line posted
5 days ago and re-seen an hour ago is still live and must count.
"""
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from nba_model.data.database.db_manager import DatabaseManager
from api import services
from api.tests.test_api_phase2 import _card, _seed


def _ts(dt: datetime) -> str:
    return dt.isoformat()


class FreshnessTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "nba.db")
        _seed(self.db_path)
        now = datetime.now(timezone.utc)
        self.old = now - timedelta(days=5)
        self.seen = now - timedelta(hours=1)
        with DatabaseManager(db_path=self.db_path) as db:
            # Clear the seed's fresh cards so only our two rows are in play.
            db.conn.execute("DELETE FROM web_prop_cards")
            db.insert_web_prop_cards([
                _card("Underdog", "LeBron James", "points", 24.5, "over", _ts(self.old), 91),
                _card("PrizePicks", "LeBron James", "points", 25.5, "over", _ts(self.old), 92),
            ])
            # Underdog's unchanged line was re-sighted an hour ago; PrizePicks' wasn't.
            db.conn.execute(
                "UPDATE web_prop_cards SET last_seen_at_utc = ? WHERE book = 'Underdog'",
                (_ts(self.seen),))
            db.conn.execute(
                "UPDATE web_prop_cards SET last_seen_at_utc = observed_at_utc WHERE book = 'PrizePicks'")
            db.conn.execute("DELETE FROM betting_lines")  # isolate the card timestamps
            db.conn.commit()
        self.env = {"NBA_DB_PATH": self.db_path}
        os.environ.update(self.env)
        from api.main import app
        self.client = TestClient(app)

    def tearDown(self):
        os.environ.pop("NBA_DB_PATH", None)
        self._tmp.cleanup()

    def test_unchanged_resight_inside_window_counts(self):
        kpis = self.client.get("/api/slate/kpis").json()
        # Only the re-sighted line is "recent"; the stale one is not.
        self.assertEqual(kpis["prop_lines_recent"], 1)

    def test_freshest_scrape_uses_last_seen(self):
        kpis = self.client.get("/api/slate/kpis").json()
        freshest = datetime.fromisoformat(kpis["freshest_scrape_utc"])
        self.assertLess(abs((freshest - self.seen).total_seconds()), 2)
        health = self.client.get("/api/health").json()
        self.assertEqual(health["freshest_scrape_utc"], kpis["freshest_scrape_utc"])

    def test_falls_back_when_column_absent(self):
        # Older DB without last_seen_at_utc (e.g. a read-only mount where the
        # data layer couldn't migrate): the query must still work.
        legacy = Path(self._tmp.name) / "legacy.db"
        with sqlite3.connect(legacy) as conn:
            conn.execute("CREATE TABLE web_prop_cards (observed_at_utc TEXT)")
            conn.execute("INSERT INTO web_prop_cards VALUES (?)", (_ts(self.seen),))

        class _Db:  # minimal stand-in exposing .conn like DatabaseManager
            def __init__(self, path):
                self.conn = sqlite3.connect(path)

        db = _Db(legacy)
        self.assertEqual(services._seen_expr(db, "web_prop_cards"), "datetime(observed_at_utc)")
        self.assertIsNotNone(services._scalar(
            db, f"SELECT MAX({services._seen_expr(db, 'web_prop_cards')}) FROM web_prop_cards"))
        db.conn.close()


if __name__ == "__main__":
    unittest.main()
