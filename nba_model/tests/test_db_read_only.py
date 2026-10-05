"""DatabaseManager(read_only=True): no DDL, no migrations, no file creation.

Flagship handoff: every normal open runs schema.sql + migrations, so pointing
a read-only service at an empty/missing NBA_DB_PATH silently grew a fresh
schema. The read-only open must refuse instead.
"""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from nba_model.data.database.db_manager import DatabaseManager, DatabaseNotReadyError


def _tables(path):
    conn = sqlite3.connect(path)
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()


class ReadOnlyOpenTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_file_raises_and_creates_nothing(self):
        path = self.dir / "sub" / "nba.db"
        with self.assertRaises(DatabaseNotReadyError):
            DatabaseManager(str(path), read_only=True)
        self.assertFalse(path.exists())
        self.assertFalse(path.parent.exists())  # no mkdir either

    def test_empty_file_raises_and_stays_empty(self):
        path = self.dir / "nba.db"
        path.touch()
        with self.assertRaises(DatabaseNotReadyError):
            DatabaseManager(str(path), read_only=True)
        self.assertEqual(path.stat().st_size, 0)
        self.assertEqual(_tables(str(path)), set())

    def test_incomplete_file_raises_without_ddl(self):
        path = str(self.dir / "nba.db")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE players (player_id INTEGER PRIMARY KEY, name TEXT)")
        conn.commit()
        conn.close()
        with self.assertRaises(DatabaseNotReadyError) as ctx:
            DatabaseManager(path, read_only=True)
        self.assertIn("missing tables", str(ctx.exception))
        self.assertEqual(_tables(path), {"players"})  # nothing grown

    def test_unmigrated_file_is_incomplete(self):
        # Fresh schema.sql only (what a pre-migration DB looks like).
        path = str(self.dir / "nba.db")
        schema = (Path(__file__).resolve().parents[1] / "data" / "database" / "schema.sql").read_text()
        conn = sqlite3.connect(path)
        conn.executescript(schema)
        conn.close()
        with self.assertRaises(DatabaseNotReadyError) as ctx:
            DatabaseManager(path, read_only=True)
        self.assertIn("missing columns", str(ctx.exception))
        self.assertIn("web_prop_cards.last_seen_at_utc", str(ctx.exception))

    def test_not_a_database_raises(self):
        path = self.dir / "nba.db"
        path.write_bytes(b"this is not sqlite" * 100)
        with self.assertRaises(DatabaseNotReadyError):
            DatabaseManager(str(path), read_only=True)

    def test_complete_db_reads_but_cannot_write(self):
        path = str(self.dir / "nba.db")
        with DatabaseManager(path) as db:  # writer open builds + migrates
            db.conn.execute("INSERT INTO players (player_id, name) VALUES (1, 'A')")
            db.conn.commit()
        before = os.path.getmtime(path), _tables(path)
        with DatabaseManager(path, read_only=True) as ro:
            self.assertTrue(ro.read_only)
            self.assertEqual(ro.conn.execute("SELECT name FROM players").fetchall(), [("A",)])
            self.assertEqual(ro.get_team_prior_inputs_map(), {})  # reader works
            with self.assertRaises(sqlite3.OperationalError):
                ro.conn.execute("INSERT INTO players (player_id, name) VALUES (2, 'B')")
        self.assertEqual((os.path.getmtime(path), _tables(path)), before)

    def test_writer_open_unchanged(self):
        path = self.dir / "new" / "nba.db"
        with DatabaseManager(str(path)) as db:
            self.assertFalse(db.read_only)
        self.assertTrue(set(DatabaseManager.required_schema()) <= _tables(str(path)))


if __name__ == "__main__":
    unittest.main()
