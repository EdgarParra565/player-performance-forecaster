"""settle_bet_log / backfill_predictions_outcomes are all-or-nothing.

Both used to UPDATE row by row and commit once at the end: a failure on row
k left rows 0..k-1 settled in the open transaction, and the next unrelated
commit persisted that partial settle. They now run inside
``DatabaseManager._atomic()`` (SAVEPOINT): a failure rolls back that call's
rows and re-raises; a caller's uncommitted work is kept.

Failures are injected with a BEFORE UPDATE trigger on the 3rd row.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from nba_model.data.database.db_manager import DatabaseManager

PID = 2544
GAME_DATE = "2026-01-05"


def _committed(path, sql):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchone()[0]
    finally:
        conn.close()


class _Case(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self._tmp.name) / "t.db")
        self.db = DatabaseManager(self.path)
        self.db.conn.execute(
            "INSERT INTO game_logs (player_id, game_id, game_date, season, points, "
            "rebounds, assists) VALUES (?, 'G1', ?, '2025-26', 30, 8, 7)", (PID, GAME_DATE))
        self.db.conn.commit()

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def _fail_on(self, table, id_col, row_id):
        self.db.conn.execute(
            f"CREATE TRIGGER boom BEFORE UPDATE ON {table} WHEN OLD.{id_col} = {row_id} "
            "BEGIN SELECT RAISE(ABORT, 'injected failure'); END")
        self.db.conn.commit()

    def _caller_pending_insert(self):
        # Uncommitted caller work that must survive the writer's rollback.
        self.db.conn.execute("INSERT INTO players (player_id, name) VALUES (77, 'Pending')")


class BackfillPredictionsAtomicTests(_Case):
    def _seed(self):
        for line in (25.5, 30.0, 35.5):
            self.db.conn.execute(
                "INSERT INTO predictions (player_id, game_date, stat_type, predicted_mean, "
                "predicted_std, prob_over, line_value) VALUES (?, ?, 'points', 28, 5, 0.5, ?)",
                (PID, GAME_DATE, line))
        self.db.conn.commit()
        return [r[0] for r in self.db.conn.execute(
            "SELECT prediction_id FROM predictions ORDER BY prediction_id")]

    def test_mid_loop_failure_rolls_back_whole_backfill(self):
        ids = self._seed()
        self._fail_on("predictions", "prediction_id", ids[2])
        self._caller_pending_insert()
        with self.assertRaises(sqlite3.DatabaseError):
            self.db.backfill_predictions_outcomes()
        self.db.conn.commit()  # the "next unrelated commit"
        self.assertEqual(_committed(
            self.path, "SELECT COUNT(*) FROM predictions WHERE actual_result IS NOT NULL"), 0)
        self.assertEqual(_committed(self.path, "SELECT COUNT(*) FROM players"), 1)

    def test_success_settles_all_and_commits(self):
        self._seed()
        res = self.db.backfill_predictions_outcomes()
        self.assertEqual(res["settled"], 3)
        self.assertEqual(_committed(
            self.path, "SELECT group_concat(outcome) FROM (SELECT outcome FROM predictions "
                       "ORDER BY prediction_id)"), "over,push,under")
        self.assertFalse(self.db.conn.in_transaction)


class SettleBetLogAtomicTests(_Case):
    def _seed(self):
        rows = [{"game_date": GAME_DATE, "player_id": PID, "player_name": "LeBron James",
                 "stat_type": "points", "book": "Underdog", "line": line, "side": "over",
                 "model_mode": "full"} for line in (25.5, 30.0, 35.5)]
        self.assertEqual(self.db.insert_bet_log_rows(rows)["inserted"], 3)
        return [r[0] for r in self.db.conn.execute("SELECT log_id FROM bet_log ORDER BY log_id")]

    def test_mid_loop_failure_rolls_back_whole_settle(self):
        ids = self._seed()
        self._fail_on("bet_log", "log_id", ids[2])
        self._caller_pending_insert()
        with self.assertRaises(sqlite3.DatabaseError):
            self.db.settle_bet_log(fill_clv=False)
        self.db.conn.commit()
        self.assertEqual(_committed(
            self.path, "SELECT COUNT(*) FROM bet_log WHERE status <> 'pending'"), 0)
        self.assertEqual(_committed(self.path, "SELECT COUNT(*) FROM players"), 1)
        # Still settleable once the cause is gone.
        self.db.conn.execute("DROP TRIGGER boom")
        self.assertEqual(self.db.settle_bet_log(fill_clv=False)["settled"], 3)

    def test_success_settles_all_and_commits(self):
        self._seed()
        res = self.db.settle_bet_log(fill_clv=False)
        self.assertEqual(res["settled"], 3)
        self.assertEqual(_committed(
            self.path, "SELECT group_concat(status) FROM (SELECT status FROM bet_log "
                       "ORDER BY log_id)"), "won,push,lost")
        self.assertFalse(self.db.conn.in_transaction)


if __name__ == "__main__":
    unittest.main()
