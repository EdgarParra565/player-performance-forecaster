"""llm_forecasts (WS11): idempotent migration, read-only compatibility,
validated inserts, and settlement through the predictions outcome logic.

The flagship opens the DB with ``mode=ro`` and refuses a file missing any
REQUIRED table, so the new table must be optional for read-only opens: an
older published snapshot without it still has to serve.
"""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from nba_model.data.database.db_manager import DatabaseManager

PID = 2544
DAY = "2026-01-05"


def _row(**overrides):
    row = {
        "run_id": "r1", "provider": "fake", "model_name": "m1",
        "prompt_version": "v1", "arm": "blind", "prompt_hash": "h" * 64,
        "player_id": PID, "player_name": "LeBron James", "game_date": DAY,
        "stat_type": "points", "line_value": 25.5, "prob_over": 0.6,
        "parse_status": "ok", "raw_response": '[{"text": "0.6"}]',
        "info_cutoff_utc": f"{DAY}T00:00:00Z", "retro": 1,
    }
    row.update(overrides)
    return row


def _committed(path, sql):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


class _TempDb(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self._tmp.name) / "nba.db")

    def tearDown(self):
        self._tmp.cleanup()


class MigrationAndReadOnlyTests(_TempDb):
    def test_writer_open_creates_table_and_is_idempotent(self):
        for _ in range(2):
            with DatabaseManager(self.path) as db:
                self.assertTrue(db.has_table("llm_forecasts"))
        cols = {r[1] for r in _committed(self.path, "PRAGMA table_info(llm_forecasts)")}
        for col in ("forecast_id", "run_id", "model_name", "prompt_version", "player_id",
                    "player_name", "game_date", "stat_type", "line_value", "prob_over",
                    "raw_response", "info_cutoff_utc", "outcome", "actual_result",
                    "created_at_utc", "retro", "arm", "prompt_hash"):
            self.assertIn(col, cols)
        indexes = {r[0] for r in _committed(
            self.path, "SELECT name FROM sqlite_master WHERE tbl_name = 'llm_forecasts' "
                       "AND type = 'index'")}
        self.assertTrue({"idx_llm_forecasts_run", "idx_llm_forecasts_prop"} <= indexes)

    def test_table_is_not_required_for_a_read_only_open(self):
        self.assertNotIn("llm_forecasts", DatabaseManager.required_schema())
        with DatabaseManager(self.path):
            pass
        conn = sqlite3.connect(self.path)  # an older snapshot: table absent
        conn.execute("DROP TABLE llm_forecasts")
        conn.commit()
        conn.close()
        before = os.path.getmtime(self.path)
        with DatabaseManager(self.path, read_only=True) as ro:
            self.assertFalse(ro.has_table("llm_forecasts"))
            self.assertTrue(ro.has_table("predictions"))
        self.assertEqual(os.path.getmtime(self.path), before)
        self.assertEqual(_committed(
            self.path, "SELECT COUNT(*) FROM sqlite_master WHERE name = 'llm_forecasts'"),
            [(0,)])

    def test_read_only_open_with_table_present(self):
        with DatabaseManager(self.path) as db:
            db.insert_llm_forecasts([_row()])
        with DatabaseManager(self.path, read_only=True) as ro:
            self.assertEqual(ro.conn.execute(
                "SELECT COUNT(*) FROM llm_forecasts").fetchone()[0], 1)
            with self.assertRaises(sqlite3.OperationalError):
                ro.insert_llm_forecasts([_row(run_id="r2")])


class InsertTests(_TempDb):
    def test_validates_and_ignores_duplicates(self):
        with DatabaseManager(self.path) as db:
            res = db.insert_llm_forecasts([
                _row(),
                _row(model_name="m2", parse_status="invalid", prob_over=None),
                _row(model_name="m3", prob_over=1.2),              # out of range
                _row(model_name="m4", prob_over=None),             # ok without prob
                _row(model_name="m5", arm="peek"),                 # unknown arm
                _row(model_name="m6", retro=2),                    # retro not 0/1
                _row(model_name="m7", parse_status="maybe"),       # unknown status
                _row(model_name="m8", player_id=None),             # missing key field
                _row(model_name="m9", info_cutoff_utc=""),         # missing cutoff
            ])
            self.assertEqual(res, {"inserted": 2, "attempted": 2,
                                   "duplicates_ignored": 0, "rejected": 7})
            again = db.insert_llm_forecasts([_row(prob_over=0.1)])
            self.assertEqual(again["inserted"], 0)
            self.assertEqual(again["duplicates_ignored"], 1)
            # The stored answer is never rewritten by a re-run.
            self.assertEqual(db.conn.execute(
                "SELECT prob_over FROM llm_forecasts WHERE model_name = 'm1'").fetchone(),
                (0.6,))
            # Market arm of the same prop is a separate forecast.
            self.assertEqual(db.insert_llm_forecasts([_row(arm="market")])["inserted"], 1)

    def test_check_constraint_backs_up_the_validator(self):
        with DatabaseManager(self.path) as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.conn.execute(
                    "INSERT INTO llm_forecasts (run_id, provider, model_name, "
                    "prompt_version, prompt_hash, player_id, player_name, game_date, "
                    "stat_type, line_value, prob_over, parse_status, info_cutoff_utc, "
                    "retro, created_at_utc) VALUES ('r', 'p', 'm', 'v1', 'h', 1, 'n', "
                    "'2026-01-01', 'points', 1.5, 1.5, 'ok', 'x', 1, 'x')")


class SettlementTests(_TempDb):
    LINES = (25.5, 30.0, 35.5)

    def _seed(self, db):
        db.conn.execute(
            "INSERT INTO game_logs (player_id, game_id, game_date, season, points, "
            "rebounds, assists) VALUES (?, 'G1', ?, '2025-26', 30, 8, 7)",
            (PID, f"{DAY}T00:00:00"))
        for line in self.LINES:
            db.conn.execute(
                "INSERT INTO predictions (player_id, game_date, stat_type, predicted_mean, "
                "predicted_std, prob_over, line_value) VALUES (?, ?, 'points', 28, 5, 0.5, ?)",
                (PID, DAY, line))
        db.conn.execute(
            "INSERT INTO predictions (player_id, game_date, stat_type, predicted_mean, "
            "predicted_std, prob_over, line_value) VALUES (?, ?, 'pra', 40, 6, 0.5, 44.5)",
            (PID, DAY))
        db.conn.commit()
        rows = [_row(line_value=line) for line in self.LINES]
        rows.append(_row(stat_type="pra", line_value=44.5))
        rows.append(_row(stat_type="points", line_value=25.5, game_date="2026-02-01",
                         info_cutoff_utc="2026-02-01T00:00:00Z"))  # no game yet
        self.assertEqual(db.insert_llm_forecasts(rows)["inserted"], 5)

    def test_same_outcomes_as_predictions(self):
        with DatabaseManager(self.path) as db:
            self._seed(db)
            pred = db.backfill_predictions_outcomes()
            llm = db.settle_llm_forecasts()
            self.assertEqual(pred["settled"], 4)
            self.assertEqual((llm["scanned"], llm["settled"], llm["remaining_pending"]),
                             (5, 4, 1))
            key_sql = ("SELECT stat_type, line_value, actual_result, outcome FROM {t} "
                       "WHERE outcome IS NOT NULL ORDER BY stat_type, line_value")
            self.assertEqual(
                db.conn.execute(key_sql.format(t="llm_forecasts")).fetchall(),
                db.conn.execute(key_sql.format(t="predictions")).fetchall())
            self.assertEqual([r[0] for r in db.conn.execute(
                "SELECT outcome FROM llm_forecasts WHERE stat_type = 'points' "
                "AND game_date = ? ORDER BY line_value", (DAY,))],
                ["over", "push", "under"])
            # Idempotent: only the future game is still pending.
            self.assertEqual(db.settle_llm_forecasts()["scanned"], 1)

    def test_mid_loop_failure_rolls_back(self):
        with DatabaseManager(self.path) as db:
            self._seed(db)
            third = db.conn.execute(
                "SELECT forecast_id FROM llm_forecasts ORDER BY forecast_id").fetchall()[2][0]
            db.conn.execute(
                f"CREATE TRIGGER boom BEFORE UPDATE ON llm_forecasts WHEN "
                f"OLD.forecast_id = {third} BEGIN SELECT RAISE(ABORT, 'injected'); END")
            db.conn.commit()
            db.conn.execute("INSERT INTO players (player_id, name) VALUES (77, 'Pending')")
            with self.assertRaises(sqlite3.DatabaseError):
                db.settle_llm_forecasts()
            db.conn.commit()
        self.assertEqual(_committed(
            self.path, "SELECT COUNT(*) FROM llm_forecasts WHERE actual_result IS NOT NULL"),
            [(0,)])
        self.assertEqual(_committed(self.path, "SELECT COUNT(*) FROM players"), [(1,)])


if __name__ == "__main__":
    unittest.main()
