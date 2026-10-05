"""Tests for the paper-trading ``bet_log`` table (WS10 Phase 2).

Round-trips insert → settle → grade against ``game_logs`` (including a push),
verifies idempotency, and checks ``clv_delta`` is filled from a closing
``betting_line_snapshots`` row when present (and left NULL otherwise). Offline,
Windows-safe teardown.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nba_model.data.database.db_manager import DatabaseManager

LEBRON_ID = 2544
GAME_DATE = "2025-04-10"


def _game_log(pid, game_date, points, rebounds, assists, minutes=34.0):
    return {
        "player_id": pid, "game_id": f"g{pid}_{game_date}",
        "game_date": game_date, "season": "2024-25",
        "matchup": "LAL vs. DEN", "home_away": "home", "result": "W",
        "minutes": minutes, "points": points, "rebounds": rebounds,
        "assists": assists, "fgm": 8, "fga": 16, "fg3m": 2, "fg3a": 6,
        "ftm": 4, "fta": 5, "oreb": 2, "dreb": 6, "steals": 1, "blocks": 0,
        "turnovers": 3, "plus_minus": 5,
    }


def _pick(stat, line, side, **extra):
    row = {
        "game_date": GAME_DATE, "player_id": LEBRON_ID,
        "player_name": "LeBron James", "stat_type": stat, "book": "Underdog",
        "line": line, "side": side, "model_prob": 0.60, "implied_prob": 0.5238,
        "edge": 0.08, "model_mode": "full", "distribution": "normal",
        "kelly_fraction": 0.05, "stake_units": 0.05,
    }
    row.update(extra)
    return row


class BetLogRoundTripTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = str(Path(self._tmp.name) / "nba.db")
        with DatabaseManager(db_path=self.db_path) as db:
            db.conn.execute(
                "INSERT INTO players (player_id, name, team) VALUES (?, ?, ?)",
                (LEBRON_ID, "LeBron James", "LAL"),
            )
            db.conn.commit()
            # Actuals: points 30, rebounds 8, assists 7.
            db.insert_game_logs(pd.DataFrame([_game_log(LEBRON_ID, GAME_DATE, 30, 8, 7)]))

    def tearDown(self):
        self._tmp.cleanup()

    def test_insert_and_grade_including_push(self):
        with DatabaseManager(db_path=self.db_path) as db:
            ins = db.insert_bet_log_rows([
                _pick("points", 25.5, "over"),    # 30 > 25.5 → won
                _pick("rebounds", 10.5, "under"),  # 8 < 10.5 → won
                _pick("rebounds", 8.0, "over"),   # 8 == 8 → push
                _pick("points", 33.5, "over"),    # 30 < 33.5 → lost
            ])
            self.assertEqual(ins, {"inserted": 4, "attempted": 4, "duplicates_ignored": 0})
            res = db.settle_bet_log()
            self.assertEqual(res["scanned"], 4)
            self.assertEqual(res["settled"], 4)
            self.assertEqual(res["remaining_pending"], 0)

            graded = dict(db.conn.execute(
                "SELECT stat_type || ':' || line || ':' || side, status "
                "FROM bet_log"
            ).fetchall())
        self.assertEqual(graded["points:25.5:over"], "won")
        self.assertEqual(graded["rebounds:10.5:under"], "won")
        self.assertEqual(graded["rebounds:8.0:over"], "push")
        self.assertEqual(graded["points:33.5:over"], "lost")

    def test_settle_is_idempotent(self):
        with DatabaseManager(db_path=self.db_path) as db:
            db.insert_bet_log_rows([_pick("points", 25.5, "over")])
            first = db.settle_bet_log()
            self.assertEqual(first["settled"], 1)
            # Second pass finds nothing pending.
            second = db.settle_bet_log()
            self.assertEqual(second["scanned"], 0)
            self.assertEqual(second["settled"], 0)
            actual = db.conn.execute(
                "SELECT actual_value, status FROM bet_log").fetchone()
        self.assertEqual(actual, (30.0, "won"))

    def test_pending_when_game_not_yet_logged(self):
        with DatabaseManager(db_path=self.db_path) as db:
            db.insert_bet_log_rows([_pick("points", 25.5, "over",
                                          game_date="2025-04-20")])
            res = db.settle_bet_log()
            self.assertEqual(res["settled"], 0)
            self.assertEqual(res["remaining_pending"], 1)
            status = db.conn.execute("SELECT status FROM bet_log").fetchone()[0]
        self.assertEqual(status, "pending")

    def test_invalid_side_is_skipped_on_insert(self):
        with DatabaseManager(db_path=self.db_path) as db:
            ins = db.insert_bet_log_rows([
                _pick("points", 25.5, "sideways"),
                _pick("points", 25.5, "over"),
            ])
        self.assertEqual(ins, {"inserted": 1, "attempted": 1, "duplicates_ignored": 0})


class BetLogClvTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = str(Path(self._tmp.name) / "nba.db")
        with DatabaseManager(db_path=self.db_path) as db:
            db.conn.execute(
                "INSERT INTO players (player_id, name, team) VALUES (?, ?, ?)",
                (LEBRON_ID, "LeBron James", "LAL"),
            )
            db.conn.commit()
            db.insert_game_logs(pd.DataFrame([_game_log(LEBRON_ID, GAME_DATE, 30, 8, 7)]))
            # Two points snapshots: open -110, close +120 (over side drifts).
            db.conn.executemany(
                """
                INSERT INTO betting_line_snapshots
                    (snapshot_ts_utc, event_id, game_date, player_id, book,
                     market_key, stat_type, line_value, over_odds, under_odds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("2025-04-09 12:00:00", "evt1", GAME_DATE, LEBRON_ID,
                     "Underdog", "points_over_under", "points", 25.5, -110, -110),
                    ("2025-04-10 17:00:00", "evt1", GAME_DATE, LEBRON_ID,
                     "Underdog", "points_over_under", "points", 25.5, 120, -140),
                ],
            )
            db.conn.commit()

    def tearDown(self):
        self._tmp.cleanup()

    def test_clv_delta_filled_from_close_snapshot(self):
        with DatabaseManager(db_path=self.db_path) as db:
            db.insert_bet_log_rows([
                _pick("points", 25.5, "over"),     # has a snapshot → CLV filled
                _pick("rebounds", 8.0, "under"),   # no snapshot → CLV NULL
            ])
            res = db.settle_bet_log(fill_clv=True)
            self.assertEqual(res["clv_filled"], 1)
            rows = dict(db.conn.execute(
                "SELECT stat_type, clv_delta FROM bet_log").fetchall())
        # close over implied = 100/220 = 0.4545; entry implied 0.5238.
        self.assertIsNotNone(rows["points"])
        self.assertAlmostEqual(rows["points"], (100.0 / 220.0) - 0.5238, places=4)
        self.assertIsNone(rows["rebounds"])

    def test_fill_clv_false_leaves_delta_null(self):
        with DatabaseManager(db_path=self.db_path) as db:
            db.insert_bet_log_rows([_pick("points", 25.5, "over")])
            res = db.settle_bet_log(fill_clv=False)
            self.assertEqual(res["clv_filled"], 0)
            clv = db.conn.execute("SELECT clv_delta FROM bet_log").fetchone()[0]
        self.assertIsNone(clv)


class BetLogDedupeTests(unittest.TestCase):
    """Review finding: re-running bet_slip for a slate double-counted stakes."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "t.db")

    def tearDown(self):
        self._tmp.cleanup()

    def test_same_pick_twice_is_ignored(self):
        with DatabaseManager(db_path=self.db_path) as db:
            first = db.insert_bet_log_rows([_pick("points", 25.5, "over")])
            again = db.insert_bet_log_rows([_pick("points", 25.5, "over"),
                                            _pick("points", 25.5, "over", book="UNDERDOG")])
            n = db.conn.execute("SELECT COUNT(*) FROM bet_log").fetchone()[0]
        self.assertEqual(first["inserted"], 1)
        self.assertEqual(again, {"inserted": 0, "attempted": 0, "duplicates_ignored": 2})
        self.assertEqual(n, 1)

    def test_distinct_picks_still_insert(self):
        with DatabaseManager(db_path=self.db_path) as db:
            res = db.insert_bet_log_rows([
                _pick("points", 25.5, "over"),
                _pick("points", 26.5, "over"),               # other line
                _pick("points", 25.5, "under"),              # other side
                _pick("points", 25.5, "over", book="PrizePicks"),
                _pick("points", 25.5, "over", model_mode="chart_mean"),
                _pick("points", 25.5, "over", game_date="2025-04-17"),
            ])
        self.assertEqual(res["inserted"], 6)

    def test_unique_index_present_and_enforced(self):
        import sqlite3
        with DatabaseManager(db_path=self.db_path) as db:
            idx = db.conn.execute(
                "SELECT name FROM sqlite_master WHERE name='ux_bet_log_pick'").fetchone()
            self.assertIsNotNone(idx)
            db.insert_bet_log_rows([_pick("points", 25.5, "over")])
            with self.assertRaises(sqlite3.IntegrityError):
                db.conn.execute(
                    "INSERT INTO bet_log (created_at_utc, game_date, player_name, stat_type, "
                    "book, line, side, model_mode) VALUES ('x', ?, 'lebron james', 'POINTS', "
                    "'underdog', 25.5, 'over', 'full')", (GAME_DATE,))

    def test_existing_duplicates_skip_index_but_inserts_still_dedupe(self):
        import sqlite3
        DatabaseManager(db_path=self.db_path).close()
        conn = sqlite3.connect(self.db_path)
        conn.execute("DROP INDEX ux_bet_log_pick")
        for _ in range(2):
            conn.execute(
                "INSERT INTO bet_log (created_at_utc, game_date, player_name, stat_type, "
                "book, line, side, model_mode) VALUES ('x', ?, 'LeBron James', 'points', "
                "'Underdog', 25.5, 'over', 'full')", (GAME_DATE,))
        conn.commit()
        conn.close()
        with DatabaseManager(db_path=self.db_path) as db:  # history kept, no index
            self.assertIsNone(db.conn.execute(
                "SELECT name FROM sqlite_master WHERE name='ux_bet_log_pick'").fetchone())
            self.assertEqual(db.conn.execute("SELECT COUNT(*) FROM bet_log").fetchone()[0], 2)
            res = db.insert_bet_log_rows([_pick("points", 25.5, "over")])
        self.assertEqual(res["inserted"], 0)


if __name__ == "__main__":
    unittest.main()
