"""Review finding: a failed executemany left a partial write that the NEXT,
unrelated commit() persisted (only insert_game_logs rolled back — and its
connection-wide rollback() also threw away a caller's pending work).

Fix: every insert helper runs inside ``DatabaseManager._atomic()`` (a
SAVEPOINT): a failure rolls back exactly that writer's rows and re-raises.

Failures are injected with a BEFORE INSERT trigger on a sentinel value, so a
batch fails on its LAST row after earlier rows already executed.
"""

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_model.data.database.db_manager import DatabaseManager

NOW = datetime.now(timezone.utc).isoformat()


def _boom_trigger(db, table, condition):
    db.conn.execute(
        f"CREATE TRIGGER boom_{table} BEFORE INSERT ON {table} "
        f"WHEN {condition} BEGIN SELECT RAISE(ABORT, 'injected failure'); END")
    db.conn.commit()


def _count(path, table):
    conn = sqlite3.connect(path)  # independent connection = what's committed
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def _games(n_good, bad=True):
    rows = [{"game_id": f"g{i}", "season": "2025-26", "season_type": "Regular Season",
             "game_date": "2026-01-0%d" % (i + 1), "team_id": 1, "team_abbrev": "LAL",
             "matchup": "LAL vs. DEN", "home_away": "home", "pts": 100, "opp_pts": 90}
            for i in range(n_good)]
    if bad:
        rows.append(dict(rows[0], game_id="BAD"))
    return rows


def _snaps(n_good, bad=True):
    rows = [{"source_url": f"https://x.example/{i}", "fetched_at_utc": NOW,
             "http_status": 200, "content_type": None, "text_content": f"t{i}",
             "text_length": 2, "content_sha256": f"h{i}"} for i in range(n_good)]
    if bad:
        rows.append(dict(rows[0], source_url="https://x.example/BAD", content_sha256="hBAD"))
    return rows


def _cards(n_good, bad=True):
    rows = [{"snapshot_id": 1, "source_url": "https://pp.example", "book": "prizepicks",
             "observed_at_utc": NOW, "player_name": f"Player {i}",
             "player_classification": "active_nba", "stat_type": "points",
             "line_value": 20.5, "side": "over", "parse_confidence": 0.9,
             "raw_card_text": "r", "parser_version": "v1", "record_sha256": f"c{i}"}
            for i in range(n_good)]
    if bad:
        rows.append(dict(rows[0], player_name="BAD", record_sha256="cBAD"))
    return rows


def _lines(n_good, bad=True):
    rows = [{"player_id": 1000 + i, "game_date": "2026-01-02", "book": "FanDuel",
             "stat_type": "points", "line_value": 20.5, "over_odds": -110,
             "under_odds": -110} for i in range(n_good)]
    if bad:
        rows.append(dict(rows[0], player_id=9999))
    return rows


class AtomicWriterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self._tmp.name) / "t.db")
        self.db = DatabaseManager(self.path)

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def _assert_all_or_nothing(self, table, write, condition):
        _boom_trigger(self.db, table, condition)
        with self.assertRaises(sqlite3.DatabaseError):
            write(bad=True)
        self.db.conn.commit()  # the "next unrelated commit"
        self.assertEqual(_count(self.path, table), 0, f"{table}: partial write persisted")
        write(bad=False)       # same connection still usable; good batch lands
        self.assertEqual(_count(self.path, table), 3)

    def test_insert_games(self):
        self._assert_all_or_nothing(
            "games", lambda bad: self.db.insert_games(_games(3, bad)), "NEW.game_id = 'BAD'")

    def test_insert_web_text_snapshots(self):
        self._assert_all_or_nothing(
            "web_text_snapshots", lambda bad: self.db.insert_web_text_snapshots(_snaps(3, bad)),
            "NEW.source_url LIKE '%BAD'")

    def test_insert_web_prop_cards(self):
        self._assert_all_or_nothing(
            "web_prop_cards", lambda bad: self.db.insert_web_prop_cards(_cards(3, bad)),
            "NEW.player_name = 'BAD'")

    def test_insert_betting_lines_records(self):
        self._assert_all_or_nothing(
            "betting_lines", lambda bad: self.db.insert_betting_lines_records(_lines(3, bad)),
            "NEW.player_id = 9999")

    def test_insert_game_logs(self):
        def write(bad):
            rows = [{"player_id": 1, "game_id": f"G{i}", "game_date": "2026-01-02",
                     "season": "2025-26", "points": 10} for i in range(3)]
            if bad:
                rows.append({"player_id": 1, "game_id": "BAD", "game_date": "2026-01-02",
                             "season": "2025-26", "points": 1})
            self.db.insert_game_logs(pd.DataFrame(rows))
        self._assert_all_or_nothing("game_logs", write, "NEW.game_id = 'BAD'")

    def test_sync_players_unique_name_collision_rolls_back_whole_upsert(self):
        # The finding's own trigger: ON CONFLICT(player_id) vs UNIQUE(name).
        self.db.conn.execute("INSERT INTO players (player_id, name) VALUES (1, 'Same Name')")
        # Ref row 2 is a new player; ref row 3 reuses player 1's name under a
        # different player_id → UNIQUE(name) fails on the 2nd upsert row.
        self.db.conn.execute(
            "INSERT INTO nba_active_players_ref (player_id, player_name, synced_at_utc) "
            "VALUES (2, 'New Guy', ?), (3, 'Same Name', ?)", (NOW, NOW))
        self.db.conn.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.sync_players_table()
        self.db.conn.commit()
        self.assertEqual(_count(self.path, "players"), 1)  # 'New Guy' not half-written

    def test_failure_keeps_callers_pending_work(self):
        _boom_trigger(self.db, "games", "NEW.game_id = 'BAD'")
        self.db.conn.execute("INSERT INTO players (player_id, name) VALUES (7, 'Pending')")
        with self.assertRaises(sqlite3.DatabaseError):
            self.db.insert_games(_games(3, bad=True))
        self.db.conn.commit()
        self.assertEqual(_count(self.path, "players"), 1)  # caller's row survived
        self.assertEqual(_count(self.path, "games"), 0)

    def test_game_logs_failure_no_longer_discards_callers_work(self):
        _boom_trigger(self.db, "game_logs", "NEW.game_id = 'BAD'")
        self.db.conn.execute("INSERT INTO players (player_id, name) VALUES (8, 'Pending')")
        with self.assertRaises(sqlite3.DatabaseError):
            self.db.insert_game_logs(pd.DataFrame([
                {"player_id": 1, "game_id": "G1", "game_date": "2026-01-02",
                 "season": "2025-26", "points": 3},
                {"player_id": 1, "game_id": "BAD", "game_date": "2026-01-02",
                 "season": "2025-26", "points": 3},
            ]))
        self.db.conn.commit()
        self.assertEqual(_count(self.path, "players"), 1)
        self.assertEqual(_count(self.path, "game_logs"), 0)

    def test_success_is_committed_and_visible_to_other_connections(self):
        self.db.insert_games(_games(3, bad=False))
        self.assertEqual(_count(self.path, "games"), 3)
        self.assertFalse(self.db.conn.in_transaction)


if __name__ == "__main__":
    unittest.main()
