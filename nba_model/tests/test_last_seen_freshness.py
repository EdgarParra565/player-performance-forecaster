"""Review finding: change-only web_prop_cards / web_team_lines inserts made
``observed_at_utc`` mean "last changed", so every ``since_hours`` reader
dropped lines that were stable across days (a 24h consensus window returned
[] for a line re-scraped every hour but unchanged for 3 days).

Fix: ``last_seen_at_utc`` is bumped on every unchanged re-scrape and the
freshness readers filter on it.
"""

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nba_model.data.database.db_manager import DatabaseManager


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _card(ts, line=25.5, sha=None, book="prizepicks", player="LeBron James"):
    return {
        "snapshot_id": 1, "source_url": f"https://{book}.example", "book": book,
        "observed_at_utc": ts, "player_name": player,
        "player_classification": "active_nba", "stat_type": "points",
        "line_value": line, "side": "over", "parse_confidence": 0.9,
        "raw_card_text": "raw", "parser_version": "v1",
        "record_sha256": sha or f"{book}-{ts}-{line}",
    }


def _team_line(ts, book, market="total", side="over", team=None, line=220.5, odds=-110):
    return {
        "snapshot_id": 1, "source_url": f"https://{book}.example", "book": book,
        "observed_at_utc": ts, "away_team": "Knicks", "home_team": "76ers",
        "market_type": market, "side": side, "team": team,
        "line_value": line, "odds_american": odds, "parse_confidence": 0.9,
        "parser_version": "v1", "record_sha256": f"{book}-{market}-{side}-{ts}",
    }


class PropCardLastSeenTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "t.db")
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def _last_seen(self):
        return self.db.conn.execute(
            "SELECT observed_at_utc, last_seen_at_utc FROM web_prop_cards").fetchall()

    def test_stable_line_stays_in_since_hours_window(self):
        first = _iso(72)
        self.db.insert_web_prop_cards([_card(first), _card(first, book="underdog")])
        self.assertEqual(self.db.get_consensus_prop_lines(since_hours=24), [])
        now = _iso(0.1)
        res = self.db.insert_web_prop_cards([_card(now), _card(now, book="underdog")])
        self.assertEqual(res["inserted"], 0)
        self.assertEqual(res["skipped_unchanged"], 2)
        rows = self.db.get_consensus_prop_lines(since_hours=24, min_books=2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n_books"], 2)
        self.assertEqual(rows[0]["latest_observed_at"], now)
        # observed_at stays "last changed" (movement history intact).
        self.assertEqual({r[0] for r in self._last_seen()}, {first})

    def test_edge_scanner_reader_uses_last_seen(self):
        from nba_model.model.edge_scanner import fetch_latest_prop_lines

        self.db.insert_web_prop_cards([_card(_iso(72))])
        now = _iso(0.1)
        self.db.insert_web_prop_cards([_card(now)])
        df = fetch_latest_prop_lines(db_path=self.db_path, since_hours=24)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["observed_at_utc"], now)

    def test_player_chart_book_lines_use_last_seen(self):
        from nba_model.visualization import player_charts as pc

        self.db.insert_web_prop_cards([_card(_iso(72))])
        self.db.insert_web_prop_cards([_card(_iso(0.1))])
        movement = pc._fetch_line_movement(self.db, "points", "LeBron James",
                                           web_lookback_hours=24)
        self.assertEqual(movement, [])  # stable → no movement, but no crash

    def test_reparse_of_older_snapshot_never_moves_last_seen_back(self):
        newer, older = _iso(1), _iso(5)
        self.db.insert_web_prop_cards([_card(_iso(10))])
        self.db.insert_web_prop_cards([_card(newer)])
        self.db.insert_web_prop_cards([_card(older)])
        self.assertEqual(self._last_seen()[0][1], newer)

    def test_in_batch_duplicate_bumps_the_row_it_wrote(self):
        t1, t2 = _iso(2), _iso(1)
        res = self.db.insert_web_prop_cards([_card(t1), _card(t2)])
        self.assertEqual(res["inserted"], 1)
        self.assertEqual(self._last_seen(), [(t1, t2)])

    def test_moved_line_old_row_keeps_its_last_seen(self):
        t0, t1 = _iso(30), _iso(1)
        self.db.insert_web_prop_cards([_card(t0, line=25.5)])
        self.db.insert_web_prop_cards([_card(t1, line=26.5)])
        rows = self.db.conn.execute(
            "SELECT line_value, last_seen_at_utc FROM web_prop_cards ORDER BY card_id"
        ).fetchall()
        self.assertEqual(rows, [(25.5, t0), (26.5, t1)])
        latest = self.db.get_consensus_prop_lines(since_hours=24)
        self.assertEqual(latest[0]["mean_line"], 26.5)


class TeamLineLastSeenTests(unittest.TestCase):
    def test_stable_team_lines_feed_hourly_reverse_engineering(self):
        from nba_model.model.team_line_reverse_engineering import (
            derive_team_priors_from_consensus,
        )

        def board(ts):
            rows = []
            for book in ("draftkings", "fanduel"):
                rows += [
                    _team_line(ts, book),
                    _team_line(ts, book, "spread", "home", "76ers", -4.5),
                    _team_line(ts, book, "spread", "away", "Knicks", 4.5),
                ]
            return rows

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "t.db")
            with DatabaseManager(db_path) as db:
                db.insert_web_team_lines(board(_iso(30)))
                res = db.insert_web_team_lines(board(_iso(0.1)))
                self.assertEqual(res["inserted"], 0)
                self.assertEqual(res["skipped_unchanged"], 6)
                consensus = db.get_consensus_team_lines(since_hours=6, min_books=2)
            self.assertEqual(len(consensus), 3)
            summary = derive_team_priors_from_consensus(
                since_hours=6.0, min_books=2, db_path=db_path)
        # Before the fix the 6h hourly window saw nothing → 0 priors.
        self.assertEqual(summary["games_with_full_priors"], 1)


class LastSeenMigrationTests(unittest.TestCase):
    def test_pre_existing_tables_gain_backfilled_column(self):
        schema = (Path(__file__).resolve().parents[1]
                  / "data" / "database" / "schema.sql").read_text()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "old.db")
            conn = sqlite3.connect(db_path)
            conn.executescript(schema)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(web_prop_cards)")}
            self.assertNotIn("last_seen_at_utc", cols)  # old-shape table
            conn.execute(
                "INSERT INTO web_prop_cards (snapshot_id, source_url, book, "
                "observed_at_utc, player_name, player_classification, stat_type, "
                "line_value, side, parse_confidence, parser_version, record_sha256) "
                "VALUES (1, 'u', 'pp', '2026-05-08T00:00:00+00:00', 'X', "
                "'active_nba', 'points', 1.5, 'over', 0.9, 'v', 's')")
            conn.commit()
            conn.close()
            for _ in range(2):  # idempotent
                with DatabaseManager(db_path) as db:
                    row = db.conn.execute(
                        "SELECT observed_at_utc, last_seen_at_utc FROM web_prop_cards"
                    ).fetchone()
                    team_cols = {r[1] for r in db.conn.execute(
                        "PRAGMA table_info(web_team_lines)")}
                self.assertEqual(row[0], row[1])
                self.assertIn("last_seen_at_utc", team_cols)


if __name__ == "__main__":
    unittest.main()
