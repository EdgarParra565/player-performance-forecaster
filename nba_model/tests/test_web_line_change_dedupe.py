"""Regression tests for change-only ingestion of scraped lines.

Re-scraping the same game must NOT pile up duplicate rows when the line is
unchanged. A new ``web_prop_cards`` / ``web_team_lines`` row is written only
when the book actually moves the line (or odds), so the line-movement history
reflects real movement instead of one row per hourly scrape.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from nba_model.data.database.db_manager import DatabaseManager


def _seed_snapshot(db: DatabaseManager, snapshot_id: int, ts: str, url: str) -> None:
    db.conn.execute(
        """
        INSERT INTO web_text_snapshots
            (snapshot_id, source_url, fetched_at_utc, http_status,
             text_content, text_length, content_sha256)
        VALUES (?, ?, ?, 200, 'x', 1, ?)
        """,
        (snapshot_id, url, ts, f"h{snapshot_id}"),
    )
    db.conn.commit()


class PropCardChangeOnlyTests(unittest.TestCase):
    def setUp(self):
        self.db_path = os.path.join(tempfile.mkdtemp(), "t.db")
        self.db = DatabaseManager(self.db_path)
        for sid in range(1, 5):
            _seed_snapshot(
                self.db, sid, f"2026-06-30T0{sid}:00:00Z", "https://prizepicks.com"
            )

    def _card(self, snapshot_id, line, sha, ts):
        return {
            "snapshot_id": snapshot_id,
            "source_url": "https://prizepicks.com",
            "book": "prizepicks",
            "observed_at_utc": ts,
            "player_name": "LeBron James",
            "player_classification": "active_nba",
            "stat_type": "points",
            "line_value": line,
            "side": "over",
            "parse_confidence": 0.9,
            "raw_card_text": f"raw-{sha}",
            "parser_version": "v1",
            "record_sha256": sha,
        }

    def _lines(self):
        return [
            r[0]
            for r in self.db.conn.execute(
                "SELECT line_value FROM web_prop_cards ORDER BY card_id"
            ).fetchall()
        ]

    def test_unchanged_line_is_skipped(self):
        first = self.db.insert_web_prop_cards(
            [self._card(1, 25.5, "a", "2026-06-30T01:00:00Z")]
        )
        second = self.db.insert_web_prop_cards(
            [self._card(2, 25.5, "b", "2026-06-30T02:00:00Z")]
        )
        self.assertEqual(first["inserted"], 1)
        self.assertEqual(second["inserted"], 0)
        self.assertEqual(second["skipped_unchanged"], 1)
        self.assertEqual(self._lines(), [25.5])

    def test_line_move_and_moveback_each_stored(self):
        self.db.insert_web_prop_cards([self._card(1, 25.5, "a", "2026-06-30T01:00:00Z")])
        self.db.insert_web_prop_cards([self._card(2, 25.5, "b", "2026-06-30T02:00:00Z")])
        self.db.insert_web_prop_cards([self._card(3, 26.5, "c", "2026-06-30T03:00:00Z")])
        self.db.insert_web_prop_cards([self._card(4, 25.5, "d", "2026-06-30T04:00:00Z")])
        # move-back to 25.5 is a real change vs the prior 26.5, so it is kept.
        self.assertEqual(self._lines(), [25.5, 26.5, 25.5])


class TeamLineChangeOnlyTests(unittest.TestCase):
    def setUp(self):
        self.db_path = os.path.join(tempfile.mkdtemp(), "t.db")
        self.db = DatabaseManager(self.db_path)
        for sid in range(1, 5):
            _seed_snapshot(
                self.db, sid, f"2026-06-30T0{sid}:00:00Z", "https://betmgm.com"
            )

    def _tl(self, snapshot_id, line, odds, sha, ts):
        return {
            "snapshot_id": snapshot_id,
            "source_url": "https://betmgm.com",
            "book": "betmgm",
            "observed_at_utc": ts,
            "away_team": "Knicks",
            "home_team": "Celtics",
            "market_type": "spread",
            "side": "home",
            "team": "Celtics",
            "line_value": line,
            "odds_american": odds,
            "parse_confidence": 0.9,
            "raw_text": f"raw-{sha}",
            "parser_version": "v1",
            "record_sha256": sha,
        }

    def _states(self):
        return self.db.conn.execute(
            "SELECT line_value, odds_american FROM web_team_lines ORDER BY line_id"
        ).fetchall()

    def test_unchanged_line_and_odds_skipped(self):
        self.db.insert_web_team_lines([self._tl(1, -3.5, -110, "t1", "2026-06-30T01:00:00Z")])
        res = self.db.insert_web_team_lines(
            [self._tl(2, -3.5, -110, "t2", "2026-06-30T02:00:00Z")]
        )
        self.assertEqual(res["inserted"], 0)
        self.assertEqual(res["skipped_unchanged"], 1)
        self.assertEqual(self._states(), [(-3.5, -110)])

    def test_odds_only_move_is_stored(self):
        self.db.insert_web_team_lines([self._tl(1, -3.5, -110, "t1", "2026-06-30T01:00:00Z")])
        res = self.db.insert_web_team_lines(
            [self._tl(2, -3.5, -115, "t2", "2026-06-30T02:00:00Z")]
        )
        self.assertEqual(res["inserted"], 1)
        self.assertEqual(self._states(), [(-3.5, -110), (-3.5, -115)])

    def test_duplicate_in_same_batch_inserts_once(self):
        res = self.db.insert_web_team_lines(
            [
                self._tl(1, -3.5, -110, "t1", "2026-06-30T01:00:00Z"),
                self._tl(2, -3.5, -110, "t2", "2026-06-30T02:00:00Z"),
            ]
        )
        self.assertEqual(res["inserted"], 1)
        self.assertEqual(res["skipped_unchanged"], 1)


def _hours_ago(hours: float) -> str:
    """UTC timestamp N hours ago, matching the scraped ``observed_at_utc`` format.

    ``_fetch_line_movement`` filters on ``datetime('now', '-48 hours')``, so
    fixtures must be date-relative — hardcoded dates rot out of the window.
    """
    ts = datetime.now(timezone.utc) - timedelta(hours=hours)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


class LineMovementSummaryTests(unittest.TestCase):
    def setUp(self):
        from nba_model.visualization import player_charts as pc

        self.pc = pc
        self.db_path = os.path.join(tempfile.mkdtemp(), "t.db")
        self.db = DatabaseManager(self.db_path)
        for sid in range(1, 6):
            _seed_snapshot(
                self.db, sid, _hours_ago(6 - sid), "https://prizepicks.com"
            )

    def _ins(self, snapshot_id, line, sha, ts, book="prizepicks"):
        self.db.conn.execute(
            """
            INSERT INTO web_prop_cards
                (snapshot_id, source_url, book, observed_at_utc, player_name,
                 player_classification, stat_type, line_value, side,
                 parse_confidence, raw_card_text, parser_version, record_sha256)
            VALUES (?, ?, ?, ?, ?, 'active_nba', 'points', ?, 'over', 0.9, ?, 'v1', ?)
            """,
            (snapshot_id, "https://prizepicks.com", book, ts, "LeBron James",
             line, f"raw-{sha}", sha),
        )
        self.db.conn.commit()

    def test_up_move_reports_from_lowest(self):
        self._ins(1, 25.5, "a", _hours_ago(5))
        self._ins(2, 26.5, "b", _hours_ago(4))
        self._ins(3, 27.0, "c", _hours_ago(3))
        mv = self.pc._fetch_line_movement(self.db, "points", "LeBron James")
        self.assertEqual(len(mv), 1)
        self.assertEqual(mv[0]["direction"], "up")
        self.assertEqual(mv[0]["previous"], 25.5)
        self.assertEqual(mv[0]["current"], 27.0)
        self.assertIn("(low)", mv[0]["text"])

    def test_down_move_reports_from_highest(self):
        self._ins(1, 30.0, "a", _hours_ago(5))
        self._ins(2, 28.5, "b", _hours_ago(4))
        mv = self.pc._fetch_line_movement(self.db, "points", "LeBron James")
        self.assertEqual(mv[0]["direction"], "down")
        self.assertEqual(mv[0]["previous"], 30.0)
        self.assertIn("(high)", mv[0]["text"])

    def test_flat_line_is_omitted(self):
        self._ins(1, 22.0, "a", _hours_ago(5))
        self._ins(2, 22.0, "b", _hours_ago(4))
        mv = self.pc._fetch_line_movement(self.db, "points", "LeBron James")
        self.assertEqual(mv, [])


if __name__ == "__main__":
    unittest.main()
