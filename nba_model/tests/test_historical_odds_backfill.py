"""Tests for historical-odds backfill (task 5a).

The backfill derives ``betting_lines`` rows from scraped ``web_prop_cards`` so
``run_distribution_sweep.py --use-market-lines`` has market data to settle on.
We validate the data path end-to-end up to the seam the sweep actually reads:
``DatabaseManager.get_market_line``.
"""

import tempfile
import unittest
from pathlib import Path

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.data.historical_odds_backfill import (
    backfill_betting_lines_from_web_prop_cards,
)

PP_URL = "https://app.prizepicks.com/board/nba"
CURRY_ID = 201939
LEBRON_ID = 2544


def _build_fixture_db(db_path: str) -> None:
    db = DatabaseManager(db_path=db_path)
    db.conn.execute(
        "INSERT INTO players(player_id, name, team) VALUES (?,?,?)",
        (CURRY_ID, "Stephen Curry", "GSW"),
    )
    db.conn.execute(
        "INSERT INTO nba_active_players_ref(player_id, player_name, synced_at_utc) "
        "VALUES (?,?,?)",
        (CURRY_ID, "Stephen Curry", "2026-03-15T00:00:00Z"),
    )
    db.conn.execute(
        "INSERT INTO web_text_snapshots(snapshot_id, source_url, fetched_at_utc, "
        "http_status, text_content, text_length, content_sha256) "
        "VALUES (1, ?, '2026-03-15T18:00:00Z', 200, 'x', 1, 'h1')",
        (PP_URL,),
    )

    def card(ts, name, line, sha):
        db.conn.execute(
            """INSERT INTO web_prop_cards(snapshot_id, source_url, book, observed_at_utc,
               player_name, player_classification, stat_type, line_value, side,
               parse_confidence, raw_card_text, parser_version, record_sha256)
               VALUES (1, ?, 'prizepicks', ?, ?, 'active_nba', 'points', ?, 'over',
                       0.9, ?, 'v', ?)""",
            (PP_URL, ts, name, line, f"raw-{sha}", sha),
        )

    # Two observations same day → latest (26.5) should win.
    card("2026-03-15T18:00:00Z", "Stephen Curry", 25.5, "a")
    card("2026-03-15T19:00:00Z", "Stephen Curry", 26.5, "b")
    # LeBron is NOT in players/ref → should be unresolved + dropped.
    card("2026-03-15T19:00:00Z", "LeBron James", 27.5, "c")
    db.conn.commit()
    db.close()


class BackfillFromWebPropCardsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "nba_data.db")
        _build_fixture_db(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_backfill_resolves_latest_line_and_drops_unresolved(self):
        summary = backfill_betting_lines_from_web_prop_cards(db_path=self.db_path)
        self.assertEqual(summary["resolved"], 1)
        self.assertEqual(summary["unresolved_players"], 1)  # LeBron dropped
        self.assertEqual(summary["inserted"], 1)

        with DatabaseManager(db_path=self.db_path) as db:
            rows = db.conn.execute(
                "SELECT player_id, game_date, book, stat_type, line_value, "
                "over_odds, under_odds FROM betting_lines"
            ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], (CURRY_ID, "2026-03-15", "prizepicks", "points", 26.5, -110, -110))

    def test_backfill_is_idempotent(self):
        first = backfill_betting_lines_from_web_prop_cards(db_path=self.db_path)
        second = backfill_betting_lines_from_web_prop_cards(db_path=self.db_path)
        self.assertEqual(first["inserted"], 1)
        self.assertEqual(second["inserted"], 0)
        self.assertEqual(second["duplicates_ignored"], 1)

    def test_dry_run_inserts_nothing(self):
        summary = backfill_betting_lines_from_web_prop_cards(
            db_path=self.db_path, dry_run=True
        )
        self.assertTrue(summary["dry_run"])
        self.assertEqual(summary["inserted"], 0)
        with DatabaseManager(db_path=self.db_path) as db:
            count = db.conn.execute("SELECT COUNT(*) FROM betting_lines").fetchone()[0]
        self.assertEqual(count, 0)

    def test_end_to_end_market_line_seam(self):
        """The sweep reads market lines via get_market_line; prove it resolves."""
        backfill_betting_lines_from_web_prop_cards(db_path=self.db_path)
        with DatabaseManager(db_path=self.db_path) as db:
            line = db.get_market_line(
                player_id=CURRY_ID, game_date="2026-03-15",
                stat_type="points", agg="median",
            )
        self.assertEqual(line, 26.5)

    def test_books_filter(self):
        summary = backfill_betting_lines_from_web_prop_cards(
            db_path=self.db_path, books=["draftkings"]  # no such book in fixture
        )
        self.assertEqual(summary["candidates"], 0)
        self.assertEqual(summary["inserted"], 0)


if __name__ == "__main__":
    unittest.main()
