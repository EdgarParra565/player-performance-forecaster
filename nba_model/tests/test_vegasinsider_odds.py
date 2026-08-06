"""Tests for the VegasInsider cross-book odds-grid parser + betting_lines ingest.

The fixture (`VI_FIXTURE`) is a trimmed excerpt of a REAL VegasInsider snapshot
(``web_text_snapshots`` for ``vegasinsider.com/nba/odds/player-props``): the
Points section (2 players) + the Rebounds section (1 player), each carrying the
verbatim 11-book header and one over-only cell per book.

Coverage mirrors ``test_new_book_scrapers``: pure-parser unit assertions on the
extractor, then a round-trip through the real ingestion orchestrator into a temp
DB, read back with the same ``cross_book_arb.fetch_two_way_lines`` the arb/edge
views use.
"""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.data.vegasinsider_odds_ingestion import ingest_vegasinsider_odds
from nba_model.model import cross_book_arb as cba
from nba_model.scrapers import vegasinsider as vi


# Trimmed, whitespace-collapsed REAL capture (see module docstring).
VI_FIXTURE = (
    "Points Odds Time Bet365 PrizePicks BetMGM DraftKings Caesars FanDuel "
    "HardRock Fanatics Sleeper Underdog RiversCasino › › › › › › › › › › › "
    "Jalen Brunson o26.5 -112 + o26.5 -137 + o26.5 -120 + o26.5 -125 + "
    "o26.5 -125 + o26.5 -113 + o26.5 -110 + o26.5 -120 + o27.5 -118 + "
    "o26.5 -137 o26.5 -121 + Tyrese Maxey o26.5 -105 + o25.5 -137 + "
    "o25.5 -125 + o25.5 -124 + o25.5 -121 + o25.5 -111 + o26.5 -105 + "
    "o25.5 -120 + o26.5 -119 + o25.5 -137 o25.5 -114 + See All Rebounds "
    "Odds Time Bet365 PrizePicks BetMGM DraftKings Caesars FanDuel HardRock "
    "Fanatics Sleeper Underdog RiversCasino › › › › › › › › › › › "
    "Victor Wembanyama o12.5 -110 + o12.5 -137 + o12.5 -120 + o12.5 -111 + "
    "o12.5 -114 + o12.5 -106 + o12.5 -115 + o12.5 -105 + o12.5 -135 + "
    "o12.5 -137 o12.5 -106 +"
)

_PLAYERS = {1: "Jalen Brunson", 2: "Tyrese Maxey", 3: "Victor Wembanyama"}


class ExtractOddsRowsTests(unittest.TestCase):
    def setUp(self):
        self.rows = vi.extract_odds_rows(VI_FIXTURE)

    def test_row_count_is_players_times_books(self):
        # 3 players × 11 books = 33 over rows.
        self.assertEqual(len(self.rows), 33)

    def test_every_book_column_present_in_order(self):
        brunson = [r for r in self.rows if r["player_name"] == "Jalen Brunson"]
        books = [r["book"] for r in brunson]
        # Aggregator labels mapped to registry names (hardrock→hardrockbet,
        # riverscasino→betrivers); bet365 stays as-is (no scraper).
        self.assertEqual(books, [
            "bet365", "prizepicks", "betmgm", "draftkings", "caesars",
            "fanduel", "hardrockbet", "fanatics", "sleeper", "underdog",
            "betrivers",
        ])

    def test_specific_cell_values(self):
        by = {(r["player_name"], r["book"], r["stat_type"]): r for r in self.rows}
        dk = by[("Jalen Brunson", "draftkings", "points")]
        self.assertEqual(dk["line_value"], 26.5)
        self.assertEqual(dk["over_odds"], -125)
        # riverscasino column → betrivers, +/- sign preserved.
        self.assertEqual(by[("Tyrese Maxey", "betrivers", "points")]["over_odds"], -114)

    def test_over_only_no_under_key(self):
        # Grid is over-only; the extractor never emits an under price.
        for r in self.rows:
            self.assertNotIn("under_odds", r)

    def test_stat_sections_normalized(self):
        stats = {r["stat_type"] for r in self.rows}
        self.assertEqual(stats, {"points", "rebounds"})

    def test_normalize_stat_mapping(self):
        self.assertEqual(vi._normalize_stat("3 Pointers"), "three_pointers_made")
        self.assertEqual(vi._normalize_stat("See All Rebounds"), "rebounds")
        self.assertEqual(vi._normalize_stat("Assists"), "assists")
        self.assertEqual(vi._normalize_stat("Points"), "points")
        self.assertIsNone(vi._normalize_stat("Steals"))

    def test_misaligned_row_dropped(self):
        # A header with a player carrying only 3 cells (not 11) is not
        # attributable → dropped, never guessed at.
        bad = (
            "Points Odds Time Bet365 PrizePicks BetMGM DraftKings Caesars "
            "FanDuel HardRock Fanatics Sleeper Underdog RiversCasino › "
            "Some Player o10.5 -110 + o10.5 -120 + o10.5 -130 +"
        )
        self.assertEqual(vi.extract_odds_rows(bad), [])

    def test_empty_text(self):
        self.assertEqual(vi.extract_odds_rows(""), [])


class IngestRoundTripTests(unittest.TestCase):
    """Snapshot text → real ingest orchestrator → temp DB → arb reader."""

    def _seed_players(self, db):
        now = datetime.now(timezone.utc).isoformat()
        db.upsert_active_players_reference([
            {"player_id": pid, "player_name": name, "synced_at_utc": now}
            for pid, name in _PLAYERS.items()
        ])

    def test_real_odds_land_in_betting_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba_data.db")
            with DatabaseManager(db_path=db_path) as db:
                self._seed_players(db)

            summary = ingest_vegasinsider_odds(
                db_path=db_path,
                snapshot_text=VI_FIXTURE,
                game_date="2026-05-08",
            )

            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["parsed_rows"], 33)
            self.assertEqual(summary["resolved"], 33)
            self.assertEqual(summary["inserted"], 33)

            with DatabaseManager(db_path=db_path) as db:
                rows = db.conn.execute(
                    "SELECT book, stat_type, line_value, over_odds, under_odds, "
                    "source FROM betting_lines ORDER BY book"
                ).fetchall()
        self.assertEqual(len(rows), 33)
        # Provenance tagged; over-only (under NULL) on every row.
        self.assertTrue(all(r[5] == "vegasinsider" for r in rows))
        self.assertTrue(all(r[4] is None for r in rows))
        self.assertTrue(all(r[3] is not None for r in rows))

    def test_reidempotent_and_reads_back_via_fetch_two_way_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba_data.db")
            with DatabaseManager(db_path=db_path) as db:
                self._seed_players(db)

            first = ingest_vegasinsider_odds(
                db_path=db_path, snapshot_text=VI_FIXTURE, game_date="2026-05-08")
            # Re-running is idempotent: exact-duplicate rows are skipped.
            second = ingest_vegasinsider_odds(
                db_path=db_path, snapshot_text=VI_FIXTURE, game_date="2026-05-08")
            self.assertEqual(first["inserted"], 33)
            self.assertEqual(second["inserted"], 0)

            two_way = cba.fetch_two_way_lines(db_path=db_path)

        # The arb/edge reader sees Brunson points across all 11 books with real
        # over odds (under NULL → no arb, but line-shopping / market-line ready).
        brunson_pts = two_way[
            (two_way["player_id"] == 1) & (two_way["stat_type"] == "points")
        ]
        self.assertEqual(len(brunson_pts), 11)
        self.assertEqual(int(brunson_pts[brunson_pts["book"] == "draftkings"]
                             ["over_odds"].iloc[0]), -125)


class EtlStepIngestionTests(unittest.TestCase):
    """The new hourly / daily ETL VI steps, reading a DB-stored snapshot.

    Exercises the step wrappers (`hourly_update._run_vegasinsider_ingestion`,
    `daily_etl._run_vegasinsider_step`) end to end: they must read the freshest
    stored VegasInsider snapshot and flow it into betting_lines, be idempotent,
    and treat an empty/offseason grid as a normal 0-row outcome (no exception).
    """

    VI_URL = "https://www.vegasinsider.com/nba/odds/player-props/"

    def _seed(self, db_path, text):
        with DatabaseManager(db_path=db_path) as db:
            now = datetime.now(timezone.utc).isoformat()
            db.upsert_active_players_reference([
                {"player_id": pid, "player_name": name, "synced_at_utc": now}
                for pid, name in _PLAYERS.items()
            ])
            db.insert_web_text_snapshots([{
                "source_url": self.VI_URL,
                "fetched_at_utc": now,
                "http_status": 200,
                "content_type": "text/html",
                "text_content": text,
                "text_length": len(text),
                "content_sha256": f"vi-{len(text)}",
            }])

    def test_hourly_step_ingests_stored_snapshot(self):
        from nba_model.data.hourly_update import _run_vegasinsider_ingestion
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba_data.db")
            self._seed(db_path, VI_FIXTURE)
            summary = _run_vegasinsider_ingestion(db_path)
            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["parsed_rows"], 33)
            self.assertEqual(summary["inserted"], 33)
            # Idempotent: a second pass over the unchanged snapshot inserts 0.
            again = _run_vegasinsider_ingestion(db_path)
            self.assertEqual(again["inserted"], 0)
            self.assertEqual(again["duplicates_ignored"], 33)
            with DatabaseManager(db_path=db_path) as db:
                n = db.conn.execute(
                    "SELECT COUNT(*) FROM betting_lines WHERE source='vegasinsider'"
                ).fetchone()[0]
            self.assertEqual(n, 33)

    def test_hourly_step_empty_offseason_grid_is_zero_not_error(self):
        from nba_model.data.hourly_update import _run_vegasinsider_ingestion
        offseason = (
            "Daily NBA Player Prop Odds 2025-26 | Best NBA Prop Odds Sign Up "
            "Subscribe Sign In Bonus Codes Expert Picks Odds Best Sportsbooks "
            "The NBA player props board is not available right now."
        )
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba_data.db")
            self._seed(db_path, offseason)
            summary = _run_vegasinsider_ingestion(db_path)  # must not raise
            self.assertEqual(summary["parsed_rows"], 0)
            self.assertEqual(summary["inserted"], 0)
            self.assertIn(summary["status"], {"success", "no_snapshot"})

    def test_daily_step_matches_hourly(self):
        from nba_model.data.daily_etl import _run_vegasinsider_step
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba_data.db")
            self._seed(db_path, VI_FIXTURE)
            summary = _run_vegasinsider_step(db_path)
            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["inserted"], 33)


class EtlStepWiringTests(unittest.TestCase):
    """The VI step is wired into both pipelines' step order + report contract."""

    def test_hourly_pipeline_lists_vegasinsider_step(self):
        import inspect
        from nba_model.data import hourly_update
        src = inspect.getsource(hourly_update.run_hourly_update)
        self.assertIn("vegasinsider_ingestion", src)
        # Ordered right after the team-line parser, before game-log refresh.
        self.assertLess(src.index("team_line_parser"), src.index("vegasinsider_ingestion"))
        self.assertLess(src.index("vegasinsider_ingestion"), src.index("game_log_refresh"))

    def test_daily_run_reports_vegasinsider_step(self):
        # Skipped when no web-text URLs (VI needs the fetched snapshot); still a
        # first-class step in the report so its counts are always visible.
        from nba_model.data.daily_etl import run_daily_etl
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba_data.db")
            with DatabaseManager(db_path=db_path):
                pass
            report = run_daily_etl(
                db_path=db_path,
                skip_game_logs=True,
                skip_team_defense=True,
                skip_odds=True,
                skip_reverse_engineering=True,
                skip_bulk_results_ingest=True,
                write_report=False,
            )
        self.assertIn("vegasinsider_ingestion", report["steps"])
        self.assertEqual(
            report["steps"]["vegasinsider_ingestion"]["status"], "skipped")
        # A skipped VI step must not flip the overall status.
        self.assertEqual(report["status"], "success")


if __name__ == "__main__":
    unittest.main()
