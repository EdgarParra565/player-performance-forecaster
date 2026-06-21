"""Tests for WS6 multi-sport scaffolding.

Covers the idempotent ``sport`` column migration, per-(book, sport) scraper
resolution, and registry-driven stat validation for non-NBA sports. The NFL
ingestion + scrapers themselves are deferred (need nfl_data_py + live
snapshots), so this guards the scaffolding they'll plug into.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.web import input_validation as iv
from nba_model import scrapers


class SportColumnMigrationTests(unittest.TestCase):
    def test_all_tables_get_sport_column_default_nba(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                for table in DatabaseManager._SPORT_COLUMN_TABLES:
                    cols = {
                        r[1] for r in db.conn.execute(
                            f"PRAGMA table_info({table})").fetchall()
                    }
                    self.assertIn("sport", cols, f"{table} missing sport")

    def test_default_value_is_nba_on_insert(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                db.conn.execute(
                    "INSERT INTO players (player_id, name) VALUES (1, 'X')")
                db.conn.commit()
                sport = db.conn.execute(
                    "SELECT sport FROM players WHERE player_id = 1").fetchone()[0]
        self.assertEqual(sport, "nba")

    def test_migration_idempotent_on_existing_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path):
                pass
            # Re-open (re-runs the migration) — must not raise or duplicate.
            with DatabaseManager(db_path=db_path) as db:
                cols = [
                    r[1] for r in db.conn.execute(
                        "PRAGMA table_info(game_logs)").fetchall()
                ]
                self.assertEqual(cols.count("sport"), 1)

    def test_migration_upgrades_legacy_table_without_sport(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "legacy.db")
            # Simulate a pre-migration DB: create predictions WITHOUT sport.
            conn = sqlite3.connect(db_path)
            conn.execute(
                "CREATE TABLE predictions (prediction_id INTEGER PRIMARY KEY, "
                "player_id INTEGER, game_date DATE, stat_type TEXT)")
            conn.commit()
            conn.close()
            with DatabaseManager(db_path=db_path) as db:
                cols = {
                    r[1] for r in db.conn.execute(
                        "PRAGMA table_info(predictions)").fetchall()
                }
        self.assertIn("sport", cols)


class ScraperSportResolutionTests(unittest.TestCase):
    def test_existing_scrapers_default_to_nba(self):
        s = scrapers.get_scraper_by_name("prizepicks")
        self.assertIsNotNone(s)
        self.assertEqual(s.sport, "nba")

    def test_book_sport_lookup(self):
        s = scrapers.get_scraper_for_book_sport("draftkings", "nba")
        self.assertIsNotNone(s)
        self.assertEqual(s.name, "draftkings")
        # No NFL config registered yet → None (not a wrong-sport match).
        self.assertIsNone(scrapers.get_scraper_for_book_sport("draftkings", "nfl"))

    def test_url_resolution_sport_filter(self):
        # NBA filter still resolves; NFL filter finds nothing yet.
        nba = scrapers.get_scraper_for_url(
            "https://app.prizepicks.com/board/nba", sport="nba")
        self.assertIsNotNone(nba)
        self.assertIsNone(
            scrapers.get_scraper_for_url(
                "https://app.prizepicks.com/board/nba", sport="nfl"))


class SportAwareStatValidationTests(unittest.TestCase):
    def test_nba_unchanged(self):
        self.assertEqual(iv.validate_stat_type("points"), "points")
        self.assertEqual(iv.validate_stat_type("3pm"), "three_pointers_made")

    def test_nfl_stats_validate_against_registry(self):
        from sports import get_sport
        nfl_stats = get_sport("nfl").stat_types
        self.assertTrue(nfl_stats, "NFL registry should list stat types")
        # A real NFL stat passes under sport='nfl' but fails under NBA.
        sample = nfl_stats[0]
        self.assertEqual(iv.validate_stat_type(sample, sport="nfl"), sample.lower())
        with self.assertRaises(iv.ValidationError):
            iv.validate_stat_type(sample, sport="nba")

    def test_nba_stat_rejected_under_nfl(self):
        with self.assertRaises(iv.ValidationError):
            iv.validate_stat_type("points", sport="nfl")


if __name__ == "__main__":
    unittest.main()
