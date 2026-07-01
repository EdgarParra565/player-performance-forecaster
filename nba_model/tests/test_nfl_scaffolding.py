"""Tests for WS6 NFL scaffolding: guarded ingestion + NFL prop scrapers.

Everything here must pass WITHOUT ``nfl_data_py`` installed and without live
data — the dependency is imported lazily and the prop parser is exercised
against committed fixtures.
"""

import unittest

import pandas as pd

from nba_model.data import nfl_results_ingestion as nfl
from nba_model.scrapers import (
    get_scraper_for_book_sport,
    get_scraper_for_url,
)
from nba_model.scrapers.nfl_props import canonical_nfl_stat, preprocess_nfl_props


class NflIngestionGuardTests(unittest.TestCase):
    def test_module_imports_without_dependency(self):
        # Importing the module must never require nfl_data_py.
        self.assertTrue(hasattr(nfl, "ingest_all"))

    def test_dependency_absent_here(self):
        # Sanity: the optional dep is not installed in this environment.
        self.assertFalse(nfl.nfl_data_py_available())

    def test_fetch_raises_actionable_error_without_dep(self):
        with self.assertRaises(RuntimeError) as ctx:
            nfl.fetch_weekly_player_stats([2025])
        self.assertIn("nfl_data_py", str(ctx.exception))
        self.assertIn("pip install", str(ctx.exception))

    def test_transform_is_pure_and_dependency_free(self):
        df = pd.DataFrame(
            [
                {
                    "player_display_name": "Patrick Mahomes",
                    "player_id": "00-0033873",
                    "season": 2025,
                    "week": 1,
                    "recent_team": "KC",
                    "position": "QB",
                    "passing_yards": 291.0,
                    "passing_tds": 3,
                    "rushing_yards": 18.0,
                },
                {
                    "player_display_name": "Christian McCaffrey",
                    "player_id": "00-0034844",
                    "season": 2025,
                    "week": 1,
                    "recent_team": "SF",
                    "position": "RB",
                    "rushing_yards": 102.0,
                    "carries": 21,
                    "receiving_yards": 44.0,
                    "receptions": 5,
                },
            ]
        )
        rows = nfl.transform_weekly_to_player_logs(df)
        self.assertEqual(len(rows), 2)
        mahomes = next(r for r in rows if r["player_name"] == "Patrick Mahomes")
        self.assertEqual(mahomes["sport"], "nfl")
        self.assertEqual(mahomes["passing_yards"], 291.0)
        self.assertEqual(mahomes["passing_touchdowns"], 3.0)
        self.assertEqual(mahomes["position"], "QB")
        mccaffrey = next(r for r in rows if r["player_name"] == "Christian McCaffrey")
        self.assertEqual(mccaffrey["rushing_attempts"], 21.0)
        self.assertEqual(mccaffrey["receptions"], 5.0)

    def test_transform_empty_frame(self):
        self.assertEqual(nfl.transform_weekly_to_player_logs(pd.DataFrame()), [])


class NflPropPreprocessTests(unittest.TestCase):
    FIXTURE = (
        "Patrick Mahomes Passing Yards 274.5 Over "
        "Patrick Mahomes Passing Yards 274.5 Under "
        "Christian McCaffrey Rushing Yards 89.5 Over "
        "Justin Jefferson Receptions 6.5 Under"
    )

    def test_canonical_stat_mapping(self):
        self.assertEqual(canonical_nfl_stat("Passing Yards"), "passing_yards")
        self.assertEqual(canonical_nfl_stat("rec yards"), "receiving_yards")
        self.assertIsNone(canonical_nfl_stat("points"))  # NBA stat, not NFL

    def test_preprocess_emits_canonical_segments(self):
        out = preprocess_nfl_props(self.FIXTURE)
        self.assertIn("Patrick Mahomes 274.5 passing_yards over", out)
        self.assertIn("Patrick Mahomes 274.5 passing_yards under", out)
        self.assertIn("Christian McCaffrey 89.5 rushing_yards over", out)
        self.assertIn("Justin Jefferson 6.5 receptions under", out)

    def test_preprocess_dedupes_repeats(self):
        doubled = self.FIXTURE + " " + self.FIXTURE
        once = preprocess_nfl_props(self.FIXTURE).split()
        twice = preprocess_nfl_props(doubled).split()
        self.assertEqual(len(once), len(twice))


class NflScraperRegistrationTests(unittest.TestCase):
    def test_nfl_book_configs_registered_and_distinct(self):
        for book in ("fanduel", "draftkings"):
            nfl_cfg = get_scraper_for_book_sport(book, "nfl")
            nba_cfg = get_scraper_for_book_sport(book, "nba")
            self.assertIsNotNone(nfl_cfg, f"{book} nfl config missing")
            self.assertEqual(nfl_cfg.sport, "nfl")
            self.assertIsNot(nfl_cfg, nba_cfg)
            # Props-first: NFL config carries a prop preprocessor.
            self.assertIsNotNone(nfl_cfg.prop_preprocess)

    def test_default_url_resolution_stays_nba(self):
        # No sport filter → the live NBA config must still win on shared domains.
        for url in (
            "https://sportsbook.draftkings.com/leagues/basketball/nba",
            "https://sportsbook.fanduel.com/navigation/nba",
        ):
            self.assertEqual(get_scraper_for_url(url).sport, "nba")


if __name__ == "__main__":
    unittest.main()
