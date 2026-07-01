"""Tests for MLB prop preprocessing + scraper registration (WS6 MLB-first).

Fixtures only — no live network. Validates both line shapes (over/under +
yes/no), hitter vs pitcher routing, registry-driven stat validation, and that
the NBA default scraper resolution is unaffected.
"""

import unittest

from nba_model.scrapers import get_scraper_for_book_sport, get_scraper_for_url
from nba_model.scrapers.mlb_props import (
    canonical_mlb_stat,
    preprocess_mlb_props,
    stat_group,
)
from nba_model.data.mlb_park_factors import adjust_for_park, park_factor
from nba_model.web import input_validation as iv


class MlbPropPreprocessTests(unittest.TestCase):
    def test_over_under_orderings(self):
        for text, expected in [
            ("Aaron Judge Total Bases Over 1.5", "Aaron Judge 1.5 total_bases over"),
            ("Mookie Betts Over 1.5 Hits", "Mookie Betts 1.5 hits over"),
            ("Shohei Ohtani 2.5 Total Bases Under", "Shohei Ohtani 2.5 total_bases under"),
        ]:
            self.assertIn(expected, preprocess_mlb_props(text))

    def test_pitcher_strikeouts_not_misparsed_as_batter(self):
        out = preprocess_mlb_props("Gerrit Cole Pitcher Strikeouts Over 6.5")
        self.assertIn("Gerrit Cole 6.5 strikeouts_pitcher over", out)
        self.assertNotIn("strikeouts_batter", out)

    def test_yes_no_markets_map_to_bernoulli_line(self):
        self.assertIn(
            "Aaron Judge 0.5 anytime_home_run over",
            preprocess_mlb_props("Aaron Judge To Hit A Home Run Yes"),
        )
        self.assertIn(
            "Juan Soto 0.5 anytime_home_run under",
            preprocess_mlb_props("Juan Soto Home Run No"),
        )

    def test_dedupe(self):
        once = preprocess_mlb_props("Mookie Betts Over 1.5 Hits").split()
        twice = preprocess_mlb_props("Mookie Betts Over 1.5 Hits Mookie Betts Over 1.5 Hits").split()
        self.assertEqual(len(once), len(twice))

    def test_stat_group_routing(self):
        self.assertEqual(stat_group("strikeouts_pitcher"), "pitching")
        self.assertEqual(stat_group("earned_runs"), "pitching")
        self.assertEqual(stat_group("total_bases"), "hitting")
        self.assertEqual(stat_group("home_runs"), "hitting")
        self.assertEqual(stat_group("anytime_home_run"), "combined")

    def test_canonical_distinguishes_batter_vs_pitcher_strikeouts(self):
        self.assertEqual(canonical_mlb_stat("strikeouts"), "strikeouts_batter")
        self.assertEqual(canonical_mlb_stat("pitcher strikeouts"), "strikeouts_pitcher")
        self.assertIsNone(canonical_mlb_stat("points"))  # NBA stat


class MlbStatRegistryValidationTests(unittest.TestCase):
    """Every stat the parser can emit must validate under sport='mlb'."""

    EMITTED = [
        "total_bases", "hits", "home_runs", "rbis", "runs_scored",
        "stolen_bases", "walks", "strikeouts_batter", "singles",
        "strikeouts_pitcher", "earned_runs", "outs_recorded",
        "hits_allowed", "walks_allowed", "wins", "anytime_home_run",
    ]

    def test_emitted_stats_validate_under_mlb(self):
        for stat in self.EMITTED:
            self.assertEqual(iv.validate_stat_type(stat, sport="mlb"), stat.lower())

    def test_mlb_stats_rejected_under_nba(self):
        with self.assertRaises(iv.ValidationError):
            iv.validate_stat_type("total_bases", sport="nba")

    def test_nba_stat_rejected_under_mlb(self):
        with self.assertRaises(iv.ValidationError):
            iv.validate_stat_type("points", sport="mlb")


class MlbScraperRegistrationTests(unittest.TestCase):
    def test_mlb_configs_registered_and_distinct(self):
        for book in ("draftkings", "fanduel"):
            mlb_cfg = get_scraper_for_book_sport(book, "mlb")
            nba_cfg = get_scraper_for_book_sport(book, "nba")
            nfl_cfg = get_scraper_for_book_sport(book, "nfl")
            self.assertIsNotNone(mlb_cfg, f"{book} mlb config missing")
            self.assertEqual(mlb_cfg.sport, "mlb")
            self.assertIsNotNone(mlb_cfg.prop_preprocess)
            self.assertIsNot(mlb_cfg, nba_cfg)
            self.assertIsNot(mlb_cfg, nfl_cfg)

    def test_default_url_resolution_stays_nba(self):
        for url in (
            "https://sportsbook.draftkings.com/leagues/basketball/nba",
            "https://sportsbook.fanduel.com/navigation/nba",
        ):
            self.assertEqual(get_scraper_for_url(url).sport, "nba")

    def test_mlb_url_filter_resolves_mlb_config(self):
        cfg = get_scraper_for_url(
            "https://sportsbook.draftkings.com/leagues/baseball/mlb", sport="mlb"
        )
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.sport, "mlb")


class MlbParkFactorTests(unittest.TestCase):
    def test_known_parks(self):
        self.assertGreater(park_factor("COL"), 1.0)   # Coors inflates
        self.assertLess(park_factor("SF"), 1.0)        # Oracle suppresses
        self.assertEqual(park_factor("ZZZ"), 1.0)      # unknown → neutral

    def test_adjust_only_park_sensitive_hitter_stats(self):
        # total_bases is park-sensitive → scaled at Coors.
        self.assertAlmostEqual(adjust_for_park(10.0, "COL", "total_bases"), 11.2, places=3)
        # pitcher strikeouts not park-sensitive → unchanged.
        self.assertEqual(adjust_for_park(7.0, "COL", "strikeouts_pitcher"), 7.0)
        # yes/no market unchanged.
        self.assertEqual(adjust_for_park(0.5, "COL", "anytime_home_run"), 0.5)


if __name__ == "__main__":
    unittest.main()
