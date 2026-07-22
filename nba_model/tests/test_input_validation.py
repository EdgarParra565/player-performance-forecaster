"""Tests for the chart-input validators in nba_model.web.input_validation."""

import unittest

from nba_model.web import input_validation as iv


class StatTypeTests(unittest.TestCase):
    def test_canonical_known_stats_pass(self):
        for stat in ("points", "assists", "rebounds", "pra", "minutes"):
            self.assertEqual(iv.validate_stat_type(stat), stat)

    def test_aliases_collapse_to_canonical(self):
        self.assertEqual(iv.validate_stat_type("3pm"), "three_pointers_made")
        self.assertEqual(iv.validate_stat_type("Three Pointers"), "three_pointers_made")
        self.assertEqual(iv.validate_stat_type("FGM"), "field_goals_made")

    def test_unknown_stat_rejected(self):
        with self.assertRaises(iv.ValidationError):
            iv.validate_stat_type("steals_per_minute")
        with self.assertRaises(iv.ValidationError):
            iv.validate_stat_type("")
        with self.assertRaises(iv.ValidationError):
            iv.validate_stat_type(None)

    def test_allowed_subset_filters(self):
        # Free-tier preview list
        self.assertEqual(
            iv.validate_stat_type("points", allowed=["points"]),
            "points",
        )
        with self.assertRaises(iv.ValidationError):
            iv.validate_stat_type("rebounds", allowed=["points"])


class TeamCodeTests(unittest.TestCase):
    def test_normalizes_case_and_punctuation(self):
        self.assertEqual(iv.validate_team_code("nyk"), "NYK")
        self.assertEqual(iv.validate_team_code("Nyk."), "NYK")
        self.assertEqual(iv.validate_team_code(" lal "), "LAL")

    def test_rejects_unknown_code(self):
        for bad in ("XYZ", "", None, "F00", "12"):
            with self.assertRaises(iv.ValidationError):
                iv.validate_team_code(bad)


class SeasonTests(unittest.TestCase):
    def test_canonical_seasons_pass(self):
        self.assertEqual(iv.validate_season("2024-25"), "2024-25")
        self.assertEqual(iv.validate_season("1999-00"), "1999-00")

    def test_rejects_inconsistent_segments(self):
        # second segment must equal (start + 1) mod 100
        with self.assertRaises(iv.ValidationError):
            iv.validate_season("2024-26")
        with self.assertRaises(iv.ValidationError):
            iv.validate_season("2024-24")

    def test_rejects_malformed(self):
        for bad in ("", "2024", "24-25", "2024/25", None):
            with self.assertRaises(iv.ValidationError):
                iv.validate_season(bad)


class RollingWindowTests(unittest.TestCase):
    def test_default_when_blank(self):
        self.assertEqual(iv.validate_rolling_window(None), 5)
        self.assertEqual(iv.validate_rolling_window(""), 5)

    def test_capped_at_hard_limit(self):
        self.assertEqual(
            iv.validate_rolling_window(9999),
            iv.ROLLING_WINDOW_HARD_CAP,
        )

    def test_rejects_negative_or_nan(self):
        with self.assertRaises(iv.ValidationError):
            iv.validate_rolling_window(0)
        with self.assertRaises(iv.ValidationError):
            iv.validate_rolling_window(-1)
        with self.assertRaises(iv.ValidationError):
            iv.validate_rolling_window("abc")


class NGamesIntegrationTests(unittest.TestCase):
    """Sanity-check the existing n_games validator alongside the new ones."""

    def test_caps_at_hard_max(self):
        self.assertEqual(iv.validate_n_games(99999), iv.N_GAMES_HARD_CAP)

    def test_rejects_below_min(self):
        with self.assertRaises(iv.ValidationError):
            iv.validate_n_games(2, min_value=3)


class MinPlayersTests(unittest.TestCase):
    def test_default_when_none(self):
        self.assertEqual(iv.validate_min_players(None), 5)

    def test_coerces_and_caps_at_roster(self):
        self.assertEqual(iv.validate_min_players("7"), 7)
        self.assertEqual(iv.validate_min_players(99), 15)

    def test_rejects_below_one_or_nan(self):
        for bad in (0, -3, float("nan"), "abc"):
            with self.assertRaises(iv.ValidationError):
                iv.validate_min_players(bad)


class SinceHoursTests(unittest.TestCase):
    def test_default_when_none(self):
        self.assertEqual(iv.validate_since_hours(None), 48.0)

    def test_coerces_and_caps_at_thirty_days(self):
        self.assertEqual(iv.validate_since_hours("12.5"), 12.5)
        self.assertEqual(iv.validate_since_hours(10_000), 24.0 * 30.0)

    def test_rejects_non_positive_or_nan(self):
        for bad in (0, -1, float("inf"), "x"):
            with self.assertRaises(iv.ValidationError):
                iv.validate_since_hours(bad)


if __name__ == "__main__":
    unittest.main()
