"""Tests for the pure-logic helpers in ``nba_model.simple_ui``.

These helpers back the desktop UI polish (fuzzy autocomplete, date-range
picker, save-snapshot export, Tk-side watchlist) so we test them at the
module level without ever instantiating a Tk root — they're standalone
functions for exactly this reason.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nba_model import simple_ui


class FuzzyMatchPlayersTests(unittest.TestCase):
    NAMES = [
        "LeBron James", "Stephen Curry", "Nikola Jokic", "Jayson Tatum",
        "Jaylen Brown", "Anthony Edwards", "Anthony Davis", "Luka Doncic",
    ]

    def test_empty_query_returns_first_n(self):
        out = simple_ui.fuzzy_match_players("", self.NAMES, limit=3)
        self.assertEqual(out, self.NAMES[:3])

    def test_prefix_beats_substring(self):
        out = simple_ui.fuzzy_match_players("leb", self.NAMES)
        # "LeBron James" starts with "leb" → must be first.
        self.assertEqual(out[0], "LeBron James")

    def test_token_prefix_match(self):
        # Query "ja" should surface "Jayson Tatum" and "Jaylen Brown" — both
        # have a token starting with "ja" — and rank above plain substrings.
        out = simple_ui.fuzzy_match_players("ja", self.NAMES, limit=4)
        self.assertIn("Jayson Tatum", out[:2])
        self.assertIn("Jaylen Brown", out[:2])

    def test_no_match_returns_empty(self):
        self.assertEqual(simple_ui.fuzzy_match_players("zzz", self.NAMES), [])

    def test_case_insensitive(self):
        out = simple_ui.fuzzy_match_players("CURRY", self.NAMES)
        self.assertIn("Stephen Curry", out)

    def test_skips_non_strings(self):
        mixed = ["LeBron James", None, 42, "Stephen Curry"]
        out = simple_ui.fuzzy_match_players("ja", mixed)
        # Filter must not crash on None/int and still return string matches.
        self.assertEqual(out, ["LeBron James"])


class ParseDateRangeTests(unittest.TestCase):
    def test_empty_returns_open_range(self):
        self.assertEqual(simple_ui.parse_date_range(""), (None, None))
        self.assertEqual(simple_ui.parse_date_range("   "), (None, None))

    def test_full_range(self):
        self.assertEqual(
            simple_ui.parse_date_range("2025-01-01..2025-03-15"),
            ("2025-01-01", "2025-03-15"),
        )

    def test_open_end(self):
        self.assertEqual(
            simple_ui.parse_date_range("2025-01-01.."),
            ("2025-01-01", None),
        )

    def test_open_start(self):
        self.assertEqual(
            simple_ui.parse_date_range("..2025-03-15"),
            (None, "2025-03-15"),
        )

    def test_single_date_used_as_both_bounds(self):
        self.assertEqual(
            simple_ui.parse_date_range("2025-03-15"),
            ("2025-03-15", "2025-03-15"),
        )

    def test_accepts_slash_format(self):
        self.assertEqual(
            simple_ui.parse_date_range("2025/03/15..2025/04/01"),
            ("2025-03-15", "2025-04-01"),
        )

    def test_swapped_bounds_raises(self):
        with self.assertRaises(ValueError):
            simple_ui.parse_date_range("2025-04-01..2025-03-15")

    def test_garbage_raises(self):
        with self.assertRaises(ValueError):
            simple_ui.parse_date_range("not-a-date")


class FilterGamesByDateRangeTests(unittest.TestCase):
    def setUp(self):
        self.games = pd.DataFrame({
            "game_date": pd.to_datetime([
                "2025-01-05", "2025-01-15", "2025-02-01",
                "2025-02-20", "2025-03-10",
            ]),
            "matchup": ["LAL vs DEN"] * 5,
        })
        self.values = np.array([22, 28, 31, 19, 25], dtype=float)

    def test_open_range_passes_through(self):
        g, v = simple_ui.filter_games_by_date_range(
            self.games, self.values, None, None)
        self.assertEqual(len(g), 5)
        np.testing.assert_array_equal(v, self.values)

    def test_inclusive_bounds(self):
        g, v = simple_ui.filter_games_by_date_range(
            self.games, self.values, "2025-01-15", "2025-02-20")
        self.assertEqual(list(g["game_date"].dt.strftime("%Y-%m-%d")),
                         ["2025-01-15", "2025-02-01", "2025-02-20"])
        np.testing.assert_array_equal(v, np.array([28, 31, 19]))

    def test_open_end_only(self):
        g, _ = simple_ui.filter_games_by_date_range(
            self.games, self.values, "2025-02-01", None)
        self.assertEqual(len(g), 3)

    def test_empty_games_short_circuits(self):
        g, v = simple_ui.filter_games_by_date_range(
            pd.DataFrame(), np.array([]), "2025-01-01", "2025-12-31")
        self.assertTrue(g.empty)


class TkWatchlistTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "tk_watchlist.json"

    def test_add_and_load(self):
        simple_ui.tk_watchlist_add("LeBron James", path=self.path)
        simple_ui.tk_watchlist_add("Nikola Jokic", path=self.path)
        self.assertEqual(
            simple_ui.tk_watchlist_load(self.path),
            ["LeBron James", "Nikola Jokic"],
        )

    def test_duplicate_add_is_noop(self):
        simple_ui.tk_watchlist_add("LeBron James", path=self.path)
        simple_ui.tk_watchlist_add("LeBron James", path=self.path)
        self.assertEqual(len(simple_ui.tk_watchlist_load(self.path)), 1)

    def test_remove(self):
        simple_ui.tk_watchlist_add("LeBron James", path=self.path)
        simple_ui.tk_watchlist_add("Nikola Jokic", path=self.path)
        simple_ui.tk_watchlist_remove("LeBron James", path=self.path)
        self.assertEqual(
            simple_ui.tk_watchlist_load(self.path), ["Nikola Jokic"])

    def test_clear(self):
        simple_ui.tk_watchlist_add("LeBron James", path=self.path)
        simple_ui.tk_watchlist_clear(path=self.path)
        self.assertEqual(simple_ui.tk_watchlist_load(self.path), [])

    def test_corrupt_file_loads_as_empty(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{not valid json", encoding="utf-8")
        self.assertEqual(simple_ui.tk_watchlist_load(self.path), [])

    def test_caps_at_limit(self):
        for i in range(simple_ui.TK_WATCHLIST_LIMIT + 5):
            simple_ui.tk_watchlist_add(f"Player {i}", path=self.path)
        self.assertEqual(
            len(simple_ui.tk_watchlist_load(self.path)),
            simple_ui.TK_WATCHLIST_LIMIT,
        )


class ExportChartSnapshotTests(unittest.TestCase):
    def test_png_and_csv_written(self):
        from matplotlib.figure import Figure
        fig = Figure(figsize=(3, 2))
        ax = fig.add_subplot(111)
        ax.plot([0, 1, 2], [1, 3, 2])
        games = pd.DataFrame({
            "game_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "matchup": ["A", "B", "C"],
        })
        values = np.array([10.0, 20.0, 30.0])
        with tempfile.TemporaryDirectory() as tmp:
            png, csv = simple_ui.export_chart_snapshot(
                fig, games, values, Path(tmp), base_name="Test/Player")
            self.assertTrue(png.exists())
            self.assertTrue(csv.exists())
            # Filename sanitized: no slashes in base name.
            self.assertNotIn("/", png.name)
            # CSV has the prepended value column.
            df = pd.read_csv(csv)
            self.assertEqual(list(df["value"]), [10.0, 20.0, 30.0])

    def test_empty_games_still_emits_csv(self):
        from matplotlib.figure import Figure
        fig = Figure(figsize=(2, 2))
        fig.add_subplot(111).plot([0, 1], [0, 1])
        with tempfile.TemporaryDirectory() as tmp:
            png, csv = simple_ui.export_chart_snapshot(
                fig, pd.DataFrame(), np.array([]), Path(tmp),
                base_name="empty")
            self.assertTrue(png.exists())
            self.assertTrue(csv.exists())
            # File is present even though no rows — caller-friendly.
            self.assertEqual(csv.read_text().strip(), "")


if __name__ == "__main__":
    unittest.main()
