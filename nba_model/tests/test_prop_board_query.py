"""Regression tests for ``_fetch_betting_lines_for_game`` IN-list binding.

sqlite cannot bind a tuple/list to a single named placeholder, so the
``stat_type``/``book`` IN-filters must expand to one placeholder per value.
A regression here previously raised
``Error binding parameter: type 'tuple' is not supported`` and broke the
hourly ``prediction_recompute`` step.
"""

import os
import tempfile
import unittest

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model.prop_board import _fetch_betting_lines_for_game


class FetchBettingLinesInBindingTests(unittest.TestCase):
    def setUp(self):
        self.db_path = os.path.join(tempfile.mkdtemp(), "t.db")
        self.db = DatabaseManager(self.db_path)
        self.db.conn.executescript(
            """
            INSERT INTO players (player_id, name, team) VALUES
                (1, 'LeBron James', 'LAL'),
                (2, 'Stephen Curry', 'GSW');
            INSERT INTO betting_lines
                (player_id, game_date, book, stat_type, line_value, over_odds, under_odds)
            VALUES
                (1, '2026-04-01', 'draftkings', 'points',   25.5, -110, -110),
                (1, '2026-04-01', 'fanduel',    'assists',   7.5, -115, -105),
                (2, '2026-04-01', 'draftkings', 'rebounds',  5.5, -120, 100),
                (2, '2026-04-01', 'betmgm',     'points',   28.5, -110, -110);
            """
        )
        self.db.conn.commit()

    def test_multi_stat_in_filter(self):
        df = _fetch_betting_lines_for_game(
            db_path=self.db_path,
            game_date="2026-04-01",
            stat_types=["points", "assists", "rebounds", "pra"],
        )
        self.assertEqual(len(df), 4)

    def test_single_stat_narrows(self):
        df = _fetch_betting_lines_for_game(
            db_path=self.db_path, game_date="2026-04-01", stat_types=["points"]
        )
        self.assertEqual(len(df), 2)
        self.assertEqual(set(df["stat_type"]), {"points"})

    def test_books_in_filter(self):
        df = _fetch_betting_lines_for_game(
            db_path=self.db_path,
            game_date="2026-04-01",
            stat_types=["points", "assists", "rebounds"],
            books=["draftkings", "fanduel"],
        )
        self.assertEqual(set(df["book"]), {"draftkings", "fanduel"})
        self.assertEqual(len(df), 3)


if __name__ == "__main__":
    unittest.main()
