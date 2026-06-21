"""Regression test for prop_board's betting-lines fetch.

Guards the bug where multi-value ``IN`` clauses were bound as a tuple to a
single named placeholder — sqlite3 cannot expand that and raised at runtime,
breaking the prop board / hourly recompute on every call.
"""

import tempfile
import unittest
from pathlib import Path

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model.prop_board import _fetch_betting_lines_for_game


class FetchBettingLinesForGameTests(unittest.TestCase):
    def _seed(self, db_path: str):
        with DatabaseManager(db_path=db_path) as db:
            db.conn.executemany(
                "INSERT INTO players (player_id, name, team) VALUES (?, ?, ?)",
                [(2544, "LeBron James", "LAL"), (1629029, "Luka Doncic", "DAL")],
            )
            db.conn.commit()
            db.insert_betting_lines_records([
                {"player_id": 2544, "game_date": "2025-03-01", "book": "FanDuel",
                 "stat_type": "points", "line_value": 25.5,
                 "over_odds": -110, "under_odds": -110},
                {"player_id": 2544, "game_date": "2025-03-01", "book": "DraftKings",
                 "stat_type": "assists", "line_value": 7.5,
                 "over_odds": -115, "under_odds": -105},
                {"player_id": 1629029, "game_date": "2025-03-01", "book": "FanDuel",
                 "stat_type": "points", "line_value": 30.5,
                 "over_odds": -110, "under_odds": -110},
            ])

    def test_multi_stat_in_clause_runs_and_returns_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            self._seed(db_path)
            # Two stat_types exercises the multi-placeholder IN expansion.
            df = _fetch_betting_lines_for_game(
                db_path, game_date="2025-03-01",
                stat_types=["points", "assists"],
            )
        self.assertFalse(df.empty)
        self.assertEqual(set(df["stat_type"]), {"points", "assists"})
        self.assertIn("player_name", df.columns)

    def test_team_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            self._seed(db_path)
            df = _fetch_betting_lines_for_game(
                db_path, game_date="2025-03-01", stat_types=["points"],
                home_team="LAL", away_team="DEN",
            )
        # Only LAL player matches the team filter; Luka (DAL) excluded.
        self.assertEqual(set(df["player_name"]), {"LeBron James"})

    def test_book_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            self._seed(db_path)
            df = _fetch_betting_lines_for_game(
                db_path, game_date="2025-03-01",
                stat_types=["points", "assists"], books=["FanDuel"],
            )
        self.assertEqual(set(df["book"]), {"FanDuel"})


if __name__ == "__main__":
    unittest.main()
