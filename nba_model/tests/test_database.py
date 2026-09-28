"""SQLite insert/query smoke test for DatabaseManager (temp DB only).

This used to be a module-level script: pytest imported it at collection time,
so every suite run opened the LIVE data/database/nba_data.db and inserted a
fabricated LeBron game log (game_id 0022400001, 28/8/10 on 2024-10-22). It now
runs against a throwaway DB; tests/conftest.py fails any test that opens the
live DB path again.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nba_model.data.database.db_manager import DatabaseManager

_SAMPLE_GAME = {
    'player_id': [2544],
    'game_id': ['0022400001'],
    'game_date': ['2024-10-22'],
    'season': ['2024-25'],
    'matchup': ['LAL vs. DEN'],
    'home_away': ['home'],
    'result': ['W'],
    'minutes': [35.5],
    'points': [28],
    'fgm': [10],
    'fga': [20],
    'fg_pct': [0.5],
    'fg3m': [2],
    'fg3a': [6],
    'fg3_pct': [0.333],
    'ftm': [6],
    'fta': [8],
    'ft_pct': [0.75],
    'oreb': [1],
    'dreb': [7],
    'rebounds': [8],
    'assists': [10],
    'steals': [2],
    'blocks': [1],
    'turnovers': [3],
    'plus_minus': [12],
}


class DatabaseSmokeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = str(Path(self._tmp.name) / "nba.db")

    def tearDown(self):
        self._tmp.cleanup()

    def test_insert_player_game_log_and_query(self):
        db = DatabaseManager(db_path=self.db_path)
        try:
            db.insert_player(2544, "LeBron James", "LAL", "F")
            db.insert_game_logs(pd.DataFrame(_SAMPLE_GAME))
            games = db.get_player_games(2544, n_games=5)
        finally:
            db.close()
        self.assertEqual(len(games), 1)
        self.assertEqual(int(games.iloc[0]['points']), 28)


if __name__ == "__main__":
    unittest.main()
