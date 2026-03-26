"""Unit tests for Backtester that do not require network access."""

import unittest
from unittest.mock import patch, MagicMock

import pandas as pd


class TestBacktesterOffline(unittest.TestCase):
    """Validate Backtester construction and methods without live API calls."""

    @patch("nba_model.evaluation.backtest.DatabaseManager")
    @patch("nba_model.evaluation.backtest.DataLoader")
    def test_backtester_init_does_not_hit_network(self, mock_loader_cls, mock_db_cls):
        mock_loader_cls.return_value = MagicMock()
        mock_db_cls.return_value = MagicMock()

        from nba_model.evaluation.backtest import Backtester

        bt = Backtester(
            start_date="2024-11-01",
            end_date="2024-12-15",
            line_value=25.5,
            stat_type="points",
        )
        self.assertIsNotNone(bt)
        self.assertEqual(bt.stat_type, "points")

    @patch("nba_model.evaluation.backtest.DatabaseManager")
    @patch("nba_model.evaluation.backtest.DataLoader")
    def test_backtester_run_with_synthetic_data(self, mock_loader_cls, mock_db_cls):
        dates = pd.date_range("2024-10-15", periods=30, freq="2D")
        game_logs = pd.DataFrame(
            {
                "GAME_DATE": dates,
                "PTS": [25 + i % 7 for i in range(30)],
                "MIN": [34.0] * 30,
                "AST": [7] * 30,
                "REB": [8] * 30,
            }
        )
        mock_loader = MagicMock()
        mock_loader.get_game_logs.return_value = game_logs
        mock_loader.load_player_data.return_value = game_logs
        mock_loader.get_player_id.return_value = 2544
        mock_loader_cls.return_value = mock_loader

        mock_db = MagicMock()
        mock_db.insert_prediction = MagicMock()
        mock_db_cls.return_value = mock_db

        from nba_model.evaluation.backtest import Backtester

        bt = Backtester(
            start_date="2024-11-01",
            end_date="2024-12-15",
            line_value=25.5,
            stat_type="points",
        )
        bt.loader = mock_loader
        bt.db = mock_db

        metrics = bt.run_backtest("LeBron James", window=5)
        self.assertIsInstance(metrics, dict)

    @patch("nba_model.evaluation.backtest.DatabaseManager")
    @patch("nba_model.evaluation.backtest.DataLoader")
    def test_backtester_with_empty_data_raises(self, mock_loader_cls, mock_db_cls):
        mock_loader = MagicMock()
        mock_loader.get_game_logs.return_value = pd.DataFrame()
        mock_loader.load_player_data.return_value = pd.DataFrame()
        mock_loader.get_player_id.return_value = None
        mock_loader_cls.return_value = mock_loader

        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from nba_model.evaluation.backtest import Backtester

        bt = Backtester(
            start_date="2024-11-01",
            end_date="2024-12-15",
            line_value=25.5,
            stat_type="points",
        )
        bt.loader = mock_loader
        bt.db = mock_db

        with self.assertRaises(KeyError):
            bt.run_backtest("Nonexistent Player", window=5)

    @patch("nba_model.evaluation.backtest.DatabaseManager")
    @patch("nba_model.evaluation.backtest.DataLoader")
    def test_backtester_rejects_future_start_date(self, mock_loader_cls, mock_db_cls):
        mock_loader_cls.return_value = MagicMock()
        mock_db_cls.return_value = MagicMock()

        from nba_model.evaluation.backtest import Backtester

        with self.assertRaises(ValueError):
            Backtester(
                start_date="2099-01-01",
                end_date="2099-12-31",
                line_value=25.5,
                stat_type="points",
            )


if __name__ == "__main__":
    unittest.main()
