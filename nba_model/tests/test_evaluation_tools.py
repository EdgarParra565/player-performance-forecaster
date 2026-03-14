import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from nba_model.evaluation.line_comparison import (
    build_book_vs_book_comparison,
    build_model_vs_book_comparison,
)
from nba_model.evaluation.monthly_diagnostics import build_monthly_diagnostics
from nba_model.evaluation.run_batch_backtest import run_batch_backtest
from nba_model.evaluation.run_distribution_sweep import build_distribution_summary
from nba_model.evaluation.run_real_data_benchmark import build_player_window_ci_summary


class EvaluationToolsTests(unittest.TestCase):
    @patch("nba_model.evaluation.run_batch_backtest.Backtester")
    @patch("nba_model.evaluation.run_batch_backtest.DataLoader")
    def test_run_batch_backtest_handles_no_bets_metrics(
        self,
        mock_loader_cls,
        mock_backtester_cls,
    ):
        loader = MagicMock()
        loader.get_player_id.return_value = 1
        loader.load_player_data.return_value = pd.DataFrame(
            [{"game_date": "2025-01-01", "points": 20}]
        )
        mock_loader_cls.return_value = loader

        backtester = MagicMock()
        backtester.run_backtest.return_value = {
            "total_games": 10,
            "bets_made": 0,
        }
        mock_backtester_cls.return_value = backtester

        results_df, failures_df = run_batch_backtest(
            players=["LeBron James"],
            windows=[5],
            stat_types=["points"],
            distributions=["normal"],
            start_date="2025-01-01",
            end_date="2025-02-01",
            history_games=120,
        )

        self.assertTrue(failures_df.empty)
        self.assertEqual(len(results_df), 1)
        self.assertIn("roi", results_df.columns)
        self.assertIn("win_rate", results_df.columns)
        self.assertEqual(int(results_df.iloc[0]["bets_made"]), 0)
        self.assertTrue(pd.isna(results_df.iloc[0]["roi"]))

    def test_player_window_ci_summary(self):
        df = pd.DataFrame(
            [
                {
                    "player_name": "LeBron James",
                    "window": 5,
                    "stat_type": "points",
                    "wins": 6,
                    "losses": 4,
                    "bets_made": 10,
                    "total_games": 12,
                    "roi": 5.5,
                    "brier_score": 0.21,
                    "win_rate": 0.60,
                },
                {
                    "player_name": "LeBron James",
                    "window": 5,
                    "stat_type": "assists",
                    "wins": 5,
                    "losses": 5,
                    "bets_made": 10,
                    "total_games": 12,
                    "roi": 1.1,
                    "brier_score": 0.24,
                    "win_rate": 0.50,
                },
            ]
        )
        summary = build_player_window_ci_summary(df)
        self.assertEqual(len(summary), 1)
        self.assertIn("win_rate_ci_lower", summary.columns)
        self.assertEqual(int(summary.iloc[0]["wins"]), 11)

    def test_distribution_summary(self):
        df = pd.DataFrame(
            [
                {
                    "distribution": "normal",
                    "stat_type": "points",
                    "player_name": "A",
                    "wins": 8,
                    "losses": 4,
                    "bets_made": 12,
                    "total_games": 15,
                    "roi": 8.0,
                    "win_rate": 0.67,
                    "brier_score": 0.20,
                    "significant_at_5pct": True,
                },
                {
                    "distribution": "normal",
                    "stat_type": "points",
                    "player_name": "B",
                    "wins": 7,
                    "losses": 5,
                    "bets_made": 12,
                    "total_games": 15,
                    "roi": 6.0,
                    "win_rate": 0.58,
                    "brier_score": 0.23,
                    "significant_at_5pct": False,
                },
            ]
        )
        summary = build_distribution_summary(df)
        self.assertEqual(len(summary), 1)
        self.assertIn("avg_brier_score", summary.columns)
        self.assertIn("win_rate_ci_lower", summary.columns)

    def test_book_vs_book_comparison(self):
        lines = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "player_name": "Player",
                    "game_date": "2025-01-01",
                    "stat_type": "points",
                    "book": "BookA",
                    "line_value": 25.5,
                    "over_odds": -110,
                    "under_odds": -110,
                },
                {
                    "player_id": 1,
                    "player_name": "Player",
                    "game_date": "2025-01-01",
                    "stat_type": "points",
                    "book": "BookB",
                    "line_value": 26.5,
                    "over_odds": -105,
                    "under_odds": -115,
                },
            ]
        )
        result = build_book_vs_book_comparison(lines_df=lines, min_books=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["best_over_line_book"], "BookA")
        self.assertEqual(result.iloc[0]["best_under_line_book"], "BookB")

    def test_model_vs_book_comparison(self):
        lines = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "player_name": "Player",
                    "game_date": "2025-01-01",
                    "stat_type": "points",
                    "book": "BookA",
                    "line_value": 24.5,
                    "over_odds": -110,
                    "under_odds": -110,
                },
                {
                    "player_id": 1,
                    "player_name": "Player",
                    "game_date": "2025-01-01",
                    "stat_type": "points",
                    "book": "BookB",
                    "line_value": 26.5,
                    "over_odds": -110,
                    "under_odds": -110,
                },
            ]
        )
        predictions = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "player_name": "Player",
                    "game_date": "2025-01-01",
                    "stat_type": "points",
                    "predicted_mean": 27.0,
                    "predicted_std": 4.0,
                    "prob_over": 0.62,
                    "model_line": 25.5,
                    "distribution": "normal",
                }
            ]
        )
        result = build_model_vs_book_comparison(lines_df=lines, predictions_df=predictions, edge_threshold=0.0)
        self.assertEqual(len(result), 1)
        self.assertIn(result.iloc[0]["recommended_side"], {"over", "under"})
        self.assertIn("best_edge", result.columns)

    def test_monthly_diagnostics_builder(self):
        df = pd.DataFrame(
            [
                {
                    "prediction_id": 1,
                    "player_id": 1,
                    "player_name": "P1",
                    "game_date": "2025-01-10",
                    "stat_type": "points",
                    "predicted_mean": 25.0,
                    "predicted_std": 5.0,
                    "prob_over": 0.61,
                    "line_value": 24.5,
                    "actual_value": 27.0,
                },
                {
                    "prediction_id": 2,
                    "player_id": 1,
                    "player_name": "P1",
                    "game_date": "2025-01-20",
                    "stat_type": "points",
                    "predicted_mean": 22.0,
                    "predicted_std": 5.0,
                    "prob_over": 0.40,
                    "line_value": 23.5,
                    "actual_value": 19.0,
                },
                {
                    "prediction_id": 3,
                    "player_id": 2,
                    "player_name": "P2",
                    "game_date": "2025-02-08",
                    "stat_type": "points",
                    "predicted_mean": 21.0,
                    "predicted_std": 4.0,
                    "prob_over": 0.57,
                    "line_value": 20.5,
                    "actual_value": 18.0,
                },
            ]
        )
        df["game_date"] = pd.to_datetime(df["game_date"])
        monthly, equity, overall = build_monthly_diagnostics(df, edge_threshold=0.55)
        self.assertGreaterEqual(len(monthly), 2)
        self.assertIn("roi", monthly.columns)
        self.assertIn("max_drawdown", overall)
        self.assertIn("drawdown", equity.columns)


if __name__ == "__main__":
    unittest.main()
