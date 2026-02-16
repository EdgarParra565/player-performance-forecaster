import unittest
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from nba_model.evaluation.backtest import Backtester
from nba_model.model.feature_engineering import add_rolling_stats
from nba_model.model.probability import prob_over
from nba_model.model.simulation import monte_carlo_over, normalize_distribution_name


class FeatureEngineeringSmokeTests(unittest.TestCase):
    def test_add_rolling_stats_handles_api_aliases(self):
        df = pd.DataFrame(
            {
                "PTS": [10, 20, 30, 40, 50],
                "AST": [3, 4, 5, 6, 7],
                "REB": [5, 6, 7, 8, 9],
                "MIN": [30, 30, 30, 30, 30],
            }
        )

        result = add_rolling_stats(df, window=3)

        self.assertIn("rolling_mean_points", result.columns)
        self.assertIn("rolling_std_points", result.columns)
        self.assertIn("rolling_mean_points_per_minute", result.columns)
        self.assertIn("rolling_mean_pra", result.columns)
        self.assertIn("rolling_std_pra", result.columns)
        self.assertIn("pts_mean", result.columns)
        self.assertIn("pts_std", result.columns)
        self.assertIn("ppm_mean", result.columns)
        self.assertAlmostEqual(result.loc[4, "rolling_mean_points"], 40.0, places=6)
        self.assertAlmostEqual(result.loc[4, "rolling_std_points"], 10.0, places=6)
        self.assertAlmostEqual(result.loc[4, "rolling_mean_pra"], 54.0, places=6)


class ProbabilitySmokeTests(unittest.TestCase):
    def test_prob_over_behaves_as_expected(self):
        self.assertAlmostEqual(prob_over(20.0, 20.0, 5.0), 0.5, places=6)
        self.assertGreater(prob_over(20.0, 22.0, 5.0), 0.5)
        self.assertLess(prob_over(20.0, 18.0, 5.0), 0.5)

    def test_monte_carlo_over_supports_distribution_families(self):
        distributions = [
            "normal",
            "student_t",
            "binomial",
            "poisson",
            "exponential",
            "uniform",
            "lognormal",
            "power_law",
        ]
        for distribution in distributions:
            prob = monte_carlo_over(
                mu=22.0,
                sigma=6.0,
                line=20.5,
                n=3000,
                distribution=distribution,
                sample_size=10,
            )
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_monte_carlo_over_rejects_invalid_distribution(self):
        with self.assertRaises(ValueError):
            monte_carlo_over(
                mu=22.0,
                sigma=6.0,
                line=20.5,
                n=1000,
                distribution="not_a_distribution",
            )

    def test_distribution_alias_normalization(self):
        self.assertEqual(normalize_distribution_name("gaussian"), "normal")
        self.assertEqual(normalize_distribution_name("t"), "student_t")
        self.assertEqual(normalize_distribution_name("pareto"), "power_law")


class BacktestSmokeTests(unittest.TestCase):
    @patch("nba_model.evaluation.backtest.DatabaseManager")
    @patch("nba_model.evaluation.backtest.DataLoader")
    def test_backtester_runs_with_synthetic_data(self, mock_loader_cls, mock_db_cls):
        dates = pd.date_range("2024-01-01", periods=70, freq="D")
        points = np.array([20 + (i % 12) for i in range(70)], dtype=float)
        assists = np.array([5 + (i % 6) for i in range(70)], dtype=float)
        rebounds = np.array([6 + (i % 7) for i in range(70)], dtype=float)
        minutes = np.array([30 + (i % 4) for i in range(70)], dtype=float)

        synthetic = pd.DataFrame(
            {
                "game_date": dates,
                "points": points,
                "assists": assists,
                "rebounds": rebounds,
                "minutes": minutes,
            }
        )

        loader_instance = MagicMock()
        loader_instance.get_player_id.return_value = 2544
        loader_instance.load_player_data.return_value = synthetic
        mock_loader_cls.return_value = loader_instance

        db_instance = MagicMock()
        db_instance.get_market_spread.return_value = None
        mock_db_cls.return_value = db_instance

        backtester = Backtester(
            start_date="2024-02-01",
            end_date="2024-03-05",
            line_value=18.5,
            stat_type="points",
            distribution="student_t",
        )
        metrics = backtester.run_backtest("LeBron James", window=5)

        self.assertGreater(metrics.get("total_games", 0), 0)
        self.assertGreater(metrics.get("bets_made", 0), 0)
        self.assertIn("roi", metrics)
        self.assertEqual(metrics.get("distribution"), "student_t")
        self.assertIn("market_spread_games", metrics)
        self.assertTrue(db_instance.insert_prediction.called)
        last_prediction = db_instance.insert_prediction.call_args[0][0]
        self.assertIn("model_config_json", last_prediction)
        model_config = json.loads(last_prediction["model_config_json"])
        self.assertEqual(model_config.get("distribution"), "student_t")

    @patch("nba_model.evaluation.backtest.DatabaseManager")
    @patch("nba_model.evaluation.backtest.DataLoader")
    def test_backtester_supports_pra_stat_type(self, mock_loader_cls, mock_db_cls):
        dates = pd.date_range("2024-01-01", periods=70, freq="D")
        points = np.array([20 + (i % 12) for i in range(70)], dtype=float)
        assists = np.array([5 + (i % 6) for i in range(70)], dtype=float)
        rebounds = np.array([6 + (i % 7) for i in range(70)], dtype=float)
        minutes = np.array([30 + (i % 4) for i in range(70)], dtype=float)

        synthetic = pd.DataFrame(
            {
                "game_date": dates,
                "points": points,
                "assists": assists,
                "rebounds": rebounds,
                "minutes": minutes,
            }
        )

        loader_instance = MagicMock()
        loader_instance.get_player_id.return_value = 2544
        loader_instance.load_player_data.return_value = synthetic
        mock_loader_cls.return_value = loader_instance

        db_instance = MagicMock()
        db_instance.get_market_spread.return_value = None
        mock_db_cls.return_value = db_instance

        backtester = Backtester(
            start_date="2024-02-01",
            end_date="2024-03-05",
            line_value=34.5,
            stat_type="pra",
        )
        metrics = backtester.run_backtest("LeBron James", window=5)

        self.assertGreater(metrics.get("total_games", 0), 0)
        self.assertIn("roi", metrics)
        self.assertTrue(db_instance.insert_prediction.called)

    @patch("nba_model.evaluation.backtest.DatabaseManager")
    @patch("nba_model.evaluation.backtest.DataLoader")
    def test_backtester_uses_market_spread_when_row_spread_missing(self, mock_loader_cls, mock_db_cls):
        dates = pd.date_range("2024-01-01", periods=70, freq="D")
        points = np.array([20 + (i % 12) for i in range(70)], dtype=float)
        assists = np.array([5 + (i % 6) for i in range(70)], dtype=float)
        rebounds = np.array([6 + (i % 7) for i in range(70)], dtype=float)
        minutes = np.array([30 + (i % 4) for i in range(70)], dtype=float)

        synthetic = pd.DataFrame(
            {
                "game_date": dates,
                "points": points,
                "assists": assists,
                "rebounds": rebounds,
                "minutes": minutes,
            }
        )

        loader_instance = MagicMock()
        loader_instance.get_player_id.return_value = 2544
        loader_instance.load_player_data.return_value = synthetic
        mock_loader_cls.return_value = loader_instance

        db_instance = MagicMock()
        db_instance.get_market_spread.return_value = 8.5
        mock_db_cls.return_value = db_instance

        backtester = Backtester(
            start_date="2024-02-01",
            end_date="2024-03-05",
            line_value=18.5,
            stat_type="points",
        )
        metrics = backtester.run_backtest("LeBron James", window=5)
        self.assertGreater(metrics.get("market_spread_games", 0), 0)
        results_df = backtester.get_results_df()
        self.assertIn("spread_source", results_df.columns)
        self.assertTrue((results_df["spread_source"] == "market_spread").any())


if __name__ == "__main__":
    unittest.main()
