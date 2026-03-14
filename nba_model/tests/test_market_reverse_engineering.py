import unittest
from unittest.mock import patch

import pandas as pd

from nba_model.evaluation.market_reverse_engineering import (
    aggregate_inferred_parameters,
    build_inferred_parameter_rows,
    build_reverse_engineering_base_table,
    run_market_reverse_engineering_continuous,
)


class MarketReverseEngineeringTests(unittest.TestCase):
    def test_build_inferred_parameter_rows_from_quotes_and_predictions(self):
        quotes_df = pd.DataFrame(
            [
                {
                    "quote_id": 1,
                    "source_table": "betting_lines",
                    "quote_ts_utc": "2026-03-01T10:00:00Z",
                    "player_id": 101,
                    "player_name": "Player One",
                    "game_date": "2026-03-05",
                    "book": "BookA",
                    "market_key": "",
                    "stat_type": "points",
                    "line_value": 24.7,
                    "over_odds": -115,
                    "under_odds": -105,
                },
                {
                    "quote_id": 2,
                    "source_table": "betting_lines",
                    "quote_ts_utc": "2026-03-01T10:00:00Z",
                    "player_id": 202,
                    "player_name": "Player Two",
                    "game_date": "2026-03-05",
                    "book": "BookA",
                    "market_key": "",
                    "stat_type": "points",
                    "line_value": 26.0,
                    "over_odds": 120,
                    "under_odds": -140,
                },
            ]
        )
        predictions_df = pd.DataFrame(
            [
                {
                    "prediction_id": 11,
                    "player_id": 101,
                    "player_name": "Player One",
                    "game_date": "2026-03-05",
                    "stat_type": "points",
                    "predicted_mean": 25.0,
                    "predicted_std": 4.0,
                    "distribution": "normal",
                },
                {
                    "prediction_id": 22,
                    "player_id": 202,
                    "player_name": "Player Two",
                    "game_date": "2026-03-05",
                    "stat_type": "points",
                    "predicted_mean": 25.0,
                    "predicted_std": 4.0,
                    "distribution": "normal",
                },
            ]
        )

        base_df = build_reverse_engineering_base_table(quotes_df, predictions_df)
        inferred_df = build_inferred_parameter_rows(base_df)

        self.assertEqual(len(base_df), 2)
        self.assertEqual(len(inferred_df), 2)
        self.assertIn("sigma_scale_consensus", inferred_df.columns)
        self.assertTrue((inferred_df["sigma_scale_consensus"] > 0).all())
        self.assertIn("line_minus_mu_sigma", inferred_df.columns)

    def test_aggregate_inferred_parameters_includes_player_segments(self):
        inferred_df = pd.DataFrame(
            [
                {
                    "source_table": "betting_line_snapshots",
                    "book": "BookA",
                    "stat_type": "points",
                    "distribution": "normal",
                    "player_id": 101,
                    "player_name": "Player One",
                    "game_date": "2026-03-01",
                    "market_key": "player_points",
                    "sigma_scale_consensus": 1.10,
                    "line_minus_mu_sigma": -0.12,
                    "vig_overround": 0.05,
                    "sigma_scale_over_under_gap": 0.03,
                    "has_two_sided_odds": 1,
                },
                {
                    "source_table": "betting_line_snapshots",
                    "book": "BookA",
                    "stat_type": "points",
                    "distribution": "normal",
                    "player_id": 101,
                    "player_name": "Player One",
                    "game_date": "2026-03-02",
                    "market_key": "player_points",
                    "sigma_scale_consensus": 1.18,
                    "line_minus_mu_sigma": -0.10,
                    "vig_overround": 0.06,
                    "sigma_scale_over_under_gap": 0.04,
                    "has_two_sided_odds": 1,
                },
                {
                    "source_table": "betting_line_snapshots",
                    "book": "BookA",
                    "stat_type": "points",
                    "distribution": "normal",
                    "player_id": 202,
                    "player_name": "Player Two",
                    "game_date": "2026-03-01",
                    "market_key": "player_points",
                    "sigma_scale_consensus": 0.92,
                    "line_minus_mu_sigma": 0.18,
                    "vig_overround": 0.05,
                    "sigma_scale_over_under_gap": 0.06,
                    "has_two_sided_odds": 1,
                },
                {
                    "source_table": "betting_line_snapshots",
                    "book": "BookA",
                    "stat_type": "points",
                    "distribution": "normal",
                    "player_id": 202,
                    "player_name": "Player Two",
                    "game_date": "2026-03-02",
                    "market_key": "player_points",
                    "sigma_scale_consensus": 0.98,
                    "line_minus_mu_sigma": 0.15,
                    "vig_overround": 0.06,
                    "sigma_scale_over_under_gap": 0.05,
                    "has_two_sided_odds": 1,
                },
            ]
        )

        summaries = aggregate_inferred_parameters(
            inferred_df=inferred_df,
            min_player_segment_rows=2,
            include_market_segments=True,
            min_market_segment_rows=2,
        )

        self.assertIn("book_stat", summaries)
        self.assertIn("book_stat_player", summaries)
        self.assertIn("book_stat_market", summaries)
        self.assertEqual(len(summaries["book_stat"]), 1)
        self.assertEqual(len(summaries["book_stat_player"]), 2)
        self.assertEqual(len(summaries["book_stat_market"]), 1)

    def test_continuous_runner_stops_when_thresholds_are_met(self):
        shared_summary = {
            "book_stat": pd.DataFrame(
                [
                    {
                        "source_table": "betting_line_snapshots",
                        "book": "BookA",
                        "stat_type": "points",
                        "distribution": "normal",
                        "rows": 30,
                        "sigma_scale_median": 1.05,
                    }
                ]
            ),
            "book_stat_player": pd.DataFrame(),
            "book_stat_market": pd.DataFrame(),
        }

        run_1_result = {
            "inferred_rows": 5,
            "book_stat_groups": 1,
            "book_stat_player_groups": 1,
        }
        run_2_result = {
            "inferred_rows": 40,
            "book_stat_groups": 2,
            "book_stat_player_groups": 6,
        }

        with patch(
            "nba_model.evaluation.market_reverse_engineering."
            "_run_market_reverse_engineering_once",
            side_effect=[
                (run_1_result, shared_summary),
                (run_2_result, shared_summary),
            ],
        ) as mock_once:
            result = run_market_reverse_engineering_continuous(
                poll_seconds=0.0,
                max_runs=5,
                min_inferred_rows=25,
                min_book_stat_groups=2,
                min_player_segment_groups=5,
                require_stability_runs=1,
            )

        self.assertEqual(mock_once.call_count, 2)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["runs_executed"], 2)
        self.assertTrue(result["coverage_ready"])
        self.assertTrue(result["stability_ready"])

    def test_continuous_runner_can_stop_on_max_runs_when_not_stable(self):
        summary_run_1 = {
            "book_stat": pd.DataFrame(
                [
                    {
                        "source_table": "betting_line_snapshots",
                        "book": "BookA",
                        "stat_type": "points",
                        "distribution": "normal",
                        "rows": 25,
                        "sigma_scale_median": 1.0,
                    }
                ]
            ),
            "book_stat_player": pd.DataFrame(),
            "book_stat_market": pd.DataFrame(),
        }
        summary_run_2 = {
            "book_stat": pd.DataFrame(
                [
                    {
                        "source_table": "betting_line_snapshots",
                        "book": "BookA",
                        "stat_type": "points",
                        "distribution": "normal",
                        "rows": 25,
                        "sigma_scale_median": 1.6,
                    }
                ]
            ),
            "book_stat_player": pd.DataFrame(),
            "book_stat_market": pd.DataFrame(),
        }
        summary_run_3 = {
            "book_stat": pd.DataFrame(
                [
                    {
                        "source_table": "betting_line_snapshots",
                        "book": "BookA",
                        "stat_type": "points",
                        "distribution": "normal",
                        "rows": 25,
                        "sigma_scale_median": 1.25,
                    }
                ]
            ),
            "book_stat_player": pd.DataFrame(),
            "book_stat_market": pd.DataFrame(),
        }

        ready_coverage_result = {
            "inferred_rows": 60,
            "book_stat_groups": 3,
            "book_stat_player_groups": 7,
        }

        with patch(
            "nba_model.evaluation.market_reverse_engineering."
            "_run_market_reverse_engineering_once",
            side_effect=[
                (ready_coverage_result, summary_run_1),
                (ready_coverage_result, summary_run_2),
                (ready_coverage_result, summary_run_3),
            ],
        ) as mock_once:
            result = run_market_reverse_engineering_continuous(
                poll_seconds=0.0,
                max_runs=3,
                min_inferred_rows=25,
                min_book_stat_groups=2,
                min_player_segment_groups=5,
                require_stability_runs=2,
                stability_tolerance=0.05,
                min_group_rows_for_stability=5,
            )

        self.assertEqual(mock_once.call_count, 3)
        self.assertEqual(result["status"], "max_runs_reached")
        self.assertEqual(result["runs_executed"], 3)
        self.assertTrue(result["coverage_ready"])
        self.assertFalse(result["stability_ready"])


if __name__ == "__main__":
    unittest.main()
