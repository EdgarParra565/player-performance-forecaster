import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from nba_model.data.daily_etl import run_daily_etl, run_with_retry


class RetryTests(unittest.TestCase):
    def test_run_with_retry_recovers_after_transient_failure(self):
        attempts = {"count": 0}

        def flaky():
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("transient")
            return {"ok": True}

        result = run_with_retry(
            step_name="flaky",
            func=flaky,
            retries=2,
            retry_delay_seconds=0,
            retry_backoff=1.0,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["attempts"], 2)
        self.assertTrue(result["result"]["ok"])


class DailyETLTests(unittest.TestCase):
    @patch("nba_model.data.daily_etl.fetch_and_store_betting_lines")
    @patch("nba_model.data.daily_etl.build_team_defense_validation_report")
    @patch("nba_model.data.daily_etl.populate_team_defense")
    @patch("nba_model.data.daily_etl.DataLoader")
    def test_run_daily_etl_successful_flow(
        self,
        mock_loader_cls,
        mock_populate_team_defense,
        mock_validation_report,
        mock_fetch_odds,
    ):
        loader = MagicMock()
        loader.load_player_data.return_value = pd.DataFrame({"points": [20, 25], "minutes": [34, 36]})
        loader.db = MagicMock()
        mock_loader_cls.return_value = loader

        mock_populate_team_defense.return_value = 30
        mock_validation_report.return_value = {
            "row_count": 30,
            "missing_teams": [],
            "unexpected_teams": [],
            "is_complete": True,
        }
        mock_fetch_odds.return_value = {
            "records_parsed": 12,
            "distinct_players": 8,
        }

        report = run_daily_etl(
            players=["LeBron James"],
            include_db_players=False,
            odds_api_key="dummy-key",
            retries=0,
            write_report=False,
        )

        self.assertEqual(report["status"], "success")
        self.assertEqual(report["steps"]["game_logs"]["status"], "success")
        self.assertEqual(report["steps"]["team_defense"]["status"], "success")
        self.assertEqual(report["steps"]["odds"]["status"], "success")

        loader.load_player_data.assert_called_once_with(
            player_name="LeBron James",
            n_games=120,
            force_refresh=True,
        )

    @patch("nba_model.data.daily_etl._default_api_key")
    def test_run_daily_etl_skips_odds_without_api_key(self, mock_default_key):
        mock_default_key.return_value = None
        report = run_daily_etl(
            players=["LeBron James"],
            include_db_players=False,
            skip_game_logs=True,
            skip_team_defense=True,
            skip_odds=False,
            odds_api_key=None,
            retries=0,
            write_report=False,
        )

        self.assertEqual(report["status"], "success")
        self.assertEqual(report["steps"]["odds"]["status"], "skipped")

    @patch("nba_model.data.daily_etl.fetch_and_store_betting_lines")
    @patch("nba_model.data.daily_etl.build_team_defense_validation_report")
    @patch("nba_model.data.daily_etl.populate_team_defense")
    @patch("nba_model.data.daily_etl.DataLoader")
    def test_run_daily_etl_partial_success_when_player_refresh_fails(
        self,
        mock_loader_cls,
        mock_populate_team_defense,
        mock_validation_report,
        mock_fetch_odds,
    ):
        loader = MagicMock()
        loader.db = MagicMock()
        loader.load_player_data.side_effect = [
            RuntimeError("api timeout"),
            pd.DataFrame({"points": [18], "minutes": [30]}),
        ]
        mock_loader_cls.return_value = loader

        mock_populate_team_defense.return_value = 30
        mock_validation_report.return_value = {
            "row_count": 30,
            "missing_teams": [],
            "unexpected_teams": [],
            "is_complete": True,
        }
        mock_fetch_odds.return_value = {
            "records_parsed": 3,
            "distinct_players": 2,
        }

        report = run_daily_etl(
            players=["Player A", "Player B"],
            include_db_players=False,
            odds_api_key="dummy-key",
            retries=0,
            write_report=False,
        )

        self.assertEqual(report["status"], "partial_success")
        self.assertEqual(report["steps"]["game_logs"]["status"], "partial_success")
        self.assertEqual(report["steps"]["team_defense"]["status"], "success")
        self.assertEqual(report["steps"]["odds"]["status"], "success")


if __name__ == "__main__":
    unittest.main()
