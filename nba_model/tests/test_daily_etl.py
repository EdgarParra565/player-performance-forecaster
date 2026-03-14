import unittest
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd

from nba_model.data.daily_etl import (
    _player_names_with_existing_game_logs,
    resolve_players,
    run_daily_etl,
    run_with_retry,
)
from nba_model.data.database.db_manager import DatabaseManager


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


class PlayerResolutionTests(unittest.TestCase):
    @patch("nba_model.data.daily_etl.nba_players.find_players_by_full_name")
    def test_player_names_with_existing_game_logs_tracks_resolved_without_logs(
        self,
        mock_find_players_by_full_name,
    ):
        def _fake_find(name):
            if name == "Has Logs":
                return [{"id": 101}]
            if name == "No Logs":
                return [{"id": 202}]
            return []

        mock_find_players_by_full_name.side_effect = _fake_find

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            with DatabaseManager(db_path=db_path) as db:
                db.conn.execute(
                    """
                    INSERT INTO game_logs (player_id, game_id, game_date, season)
                    VALUES (?, ?, ?, ?)
                    """,
                    (101, "G1", "2025-01-01", "2024-25"),
                )
                db.conn.commit()

            names_with_logs, unresolved_names, names_without_logs = (
                _player_names_with_existing_game_logs(
                    ["Has Logs", "No Logs", "Unknown"],
                    db_path=db_path,
                )
            )

        self.assertEqual(names_with_logs, {"Has Logs"})
        self.assertEqual(unresolved_names, {"Unknown"})
        self.assertEqual(names_without_logs, {"No Logs"})

    @patch("nba_model.data.daily_etl._list_active_player_names")
    @patch("nba_model.data.daily_etl._list_players_from_db")
    def test_resolve_players_all_db_players_uses_full_db_pool(
        self,
        mock_list_players_from_db,
        mock_list_active_player_names,
    ):
        mock_list_players_from_db.return_value = [
            "Player 1",
            "Player 2",
            "Player 3",
            "Player 4",
            "Player 5",
            "Player 6",
        ]
        mock_list_active_player_names.return_value = []

        selected = resolve_players(
            explicit_players=None,
            include_db_players=False,
            all_db_players=True,
            db_path="data/database/test.db",
        )

        self.assertEqual(
            selected,
            ["Player 1", "Player 2", "Player 3", "Player 4", "Player 5", "Player 6"],
        )
        mock_list_active_player_names.assert_not_called()

    @patch("nba_model.data.daily_etl._list_active_player_names")
    @patch("nba_model.data.daily_etl._list_players_from_db")
    def test_resolve_players_min_players_expands_beyond_default_seed(
        self,
        mock_list_players_from_db,
        mock_list_active_player_names,
    ):
        mock_list_players_from_db.return_value = []
        mock_list_active_player_names.return_value = [
            "LeBron James",
            "Anthony Edwards",
            "Giannis Antetokounmpo",
            "Kevin Durant",
        ]

        selected = resolve_players(
            explicit_players=None,
            include_db_players=False,
            all_db_players=False,
            min_players=7,
            db_path="data/database/test.db",
        )

        self.assertEqual(len(selected), 7)
        self.assertEqual(selected[:5], [
            "LeBron James",
            "Stephen Curry",
            "Nikola Jokic",
            "Luka Doncic",
            "Jayson Tatum",
        ])
        self.assertEqual(selected[5:], ["Anthony Edwards", "Giannis Antetokounmpo"])

    @patch("nba_model.data.daily_etl._player_names_with_existing_game_logs")
    @patch("nba_model.data.daily_etl._list_players_from_db")
    def test_resolve_players_skip_zero_game_players_keeps_explicit(
        self,
        mock_list_players_from_db,
        mock_player_names_with_existing_game_logs,
    ):
        mock_list_players_from_db.return_value = ["DB Player 1", "DB Player 2"]
        mock_player_names_with_existing_game_logs.return_value = (
            {"DB Player 1"},
            set(),
            {"DB Player 2"},
        )

        selected = resolve_players(
            explicit_players=["Manual Player"],
            include_db_players=True,
            skip_zero_game_players=True,
            db_path="data/database/test.db",
        )

        self.assertEqual(selected, ["Manual Player", "DB Player 1"])

    @patch("nba_model.data.daily_etl._list_players_from_db")
    def test_resolve_players_skip_zero_game_players_keeps_db_names_when_logs_table_empty(
        self,
        mock_list_players_from_db,
    ):
        mock_list_players_from_db.return_value = ["LeBron James", "Stephen Curry"]
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            selected = resolve_players(
                explicit_players=None,
                include_db_players=True,
                db_path=db_path,
                all_db_players=True,
                skip_zero_game_players=True,
            )

        self.assertEqual(selected, ["LeBron James", "Stephen Curry"])


class DailyETLTests(unittest.TestCase):
    @patch("nba_model.data.daily_etl.run_market_reverse_engineering_continuous")
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
        mock_reverse_engineering,
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
        mock_reverse_engineering.return_value = {
            "status": "ready",
            "inferred_rows": 42,
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
        self.assertEqual(report["steps"]["reverse_engineering"]["status"], "success")
        self.assertIn("player_selection_summary", report)
        self.assertEqual(report["player_selection_summary"]["explicit_count"], 1)
        self.assertEqual(report["player_selection_summary"]["db_count"], 0)

        loader.load_player_data.assert_called_once_with(
            player_name="LeBron James",
            n_games=120,
            force_refresh=True,
        )

    @patch("nba_model.data.daily_etl.run_market_reverse_engineering_continuous")
    @patch("nba_model.data.daily_etl._default_api_key")
    def test_run_daily_etl_skips_odds_without_api_key(
        self,
        mock_default_key,
        mock_reverse_engineering,
    ):
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
        self.assertEqual(report["steps"]["reverse_engineering"]["status"], "skipped")
        mock_reverse_engineering.assert_not_called()

    @patch("nba_model.data.daily_etl.run_market_reverse_engineering_continuous")
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
        mock_reverse_engineering,
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
        mock_reverse_engineering.return_value = {
            "status": "ready",
            "inferred_rows": 18,
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
        self.assertEqual(report["steps"]["reverse_engineering"]["status"], "success")

    @patch("nba_model.data.daily_etl.run_market_reverse_engineering_continuous")
    @patch("nba_model.data.daily_etl.fetch_and_store_betting_lines")
    @patch("nba_model.data.daily_etl.build_team_defense_validation_report")
    @patch("nba_model.data.daily_etl.populate_team_defense")
    @patch("nba_model.data.daily_etl.DataLoader")
    def test_run_daily_etl_reverse_engineering_partial_when_not_ready(
        self,
        mock_loader_cls,
        mock_populate_team_defense,
        mock_validation_report,
        mock_fetch_odds,
        mock_reverse_engineering,
    ):
        loader = MagicMock()
        loader.load_player_data.return_value = pd.DataFrame({"points": [22], "minutes": [32]})
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
            "records_parsed": 4,
            "distinct_players": 2,
        }
        mock_reverse_engineering.return_value = {
            "status": "max_runs_reached",
            "runs_executed": 3,
            "inferred_rows": 9,
        }

        report = run_daily_etl(
            players=["LeBron James"],
            include_db_players=False,
            odds_api_key="dummy-key",
            retries=0,
            write_report=False,
        )

        self.assertEqual(report["status"], "partial_success")
        self.assertEqual(report["steps"]["reverse_engineering"]["status"], "partial_success")

    @patch("nba_model.data.daily_etl._list_active_player_names")
    @patch("nba_model.data.daily_etl._list_players_from_db")
    @patch("nba_model.data.daily_etl._default_api_key")
    def test_run_daily_etl_reports_player_source_breakdown(
        self,
        mock_default_key,
        mock_list_players_from_db,
        mock_list_active_player_names,
    ):
        mock_default_key.return_value = None
        mock_list_players_from_db.return_value = ["DB Player 1"]
        mock_list_active_player_names.return_value = [
            "DB Player 1",
            "Active Player 2",
            "Active Player 3",
        ]

        report = run_daily_etl(
            players=None,
            include_db_players=True,
            min_players=3,
            skip_game_logs=True,
            skip_team_defense=True,
            skip_odds=False,
            retries=0,
            write_report=False,
        )

        summary = report["player_selection_summary"]
        self.assertEqual(summary["explicit_count"], 0)
        self.assertEqual(summary["db_count"], 1)
        self.assertEqual(summary["default_seed_count"], 0)
        self.assertEqual(summary["min_topup_count"], 2)
        self.assertEqual(summary["total_selected"], 3)

    @patch("nba_model.data.daily_etl._player_names_with_existing_game_logs")
    @patch("nba_model.data.daily_etl._list_players_from_db")
    @patch("nba_model.data.daily_etl._default_api_key")
    def test_run_daily_etl_reports_zero_game_skip_summary(
        self,
        mock_default_key,
        mock_list_players_from_db,
        mock_player_names_with_existing_game_logs,
    ):
        mock_default_key.return_value = None
        mock_list_players_from_db.return_value = ["DB Player 1", "DB Player 2"]
        mock_player_names_with_existing_game_logs.return_value = (
            {"DB Player 1"},
            set(),
            {"DB Player 2"},
        )

        report = run_daily_etl(
            players=None,
            include_db_players=True,
            skip_zero_game_players=True,
            skip_game_logs=True,
            skip_team_defense=True,
            skip_odds=False,
            retries=0,
            write_report=False,
        )

        summary = report["player_selection_summary"]
        self.assertTrue(summary["skip_zero_game_players"])
        self.assertEqual(summary["zero_game_skipped_count"], 1)
        self.assertEqual(summary["zero_game_resolved_count"], 1)
        self.assertEqual(summary["db_count"], 1)
        self.assertEqual(summary["total_selected"], 1)
        self.assertIn("DB Player 2", summary["zero_game_skipped_examples"])


if __name__ == "__main__":
    unittest.main()
