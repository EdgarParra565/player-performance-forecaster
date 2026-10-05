"""Unit tests for daily ETL orchestration and retry/reporting behavior."""

import unittest
from datetime import datetime, timedelta, timezone
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


# Several run_daily_etl tests pass web_text_urls without a db_path, which would
# run the VegasInsider ingest steps against the default (live) DB. Stub them
# module-wide; their behaviour is covered in test_vegasinsider_odds /
# test_vegasinsider_mlb_ingestion against temp DBs.
_VI_STEP_PATCHES = [
    patch("nba_model.data.daily_etl._run_vegasinsider_step",
          return_value={"status": "success", "inserted": 0}),
    patch("nba_model.data.daily_etl._run_vegasinsider_mlb_props_step",
          return_value={"status": "success", "inserted": 0}),
]


_REAL_RUN_DAILY_ETL = run_daily_etl
_ISOLATION_TMP = None


def _isolated_run_daily_etl(*args, **kwargs):
    # Most tests below omit db_path, and run_daily_etl's default is the LIVE
    # data/database/nba_data.db: _record_odds_poll_run then wrote a fake
    # 'success' odds_poll_runs row per run (the whole live table was test
    # pollution), which feeds the 24h odds-freshness guard. Default every call
    # to a throwaway DB unless the test passes its own.
    kwargs.setdefault("db_path", str(Path(_ISOLATION_TMP.name) / "nba_data.db"))
    return _REAL_RUN_DAILY_ETL(*args, **kwargs)


def setUpModule():
    global _ISOLATION_TMP, run_daily_etl
    _ISOLATION_TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    run_daily_etl = _isolated_run_daily_etl
    for p in _VI_STEP_PATCHES:
        p.start()


def tearDownModule():
    global run_daily_etl
    for p in _VI_STEP_PATCHES:
        p.stop()
    run_daily_etl = _REAL_RUN_DAILY_ETL
    _ISOLATION_TMP.cleanup()


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

        # Report/alert contract must survive the structured-logging migration:
        # the embedded `alert` marker is still present and well-shaped.
        self.assertIn("alert", report)
        self.assertFalse(report["alert"]["alert"])  # clean run → no alert
        self.assertEqual(report["alert"]["severity"], "ok")

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
    def test_run_daily_etl_reuses_recent_odds_poll_within_guard_window(
        self,
        mock_fetch_odds,
        mock_reverse_engineering,
    ):
        mock_reverse_engineering.return_value = {
            "status": "ready",
            "inferred_rows": 8,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            with DatabaseManager(db_path=db_path) as db:
                db.conn.execute(
                    """
                    INSERT INTO odds_poll_runs (polled_at_utc, status)
                    VALUES (?, ?)
                    """,
                    (datetime.now(timezone.utc).isoformat(), "success"),
                )
                db.conn.commit()

            report = run_daily_etl(
                players=["LeBron James"],
                include_db_players=False,
                skip_game_logs=True,
                skip_team_defense=True,
                skip_odds=False,
                odds_api_key="dummy-key",
                odds_min_hours_between_polls=24.0,
                retries=0,
                db_path=db_path,
                write_report=False,
            )

        self.assertEqual(report["steps"]["odds"]["status"], "success")
        self.assertEqual(
            report["steps"]["odds"]["result"]["status"],
            "reused_recent_poll",
        )
        self.assertFalse(report["steps"]["odds"]["result"]["poll_executed"])
        self.assertEqual(report["steps"]["reverse_engineering"]["status"], "success")
        mock_fetch_odds.assert_not_called()
        mock_reverse_engineering.assert_called_once()

    @patch("nba_model.data.daily_etl.run_market_reverse_engineering_continuous")
    @patch("nba_model.data.daily_etl.fetch_and_store_betting_lines")
    def test_run_daily_etl_force_odds_poll_ignores_guard_window(
        self,
        mock_fetch_odds,
        mock_reverse_engineering,
    ):
        mock_fetch_odds.return_value = {
            "records_parsed": 2,
            "records_valid": 2,
            "db_inserted": 2,
            "snapshots_inserted": 2,
        }
        mock_reverse_engineering.return_value = {
            "status": "ready",
            "inferred_rows": 4,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nba_data.db")
            with DatabaseManager(db_path=db_path) as db:
                db.conn.execute(
                    """
                    INSERT INTO odds_poll_runs (polled_at_utc, status)
                    VALUES (?, ?)
                    """,
                    (
                        (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
                        "success",
                    ),
                )
                db.conn.commit()

            report = run_daily_etl(
                players=["LeBron James"],
                include_db_players=False,
                skip_game_logs=True,
                skip_team_defense=True,
                skip_odds=False,
                odds_api_key="dummy-key",
                odds_min_hours_between_polls=24.0,
                force_odds_poll=True,
                retries=0,
                db_path=db_path,
                write_report=False,
            )

            with DatabaseManager(db_path=db_path) as db:
                rows = db.conn.execute(
                    "SELECT status FROM odds_poll_runs ORDER BY poll_id ASC"
                ).fetchall()

        self.assertEqual(report["steps"]["odds"]["status"], "success")
        mock_fetch_odds.assert_called_once()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1][0], "success")

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

    @patch("nba_model.data.daily_etl.run_market_reverse_engineering_continuous")
    @patch("nba_model.data.daily_etl.parse_and_store_web_prop_cards")
    @patch("nba_model.data.daily_etl.fetch_and_store_web_text")
    @patch("nba_model.data.daily_etl.fetch_and_store_betting_lines")
    @patch("nba_model.data.daily_etl._default_api_key")
    def test_run_daily_etl_runs_browser_parser_when_web_urls_present(
        self,
        mock_default_key,
        mock_fetch_odds,
        mock_fetch_web_text,
        mock_parse_web_props,
        mock_reverse_engineering,
    ):
        mock_default_key.return_value = "dummy-key"
        mock_fetch_odds.return_value = {"records_parsed": 1}
        mock_fetch_web_text.return_value = {
            "status": "success",
            "fetched_count": 1,
            "db_inserted": 1,
        }
        mock_parse_web_props.return_value = {
            "status": "success",
            "cards_extracted": 2,
            "cards_retained": 2,
            "db_inserted": 2,
        }
        mock_reverse_engineering.return_value = {
            "status": "ready",
            "inferred_rows": 3,
        }

        report = run_daily_etl(
            players=["LeBron James"],
            include_db_players=False,
            skip_game_logs=True,
            skip_team_defense=True,
            web_text_urls=["https://example.com/props"],
            retries=0,
            write_report=False,
        )

        self.assertEqual(report["steps"]["web_text"]["status"], "success")
        self.assertEqual(report["steps"]["browser_parser"]["status"], "success")
        self.assertEqual(report["steps"]["reverse_engineering"]["status"], "success")
        mock_parse_web_props.assert_called_once()

    @patch("nba_model.data.daily_etl.parse_and_store_web_prop_cards")
    @patch("nba_model.data.daily_etl.fetch_and_store_web_text")
    def test_run_daily_etl_skips_browser_parser_when_flag_set(
        self,
        mock_fetch_web_text,
        mock_parse_web_props,
    ):
        mock_fetch_web_text.return_value = {
            "status": "success",
            "fetched_count": 1,
            "db_inserted": 1,
        }
        report = run_daily_etl(
            players=["LeBron James"],
            include_db_players=False,
            skip_game_logs=True,
            skip_team_defense=True,
            skip_odds=True,
            skip_browser_parser=True,
            skip_reverse_engineering=True,
            web_text_urls=["https://example.com/props"],
            retries=0,
            write_report=False,
        )

        self.assertEqual(report["steps"]["web_text"]["status"], "success")
        self.assertEqual(report["steps"]["browser_parser"]["status"], "skipped")
        mock_parse_web_props.assert_not_called()

    @patch("nba_model.data.daily_etl.parse_and_store_web_prop_cards")
    @patch("nba_model.data.daily_etl.fetch_and_store_web_text")
    def test_run_daily_etl_browser_parser_partial_status_rolls_up(
        self,
        mock_fetch_web_text,
        mock_parse_web_props,
    ):
        mock_fetch_web_text.return_value = {
            "status": "success",
            "fetched_count": 1,
            "db_inserted": 1,
        }
        mock_parse_web_props.return_value = {
            "status": "partial_success",
            "cards_extracted": 0,
            "cards_retained": 0,
            "db_inserted": 0,
        }
        report = run_daily_etl(
            players=["LeBron James"],
            include_db_players=False,
            skip_game_logs=True,
            skip_team_defense=True,
            skip_odds=True,
            skip_reverse_engineering=True,
            web_text_urls=["https://example.com/props"],
            retries=0,
            write_report=False,
        )

        self.assertEqual(report["steps"]["browser_parser"]["status"], "partial_success")
        self.assertEqual(report["status"], "partial_success")

    @patch("nba_model.model.web_text_ingestion.playwright_is_available",
           return_value=True)
    @patch("nba_model.data.daily_etl.fetch_and_store_web_text")
    def test_run_daily_etl_passes_browser_session_flags_to_web_text_step(
        self,
        mock_fetch_web_text,
        _mock_pw_available,
    ):
        # The Playwright preflight guard in daily_etl trips whenever a
        # browser fetch is requested (auth-state / user-data-dir / CDP port)
        # and the interpreter lacks the `playwright` package. That's correct
        # production behavior but it makes this test environment-dependent
        # (passes in venv, fails in Docker which intentionally skips
        # Playwright). Mock the preflight as available so we exercise the
        # kwarg-forwarding contract regardless of the host's Playwright
        # state. The negative branch is covered by PlaywrightGuardTests.
        mock_fetch_web_text.return_value = {
            "status": "success",
            "fetched_count": 1,
            "db_inserted": 1,
        }

        run_daily_etl(
            players=["LeBron James"],
            include_db_players=False,
            skip_game_logs=True,
            skip_team_defense=True,
            skip_odds=True,
            skip_browser_parser=True,
            skip_reverse_engineering=True,
            web_text_urls=["https://example.com/props"],
            browser_auth_state_file="data/config/auth/underdog_state.json",
            browser_user_data_dir="data/config/auth/underdog_profile",
            retries=0,
            write_report=False,
        )

        _, kwargs = mock_fetch_web_text.call_args
        self.assertEqual(
            kwargs.get("browser_auth_state_file"),
            "data/config/auth/underdog_state.json",
        )
        self.assertEqual(
            kwargs.get("browser_user_data_dir"),
            "data/config/auth/underdog_profile",
        )


class PlaywrightGuardTests(unittest.TestCase):
    """Review finding: the Playwright guard raised AFTER bulk/game_logs/
    team_defense/odds had run, so no report was written and no alert fired.
    Now the preflight runs FIRST, web_text records a failed step (no URL loop
    through the broken import), and the run still writes its report."""

    @patch("nba_model.data.daily_etl.fetch_and_store_web_text")
    @patch("nba_model.data.daily_etl.fetch_and_store_betting_lines")
    @patch("nba_model.model.web_text_ingestion.playwright_is_available",
           return_value=False)
    def test_missing_playwright_fails_step_and_still_writes_report(
        self, mock_pw, mock_odds, mock_fetch,
    ):
        order = []
        mock_pw.side_effect = lambda: (order.append("preflight"), False)[1]
        mock_odds.side_effect = lambda **_kw: (order.append("odds"), {"events_processed": 0})[1]
        with tempfile.TemporaryDirectory() as tmp:
            report = run_daily_etl(
                players=["LeBron James"],
                include_db_players=False,
                skip_bulk_results_ingest=True,
                skip_game_logs=True,
                skip_team_defense=True,
                skip_reverse_engineering=True,
                odds_api_key="test-key",
                web_text_urls=["https://example.com/props"],
                chrome_debug_port=9222,
                retries=0,
                write_report=True,
                report_dir=tmp,
            )
            self.assertTrue(Path(report["report_path"]).exists())
        self.assertEqual(order, ["preflight", "odds"])  # guard before other steps
        web = report["steps"]["web_text"]
        self.assertEqual(web["status"], "failed")
        self.assertEqual(web["reason"], "browser_preflight")
        self.assertIn("Playwright", web["error"])
        self.assertIn(".venv/bin/python3", web["error"])
        mock_fetch.assert_not_called()  # no per-URL retry loop
        self.assertEqual(report["steps"]["browser_parser"]["status"], "skipped")
        self.assertEqual(report["steps"]["odds"]["status"], "success")  # other steps ran
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["alert"]["severity"], "error")

    @patch("nba_model.data.daily_etl.fetch_and_store_web_text",
           return_value={"status": "success", "fetched_count": 1})
    @patch("nba_model.model.web_text_ingestion.playwright_is_available",
           return_value=False)
    def test_requests_fetch_does_not_need_playwright(self, _mock_pw, mock_fetch):
        report = run_daily_etl(
            players=["LeBron James"], include_db_players=False,
            skip_bulk_results_ingest=True, skip_game_logs=True,
            skip_team_defense=True, skip_odds=True, skip_browser_parser=True,
            skip_reverse_engineering=True,
            web_text_urls=["https://example.com/props"],
            retries=0, write_report=False,
        )
        mock_fetch.assert_called_once()
        self.assertEqual(report["steps"]["web_text"]["status"], "success")


class OddsAuthErrorStepTests(unittest.TestCase):
    """Review finding: HTTP 401/403/429 on event discovery was swallowed →
    odds step "success, events_processed 0"."""

    def test_auth_error_fails_odds_step_without_retrying(self):
        from nba_model.model.odds_ingestion import OddsApiAuthError

        calls = []

        def boom(**_kw):
            calls.append(1)
            raise OddsApiAuthError("Odds API HTTP 401 (auth / plan / quota): bad key", 401)

        with patch("nba_model.data.daily_etl.fetch_and_store_betting_lines", side_effect=boom):
            report = run_daily_etl(
                players=["LeBron James"], include_db_players=False,
                skip_bulk_results_ingest=True, skip_game_logs=True,
                skip_team_defense=True, skip_reverse_engineering=True,
                skip_browser_parser=True, odds_api_key="bad", retries=2,
                retry_delay_seconds=0, write_report=False,
            )
        self.assertEqual(report["steps"]["odds"]["status"], "failed")
        self.assertEqual(len(calls), 1)  # retryable=False → no re-runs
        self.assertIn("401", str(report["steps"]["odds"]["error"]))
        self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
