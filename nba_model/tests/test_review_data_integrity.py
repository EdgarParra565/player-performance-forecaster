"""Regression tests for data/ETL fixes from the 2026-09-25 code review.

One class per fixed finding in review_findings_data.md. All temp-DB / mocked;
no network, never the live DB (see tests/conftest.py).
"""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from nba_model.data import hourly_update
from nba_model.data.data_loader import DataLoader, current_nba_season
from nba_model.data.database.db_manager import DatabaseManager
from nba_model.data.etl_alerts import build_alert
from nba_model.evaluation.monthly_diagnostics import load_prediction_actuals
from nba_model.model import edge_scanner as es
from nba_model.model import odds_ingestion as oi
from nba_model.tests.test_bet_log import GAME_DATE, LEBRON_ID, _game_log, _pick


class _TmpDb(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = str(Path(self._tmp.name) / "nba.db")

    def tearDown(self):
        self._tmp.cleanup()


def _mlb_rec(game_date, sha, observed):
    return {
        "snapshot_id": 1, "source_url": "https://vi.test/mlb", "source": "vegasinsider",
        "book": "bet365", "observed_at_utc": observed, "game_date": game_date,
        "player_name": "Aaron Judge", "stat_type": "anytime_home_run",
        "stat_group": "combined", "market_shape": "yes_no", "line_value": 0.5,
        "side": "over", "over_odds": 250, "under_odds": None,
        "parser_version": "t", "record_sha256": sha,
    }


class MlbPropLinesGameDateKeyTests(_TmpDb):
    def test_same_price_on_next_slate_still_lands(self):
        with DatabaseManager(db_path=self.db_path) as db:
            first = db.insert_mlb_prop_lines(
                [_mlb_rec("2026-09-24", "a", "2026-09-24T18:00:00+00:00")])
            second = db.insert_mlb_prop_lines(
                [_mlb_rec("2026-09-25", "b", "2026-09-25T18:00:00+00:00")])
            again = db.insert_mlb_prop_lines(
                [_mlb_rec("2026-09-25", "c", "2026-09-25T19:00:00+00:00")])
            dates = [r[0] for r in db.conn.execute(
                "SELECT game_date FROM mlb_prop_lines ORDER BY game_date")]
        self.assertEqual((first["inserted"], second["inserted"]), (1, 1))
        self.assertEqual(again["skipped_unchanged"], 1)  # same slate, unchanged
        self.assertEqual(dates, ["2026-09-24", "2026-09-25"])


class SinceHoursBoundaryTests(_TmpDb):
    def _card(self, observed, idx, line):
        return {
            "snapshot_id": 1, "source_url": "https://ud.test/nba", "book": "Underdog",
            "observed_at_utc": observed, "player_name": f"Player {idx}",
            "player_classification": "active_nba", "stat_type": "points",
            "line_value": line, "side": "over", "parse_confidence": 0.99,
            "parser_version": "t", "record_sha256": f"sha-{idx}",
        }

    def test_row_older_than_window_on_cutoff_date_is_excluded(self):
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=48)
        stale = (cutoff - timedelta(minutes=1)).isoformat()  # 'T...+00:00'
        fresh = (now - timedelta(hours=1)).isoformat()
        with DatabaseManager(db_path=self.db_path) as db:
            db.insert_web_prop_cards([self._card(stale, 1, 20.5),
                                      self._card(fresh, 2, 21.5)])
        lines = es.fetch_latest_prop_lines(self.db_path, since_hours=48)
        self.assertEqual(sorted(lines["player_name"]), ["Player 2"])


class ClosingClvSameLineTests(_TmpDb):
    def _seed(self, snapshots):
        with DatabaseManager(db_path=self.db_path) as db:
            db.conn.execute("INSERT INTO players (player_id, name, team) VALUES (?, ?, ?)",
                            (LEBRON_ID, "LeBron James", "LAL"))
            db.conn.commit()
            db.insert_game_logs(pd.DataFrame([_game_log(LEBRON_ID, GAME_DATE, 30, 8, 7)]))
            db.conn.executemany(
                """INSERT INTO betting_line_snapshots
                   (snapshot_ts_utc, event_id, game_date, player_id, book,
                    market_key, stat_type, line_value, over_odds, under_odds)
                   VALUES (?, 'e', ?, ?, ?, 'm', 'points', ?, ?, ?)""",
                [(ts, GAME_DATE, LEBRON_ID, book, line, o, u)
                 for ts, book, line, o, u in snapshots],
            )
            db.conn.commit()

    def _settle(self):
        with DatabaseManager(db_path=self.db_path) as db:
            db.insert_bet_log_rows([_pick("points", 25.5, "over", book="FanDuel")])
            db.settle_bet_log(fill_clv=True)
            return db.conn.execute("SELECT clv_delta FROM bet_log").fetchone()[0]

    def test_close_at_a_different_line_is_ignored(self):
        self._seed([("2025-04-10 17:00:00", "DraftKings", 22.5, 120, -140)])
        self.assertIsNone(self._settle())

    def test_same_line_prefers_the_picks_own_book(self):
        self._seed([
            ("2025-04-10 16:00:00", "FanDuel", 25.5, -140, 120),
            ("2025-04-10 17:00:00", "DraftKings", 25.5, 150, -170),
        ])
        self.assertAlmostEqual(self._settle(), 140 / 240 - 0.5238, places=4)


class MonthlyDiagnosticsDateJoinTests(_TmpDb):
    def test_iso_game_log_dates_join_date_only_predictions(self):
        with DatabaseManager(db_path=self.db_path) as db:
            db.insert_game_logs(pd.DataFrame([_game_log(LEBRON_ID, "2025-04-10T00:00:00", 30, 8, 7)]))
            db.conn.execute(
                """INSERT INTO predictions (player_id, game_date, stat_type,
                       predicted_mean, predicted_std, prob_over, line_value)
                   VALUES (?, '2025-04-10', 'points', 27.0, 5.0, 0.6, 25.5)""",
                (LEBRON_ID,))
            db.conn.commit()
        df = load_prediction_actuals(db_path=self.db_path)
        self.assertEqual(float(df.iloc[0]["actual_value"]), 30.0)


class HourlyStepResultStatusTests(unittest.TestCase):
    def _report(self):
        return {"steps": {}, "failed_steps": []}

    def test_returned_failed_status_is_a_failed_step(self):
        report = self._report()
        ok = hourly_update._record_step(report, "web_text",
                                        lambda: {"status": "failed", "failed_count": 16})
        self.assertFalse(ok)
        self.assertEqual(report["failed_steps"], ["web_text"])
        self.assertFalse(report["steps"]["web_text"]["ok"])
        self.assertTrue(build_alert(report)["alert"])

    def test_partial_status_warns_but_does_not_fail(self):
        report = self._report()
        self.assertTrue(hourly_update._record_step(
            report, "web_text", lambda: {"status": "partial_success"}))
        self.assertEqual(report["failed_steps"], [])
        self.assertEqual(build_alert(report)["severity"], "warning")

    def test_nested_failed_status_alerts(self):
        report = {"steps": {"x": {"ok": True, "result": {"status": "failed"}}}}
        self.assertEqual(build_alert(report)["severity"], "error")


class OddsFreshnessGuardTests(_TmpDb):
    @patch("nba_model.data.daily_etl.run_market_reverse_engineering_continuous")
    @patch("nba_model.data.daily_etl.fetch_and_store_betting_lines")
    def test_recent_failed_poll_does_not_suppress_retry(self, mock_fetch, mock_re):
        from nba_model.data.daily_etl import run_daily_etl
        mock_fetch.return_value = {"records_parsed": 1, "records_valid": 1,
                                   "db_inserted": 1, "snapshots_inserted": 1}
        mock_re.return_value = {"status": "ready", "inferred_rows": 0}
        with DatabaseManager(db_path=self.db_path) as db:
            db.conn.execute("INSERT INTO odds_poll_runs (polled_at_utc, status) VALUES (?, ?)",
                            (datetime.now(timezone.utc).isoformat(), "failed"))
            db.conn.commit()
        run_daily_etl(players=["LeBron James"], include_db_players=False,
                      skip_game_logs=True, skip_team_defense=True,
                      skip_bulk_results_ingest=True, skip_odds=False,
                      odds_api_key="dummy", odds_min_hours_between_polls=24.0,
                      retries=0, db_path=self.db_path, write_report=False)
        mock_fetch.assert_called_once()


class BulkResultsStatusTests(_TmpDb):
    def _run(self, ingest_result):
        from nba_model.data.daily_etl import run_daily_etl
        with patch("nba_model.data.nba_results_ingestion.ingest_all",
                   return_value=ingest_result):
            return run_daily_etl(players=["LeBron James"], include_db_players=False,
                                 skip_game_logs=True, skip_team_defense=True,
                                 skip_odds=True, retries=0, db_path=self.db_path,
                                 write_report=False)

    def test_every_season_failed_is_a_failed_step(self):
        err = {"seasons": [{"season": "2025-26", "error": "ConnectionError: x"}]}
        report = self._run({"games": err, "player_logs": err})
        self.assertEqual(report["steps"]["bulk_results_ingest"]["status"], "failed")

    def test_some_seasons_failed_is_partial(self):
        ok = {"seasons": [{"season": "2025-26", "inserted": 1}]}
        err = {"seasons": [{"season": "2024-25", "error": "ConnectionError: x"}]}
        report = self._run({"games": ok, "player_logs": err})
        step = report["steps"]["bulk_results_ingest"]
        self.assertEqual(step["result"]["status"], "partial_success")


class LoaderSeasonAndDbPathTests(_TmpDb):
    def test_current_season_rollover(self):
        self.assertEqual(current_nba_season(datetime(2026, 9, 25, tzinfo=timezone.utc)), "2025-26")
        self.assertEqual(current_nba_season(datetime(2026, 10, 1, tzinfo=timezone.utc)), "2026-27")

    def test_fetch_requests_current_season(self):
        loader = DataLoader(cache_dir=self._tmp.name, db_path=self.db_path)
        fake = MagicMock()
        fake.get_data_frames.return_value = [pd.DataFrame()]
        with patch.object(loader, "get_player_id", return_value=LEBRON_ID), \
             patch("nba_model.data.data_loader.playergamelogs.PlayerGameLogs",
                   return_value=fake) as endpoint, \
             patch.object(loader, "_clean_game_logs", return_value=pd.DataFrame()), \
             patch("nba_model.data.data_loader.time.sleep"):
            loader.load_player_data("LeBron James", n_games=1, force_refresh=True)
        self.assertEqual(endpoint.call_args.kwargs["season_nullable"], current_nba_season())

    def test_hourly_refresh_uses_the_run_db_path(self):
        with patch("nba_model.data.data_loader.DataLoader") as loader_cls:
            hourly_update._run_game_log_refresh(self.db_path, max_players=5)
        loader_cls.assert_called_once_with(db_path=self.db_path)


class OddsApiKeyRedactionTests(unittest.TestCase):
    def test_request_errors_never_carry_the_key(self):
        import requests
        exc = requests.ConnectionError(
            "Max retries exceeded with url: /v4/sports?apiKey=SECRETKEY123&dateFormat=iso")
        with patch("requests.get", side_effect=exc):
            with self.assertRaises(requests.ConnectionError) as ctx:
                oi._get_json("https://api.test", {"apiKey": "SECRETKEY123"}, retries=0)
        self.assertNotIn("SECRETKEY123", str(ctx.exception))
        self.assertIn("apiKey=***", str(ctx.exception))


class MlbFinalGamesOnlyTests(_TmpDb):
    def test_live_game_is_not_ingested(self):
        from nba_model.data import mlb_results_ingestion as mri
        schedule = [{"game_pk": 1, "status": "Live", "game_date": "2026-09-25", "season": 2026},
                    {"game_pk": 2, "status": "Final", "game_date": "2026-09-25", "season": 2026}]
        with patch.object(mri, "fetch_schedule", return_value={}), \
             patch.object(mri, "transform_schedule", return_value=schedule), \
             patch.object(mri, "fetch_boxscore", return_value={}) as box, \
             patch.object(mri, "transform_boxscore_to_player_logs", return_value=[]):
            mri.ingest_date_range("2026-09-25", "2026-09-25", db_path=self.db_path)
        self.assertEqual([c.args[0] for c in box.call_args_list], [2])


if __name__ == "__main__":
    unittest.main()
