"""Tests for the hourly ETL runner.

Coverage:
- Preflight guard fires when Chrome :9222 is unreachable or Playwright is
  missing (the hourly job MUST fail loudly in those cases, not silently
  retry every URL).
- The lockfile blocks overlapping runs so a stuck hourly job doesn't get
  trampled by the next tick.
- Successful + partial-failure runs both emit a JSON report with the
  per-step shape callers can grep for.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nba_model.data import hourly_update


class CheckChromeCdpReachableTests(unittest.TestCase):
    def test_unreachable_port_returns_ok_false_with_actionable_error(self):
        from nba_model.model.web_text_ingestion import check_chrome_cdp_reachable

        # Port 1 is universally unbound on a normal user account, so this
        # check is platform-stable.
        result = check_chrome_cdp_reachable(1, host="127.0.0.1", timeout_seconds=0.5)
        self.assertFalse(result["ok"])
        self.assertIn("Chrome CDP unreachable", result["error"])
        self.assertIsNone(result["version"])


class PreflightTests(unittest.TestCase):
    def test_preflight_raises_when_chrome_unreachable(self):
        with patch(
            "nba_model.model.web_text_ingestion.playwright_is_available",
            return_value=True,
        ), patch(
            "nba_model.model.web_text_ingestion.check_chrome_cdp_reachable",
            return_value={"ok": False, "version": None, "error": "boom"},
        ):
            with self.assertRaises(RuntimeError) as ctx:
                hourly_update._run_preflight(
                    chrome_port=9222, chrome_host="127.0.0.1",
                    require_playwright=False,
                )
            self.assertIn("boom", str(ctx.exception))
            self.assertIn("--remote-debugging-port", str(ctx.exception))

    def test_preflight_raises_when_playwright_missing_and_required(self):
        with patch(
            "nba_model.model.web_text_ingestion.playwright_is_available",
            return_value=False,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                hourly_update._run_preflight(
                    chrome_port=9222, chrome_host="127.0.0.1",
                    require_playwright=True,
                )
            self.assertIn("Playwright", str(ctx.exception))
            self.assertIn(".venv/bin/python3", str(ctx.exception))

    def test_preflight_succeeds_when_both_ok(self):
        with patch(
            "nba_model.model.web_text_ingestion.playwright_is_available",
            return_value=True,
        ), patch(
            "nba_model.model.web_text_ingestion.check_chrome_cdp_reachable",
            return_value={"ok": True, "version": "Chrome/120", "error": None},
        ):
            result = hourly_update._run_preflight(
                chrome_port=9222, chrome_host="127.0.0.1",
                require_playwright=True,
            )
        self.assertTrue(result["chrome"]["ok"])
        self.assertEqual(result["chrome"]["version"], "Chrome/120")


class LockfileTests(unittest.TestCase):
    def test_second_acquire_returns_false_when_already_held(self):
        with tempfile.TemporaryDirectory() as tmp:
            lockfile = os.path.join(tmp, "test.lock")
            with hourly_update._acquire_lock(lockfile) as acquired_first:
                self.assertTrue(acquired_first)
                with hourly_update._acquire_lock(lockfile) as acquired_second:
                    self.assertFalse(acquired_second)


class RunHourlyUpdateReportTests(unittest.TestCase):
    def _patched_run(self, tmpdir: str) -> dict:
        """Run with every step shimmed to a fast return so we exercise the
        report-writer + exit-code branches without touching the network."""
        patches = [
            patch.object(hourly_update, "_run_preflight",
                         return_value={"playwright_available": True,
                                       "chrome": {"ok": True}}),
            patch.object(hourly_update, "_run_web_text",
                         return_value={"urls": 3, "fetched": 3}),
            patch.object(hourly_update, "_run_browser_prop_parser",
                         return_value={"snapshots_parsed": 2}),
            patch.object(hourly_update, "_run_team_line_parser",
                         return_value={"snapshots_parsed": 1}),
            patch.object(hourly_update, "_run_game_log_refresh",
                         return_value={"players_refreshed": 5, "failures": []}),
            patch.object(hourly_update, "_run_players_table_sync",
                         return_value={"upserted": 530, "attempted": 530,
                                       "patched_from_game_logs": 1}),
            patch.object(hourly_update, "_run_reverse_engineering",
                         return_value={"updated": 12}),
            patch.object(hourly_update, "_run_outcome_settlement",
                         return_value={"settled": 0, "pending": 4}),
            patch.object(hourly_update, "_run_prediction_recompute",
                         return_value={"scored": 7}),
        ]
        for p in patches:
            p.start()
        try:
            return hourly_update.run_hourly_update(
                report_dir=tmpdir,
                require_playwright=False,
            )
        finally:
            for p in patches:
                p.stop()

    def test_full_run_writes_report_with_all_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self._patched_run(tmp)
            self.assertTrue(report["ok"])
            self.assertEqual(report["exit_code"], hourly_update.EXIT_OK)
            self.assertEqual(report["failed_steps"], [])
            for step in (
                "preflight", "web_text", "browser_prop_parser",
                "team_line_parser", "game_log_refresh",
                "players_table_sync",
                "reverse_engineering", "outcome_settlement",
                "prediction_recompute",
            ):
                self.assertIn(step, report["steps"])
                self.assertTrue(report["steps"][step]["ok"], f"step {step}")
            self.assertTrue(report["report_path"].endswith(".json"))
            saved = json.loads(Path(report["report_path"]).read_text())
            self.assertTrue(saved["ok"])

    def test_preflight_failure_short_circuits_with_exit_78(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            hourly_update, "_run_preflight",
            side_effect=RuntimeError("chrome down"),
        ):
            report = hourly_update.run_hourly_update(
                report_dir=tmp, require_playwright=False,
            )
        self.assertFalse(report["ok"])
        self.assertEqual(report["exit_code"], hourly_update.EXIT_PREFLIGHT_FAILED)
        self.assertIn("preflight", report["failed_steps"])
        # No downstream step should have been recorded — preflight blocks the rest.
        for step in ("web_text", "browser_prop_parser", "team_line_parser"):
            self.assertNotIn(step, report["steps"])

    def test_step_failure_records_traceback_and_exits_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            patches = [
                patch.object(hourly_update, "_run_preflight",
                             return_value={"playwright_available": True,
                                           "chrome": {"ok": True}}),
                patch.object(hourly_update, "_run_web_text",
                             side_effect=RuntimeError("network blew up")),
                patch.object(hourly_update, "_run_browser_prop_parser",
                             return_value={"snapshots_parsed": 0}),
                patch.object(hourly_update, "_run_team_line_parser",
                             return_value={"snapshots_parsed": 0}),
                patch.object(hourly_update, "_run_game_log_refresh",
                             return_value={"players_refreshed": 0, "failures": []}),
                patch.object(hourly_update, "_run_players_table_sync",
                             return_value={"upserted": 0, "attempted": 0,
                                           "patched_from_game_logs": 0}),
                patch.object(hourly_update, "_run_reverse_engineering",
                             return_value={"updated": 0}),
                patch.object(hourly_update, "_run_outcome_settlement",
                             return_value={"settled": 0}),
                patch.object(hourly_update, "_run_prediction_recompute",
                             return_value={"scored": 0}),
            ]
            for p in patches:
                p.start()
            try:
                report = hourly_update.run_hourly_update(
                    report_dir=tmp, require_playwright=False,
                )
            finally:
                for p in patches:
                    p.stop()
        self.assertFalse(report["ok"])
        self.assertEqual(report["exit_code"], hourly_update.EXIT_STEP_FAILED)
        self.assertEqual(report["failed_steps"], ["web_text"])
        self.assertIn("traceback", report["steps"]["web_text"])


class BetLogSettlementStepTests(unittest.TestCase):
    """The optional bet_log settlement step is flag-gated (default OFF)."""

    def _core_patches(self):
        return [
            patch.object(hourly_update, "_run_preflight",
                         return_value={"playwright_available": True,
                                       "chrome": {"ok": True}}),
            patch.object(hourly_update, "_run_web_text", return_value={"urls": 0}),
            patch.object(hourly_update, "_run_browser_prop_parser", return_value={}),
            patch.object(hourly_update, "_run_team_line_parser", return_value={}),
            patch.object(hourly_update, "_run_game_log_refresh",
                         return_value={"players_refreshed": 0, "failures": []}),
            patch.object(hourly_update, "_run_players_table_sync", return_value={}),
            patch.object(hourly_update, "_run_reverse_engineering", return_value={}),
            patch.object(hourly_update, "_run_outcome_settlement", return_value={}),
            patch.object(hourly_update, "_run_prediction_recompute",
                         return_value={"scored": 0}),
        ]

    def _run(self, tmp, **kwargs):
        patches = self._core_patches()
        patches.append(patch.object(
            hourly_update, "_run_bet_log_settlement",
            return_value={"settle": {"settled": 1}, "calibration": {"settled_rows": 1}}))
        for p in patches:
            p.start()
        try:
            return hourly_update.run_hourly_update(
                report_dir=tmp, require_playwright=False, **kwargs)
        finally:
            for p in patches:
                p.stop()

    def test_step_absent_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self._run(tmp)
        self.assertNotIn("bet_log_settlement", report["steps"])

    def test_step_present_when_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self._run(tmp, settle_bet_log=True)
        self.assertIn("bet_log_settlement", report["steps"])
        self.assertTrue(report["steps"]["bet_log_settlement"]["ok"])


class PersistPredictionsTests(unittest.TestCase):
    """_persist_predictions widens the predictions table and is idempotent."""

    def _board_line(self, stat, line, mu):
        from nba_model.model.prop_board import BoardLine
        return BoardLine(
            game_date="2025-04-01", team="LAL", player_name="LeBron James",
            stat_type=stat, line_value=line, over_odds=-110, under_odds=-110,
            mu=mu, sigma=5.0, distribution="normal", prob_over=0.55,
            implied_over_prob=0.52, implied_under_prob=0.52,
            ev_over=0.06, ev_under=-0.04,
        )

    def test_persists_multiple_stats_idempotently(self):
        from nba_model.data.database.db_manager import DatabaseManager
        from nba_model.data.hourly_update import _persist_predictions
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "nba.db")
            with DatabaseManager(db_path=db_path):
                pass
            lines = [
                self._board_line("points", 25.5, 26.0),
                self._board_line("assists", 7.5, 8.0),
                self._board_line("rebounds", 7.5, 8.0),
            ]
            name_to_id = {"LeBron James": 2544}
            n1 = _persist_predictions(db_path, lines, name_to_id, "2025-04-01")
            self.assertEqual(n1, 3)
            # Re-run (next hourly tick): still 3 rows total, not 6.
            _persist_predictions(db_path, lines, name_to_id, "2025-04-01")
            with DatabaseManager(db_path=db_path) as db:
                total = db.conn.execute(
                    "SELECT COUNT(*) FROM predictions").fetchone()[0]
                stats = {r[0] for r in db.conn.execute(
                    "SELECT DISTINCT stat_type FROM predictions").fetchall()}
            self.assertEqual(total, 3)
            self.assertEqual(stats, {"points", "assists", "rebounds"})

    def test_skips_unknown_player(self):
        from nba_model.data.database.db_manager import DatabaseManager
        from nba_model.data.hourly_update import _persist_predictions
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "nba.db")
            with DatabaseManager(db_path=db_path):
                pass
            lines = [self._board_line("points", 25.5, 26.0)]
            n = _persist_predictions(db_path, lines, {}, "2025-04-01")
        self.assertEqual(n, 0)

    def test_skips_non_finite_moment(self):
        # Defense in depth: a NaN projection (too few valid games survived the
        # NULL-in-window repair) must never be persisted. SQLite would store the
        # NaN REAL as NULL and read it back as a bogus prediction.
        from nba_model.data.database.db_manager import DatabaseManager
        from nba_model.data.hourly_update import _persist_predictions
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "nba.db")
            with DatabaseManager(db_path=db_path):
                pass
            good = self._board_line("points", 25.5, 26.0)
            bad = self._board_line("rebounds", 6.5, float("nan"))
            n = _persist_predictions(
                db_path, [good, bad], {"LeBron James": 2544}, "2025-04-01")
            with DatabaseManager(db_path=db_path) as db:
                rows = db.conn.execute(
                    "SELECT stat_type, predicted_mean FROM predictions"
                ).fetchall()
        self.assertEqual(n, 1)
        self.assertEqual([r[0] for r in rows], ["points"])
        self.assertFalse(any(r[1] is None for r in rows))


def _games_frame(pid, rebounds_per_game):
    """A get_player_games-style frame (newest-first / DESC) with 12 games."""
    import pandas as pd
    points = [16, 18, 20, 22, 16, 18, 20, 22, 18, 20, 24, 26]
    rows = []
    for i, (p, r) in enumerate(zip(points, rebounds_per_game), start=1):
        rows.append({
            "player_id": pid, "game_id": f"g{pid}_{i}",
            "game_date": f"2025-04-{i:02d}", "minutes": 34.0,
            "points": float(p), "rebounds": r, "assists": 7.0,
        })
    return pd.DataFrame(rows[::-1]).reset_index(drop=True)  # DESC by date


class NanRowRecomputeDefenseTests(unittest.TestCase):
    """The hourly recompute path (build_history -> _build_board_lines ->
    _persist_predictions) must not write NaN rows for NULL-in-window data."""

    def test_hourly_style_recompute_over_null_data_writes_no_nan(self):
        import pandas as pd
        from nba_model.data.database.db_manager import DatabaseManager
        from nba_model.data.hourly_update import _persist_predictions
        from nba_model.model.prop_board import (
            _build_board_lines,
            build_history_from_games,
        )

        # Null Guy: 1 NULL in the trailing window -> finite μ over survivors.
        # Sparse Guy: 6 NULLs -> μ stays NaN -> must be dropped, not persisted.
        null_reb = [8.0] * 11 + [None]
        sparse_reb = [8.0] * 6 + [None] * 6
        histories = {
            "Null Guy": build_history_from_games(_games_frame(700, null_reb), 10),
            "Sparse Guy": build_history_from_games(_games_frame(701, sparse_reb), 10),
        }
        rows = pd.DataFrame([
            {"game_date": "2025-04-12", "player_id": 700,
             "player_name": "Null Guy", "team": None, "book": "Underdog",
             "stat_type": "rebounds", "line_value": 6.5,
             "over_odds": None, "under_odds": None},
            {"game_date": "2025-04-12", "player_id": 701,
             "player_name": "Sparse Guy", "team": None, "book": "Underdog",
             "stat_type": "rebounds", "line_value": 6.5,
             "over_odds": None, "under_odds": None},
        ])
        board = _build_board_lines(
            rows=rows, player_histories=histories, rolling_window=10)
        name_to_id = {"Null Guy": 700, "Sparse Guy": 701}

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "nba.db")
            with DatabaseManager(db_path=db_path):
                pass
            persisted = _persist_predictions(
                db_path, board, name_to_id, "2025-04-12")
            with DatabaseManager(db_path=db_path) as db:
                all_rows = db.conn.execute(
                    "SELECT player_id, predicted_mean, predicted_std, prob_over "
                    "FROM predictions"
                ).fetchall()
                nan_rows = db.conn.execute(
                    "SELECT COUNT(*) FROM predictions WHERE predicted_mean IS NULL "
                    "OR predicted_std IS NULL OR prob_over IS NULL"
                ).fetchone()[0]

        self.assertEqual(persisted, 1)                # only Null Guy survives
        self.assertEqual([r[0] for r in all_rows], [700])
        self.assertEqual(nan_rows, 0)                 # no NaN leaked into DB

    def test_delete_nonfinite_predictions_cleans_persisted_nan(self):
        from nba_model.data.database.db_manager import DatabaseManager
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                db.insert_prediction({
                    "player_id": 1, "game_date": "2025-04-12",
                    "stat_type": "points", "predicted_mean": 20.0,
                    "predicted_std": 5.0, "prob_over": 0.55, "line_value": 17.5,
                })
                # A pre-fix NaN row: SQLite stores the NaN REAL as NULL.
                db.insert_prediction({
                    "player_id": 2, "game_date": "2025-04-12",
                    "stat_type": "rebounds", "predicted_mean": float("nan"),
                    "predicted_std": float("nan"), "prob_over": float("nan"),
                    "line_value": 6.5,
                })
                removed = db.delete_nonfinite_predictions()
                remaining = db.conn.execute(
                    "SELECT player_id FROM predictions").fetchall()
                # Idempotent: a second sweep removes nothing.
                removed_again = db.delete_nonfinite_predictions()
        self.assertEqual(removed, 1)
        self.assertEqual([r[0] for r in remaining], [1])
        self.assertEqual(removed_again, 0)


if __name__ == "__main__":
    unittest.main()
