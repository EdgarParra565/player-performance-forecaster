"""Tests for Edge Scanner "full" mode (WS9 — unify scoring).

Full mode drives the same prop_board projection stack the hourly ``predictions``
recompute uses: rolling μ/σ + ``blend_team_prior`` (pace / implied team total) +
the per-stat default distribution from ``get_default_distribution``. These tests
seed a throwaway SQLite DB and assert:

  * full mode ≠ chart_mean when a team prior is present (rolling window + blend),
  * rebounds carries the ``poisson`` default distribution,
  * the LeBron user story still holds in full mode (line 17.5, μ≈19 → p_over>.5),
  * the SCORED_COLUMNS prefix + appended mode/dist columns (Agent B contract),
  * scanner full mode reproduces the numbers ``_build_board_lines`` (the hourly
    path) produces for the same inputs — i.e. no scanner/hourly drift.

All offline (DB-direct; full mode never calls DataLoader / the NBA API), with
Windows-safe teardown.
"""

import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model import edge_scanner as es
from nba_model.model import prop_board

LEBRON_ID = 2544
NEUTRAL_ID = 9002

# Newest-10 points averaging exactly 19.0 (matches the base LeBron story).
NEWEST_10_POINTS = [16, 18, 20, 22, 16, 18, 20, 22, 18, 20]
OLD_POINTS = 30  # five older games, so chart_mean (all games) ≠ rolling-10.


def _utc(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _game_row(pid, i, points, rebounds=8):
    return {
        "player_id": pid, "game_id": f"g{pid}_{i}",
        "game_date": f"2025-04-{i:02d}", "season": "2024-25",
        "matchup": "LAL vs. DEN", "home_away": "home",
        "result": "W", "minutes": 34.0,
        "points": points, "rebounds": rebounds, "assists": 7,
        "fgm": 8, "fga": 16, "fg3m": 2, "fg3a": 6, "ftm": 4, "fta": 5,
        "oreb": 2, "dreb": 6, "steals": 1, "blocks": 0, "turnovers": 3,
        "plus_minus": 5,
    }


def _card(book, player, stat, line, side, observed, idx):
    return {
        "snapshot_id": 1,
        "source_url": f"https://{book}.test/nba",
        "book": book,
        "observed_at_utc": observed,
        "player_name": player,
        "player_classification": "active_nba",
        "stat_type": stat,
        "line_value": line,
        "side": side,
        "parse_confidence": 0.99,
        "parser_version": "test-1",
        "record_sha256": f"sha-{book}-{player}-{stat}-{side}-{idx}",
    }


class EdgeScannerFullTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = str(Path(self._tmp.name) / "nba.db")
        now = datetime.now(timezone.utc)
        self.recent = _utc(now - timedelta(hours=1))

        # LeBron: 5 old 30-pt games (i=1..5) + newest 10 averaging 19 (i=6..15).
        lebron_rows = [_game_row(LEBRON_ID, i, OLD_POINTS) for i in range(1, 6)]
        lebron_rows += [
            _game_row(LEBRON_ID, 6 + j, p)
            for j, p in enumerate(NEWEST_10_POINTS)
        ]
        # Neutral Guy: 10 games averaging 19, on a team with NO prior.
        neutral_rows = [
            _game_row(NEUTRAL_ID, 1 + j, p)
            for j, p in enumerate(NEWEST_10_POINTS)
        ]

        with DatabaseManager(db_path=self.db_path) as db:
            db.upsert_active_players_reference([
                {"player_id": LEBRON_ID, "player_name": "LeBron James",
                 "synced_at_utc": self.recent},
                {"player_id": NEUTRAL_ID, "player_name": "Neutral Guy",
                 "synced_at_utc": self.recent},
            ])
            db.insert_game_logs(pd.DataFrame(lebron_rows))
            db.insert_game_logs(pd.DataFrame(neutral_rows))
            # LeBron has a team (LAL); Neutral Guy is absent from players → no prior.
            db.conn.executemany(
                "INSERT INTO players (player_id, name, team) VALUES (?, ?, ?)",
                [(LEBRON_ID, "LeBron James", "LAL")],
            )
            db.conn.commit()
            # Team prior: pace 1.2 nudges LAL projections up (factor 1.06 at
            # alpha=0.3). No `games` rows → team_recent_avg_total is None, so the
            # implied-team-total branch is skipped and the blend is pace-only.
            db.upsert_team_priors([{
                "away_team": "DEN", "home_team": "LAL",
                "computed_at_utc": self.recent,
                "consensus_total": 235.0, "home_spread": -4.0, "away_spread": 4.0,
                "home_team_total": 119.5, "away_team_total": 115.5,
                "home_win_prob_devig": 0.6, "away_win_prob_devig": 0.4,
                "pace_factor": 1.2, "n_books": 3,
                "latest_observed_at": self.recent,
            }])
            db.insert_web_prop_cards([
                _card("Underdog", "LeBron James", "points", 17.5, "over",
                      self.recent, 1),
                _card("Underdog", "LeBron James", "rebounds", 6.5, "over",
                      self.recent, 2),
                _card("Underdog", "Neutral Guy", "points", 17.5, "over",
                      self.recent, 3),
            ])

    def tearDown(self):
        self._tmp.cleanup()

    def _scored(self, model_mode):
        lines = es.fetch_latest_prop_lines(self.db_path)
        return es.score_prop_edges(
            lines, db_path=self.db_path, n_games=25, model_mode=model_mode,
        )


class FullModeProjectionTests(EdgeScannerFullTestBase):
    def test_full_differs_from_chart_mean_with_prior(self):
        full = self._scored("full")
        chart = self._scored("chart_mean")
        lf = full[(full["player_name"] == "LeBron James") &
                  (full["stat_type"] == "points")].iloc[0]
        lc = chart[(chart["player_name"] == "LeBron James") &
                   (chart["stat_type"] == "points")].iloc[0]
        # chart_mean averages ALL 15 games; full uses the newest-10 rolling
        # mean (19.0) blended by the pace prior (×1.06).
        self.assertAlmostEqual(lc["model_mu"], 340.0 / 15.0, places=3)
        self.assertAlmostEqual(lf["model_mu"], 19.0 * 1.06, places=3)
        self.assertNotAlmostEqual(lf["model_mu"], lc["model_mu"], places=2)

    def test_full_mode_column_records_mode_and_distribution(self):
        full = self._scored("full")
        self.assertTrue((full["model_mode"] == "full").all())
        pts = full[full["stat_type"] == "points"].iloc[0]
        self.assertEqual(pts["distribution"], "normal")

    def test_rebounds_uses_poisson(self):
        full = self._scored("full")
        reb = full[(full["player_name"] == "LeBron James") &
                   (full["stat_type"] == "rebounds")].iloc[0]
        self.assertEqual(reb["distribution"], "poisson")
        # μ = rolling rebounds (8.0); line 6.5 below μ → over favored.
        self.assertGreater(reb["p_over"], 0.5)
        self.assertEqual(reb["best_side"], "over")

    def test_lebron_story_holds_without_prior(self):
        # Neutral Guy has no team prior → full μ is the raw rolling-10 mean (19).
        full = self._scored("full")
        ng = full[full["player_name"] == "Neutral Guy"].iloc[0]
        self.assertAlmostEqual(ng["model_mu"], 19.0, places=3)
        self.assertEqual(ng["book_line"], 17.5)
        self.assertGreater(ng["p_over"], 0.5)
        self.assertEqual(ng["best_side"], "over")

    def test_unprojectable_stat_dropped_in_full_mode(self):
        # A steals prop can't be projected by the rolling stack → row dropped,
        # never raised.
        with DatabaseManager(db_path=self.db_path) as db:
            db.insert_web_prop_cards([
                _card("Underdog", "LeBron James", "steals", 1.5, "over",
                      self.recent, 9),
            ])
        full = self._scored("full")
        self.assertFalse((full["stat_type"] == "steals").any())

    def test_rolling_window_below_two_drops_rows_not_raises(self):
        # add_rolling_stats requires window >= 2; full mode must swallow that
        # per-player and drop the row, never abort the whole slate scan.
        lines = es.fetch_latest_prop_lines(self.db_path)
        scored = es.score_prop_edges(
            lines, db_path=self.db_path, n_games=25,
            model_mode="full", rolling_window=1,
        )
        self.assertTrue(scored.empty)
        self.assertEqual(list(scored.columns), es.SCORED_COLUMNS_FULL)

    def test_single_null_in_window_yields_finite_mu_not_dropped(self):
        # Root fix: one NULL stat value inside the rolling window is *excluded*
        # from μ/σ (available-case) rather than voiding the projection. The row
        # survives with a finite μ (rebounds = 8 over the 9 valid games), and no
        # NaN leaks into any row's model_mu.
        with DatabaseManager(db_path=self.db_path) as db:
            db.upsert_active_players_reference([
                {"player_id": 9500, "player_name": "Null Guy",
                 "synced_at_utc": self.recent}])
            null_rows = []
            for j, p in enumerate(NEWEST_10_POINTS):
                row = _game_row(9500, j + 1, p)  # rebounds defaults to 8
                if j == len(NEWEST_10_POINTS) - 1:  # newest game: NULL rebounds
                    row["rebounds"] = None
                null_rows.append(row)
            db.insert_game_logs(pd.DataFrame(null_rows))
            db.insert_web_prop_cards([
                _card("Underdog", "Null Guy", "rebounds", 6.5, "over",
                      self.recent, 20)])
        full = self._scored("full")
        ng = full[full["player_name"] == "Null Guy"]
        self.assertEqual(len(ng), 1)
        self.assertTrue(math.isfinite(float(ng.iloc[0]["model_mu"])))
        # No team prior for Null Guy (absent from players) → μ is the plain mean.
        self.assertAlmostEqual(float(ng.iloc[0]["model_mu"]), 8.0, places=6)
        self.assertFalse(full["model_mu"].isna().any())

    def test_too_few_valid_in_window_is_dropped_not_nan_row(self):
        # When the window is mostly NULL (fewer than MIN_VALID_WINDOW_FRACTION of
        # the games survive), μ/σ stays NaN and full mode drops the row rather
        # than persist a NaN (which, for poisson stats, otherwise sorts to the
        # top as a fake edge).
        with DatabaseManager(db_path=self.db_path) as db:
            db.upsert_active_players_reference([
                {"player_id": 9600, "player_name": "Sparse Guy",
                 "synced_at_utc": self.recent}])
            sparse_rows = []
            for j, p in enumerate(NEWEST_10_POINTS):
                row = _game_row(9600, j + 1, p)
                if j >= 4:  # 6 of the newest 10 rebounds are NULL → 4 valid < 5
                    row["rebounds"] = None
                sparse_rows.append(row)
            db.insert_game_logs(pd.DataFrame(sparse_rows))
            db.insert_web_prop_cards([
                _card("Underdog", "Sparse Guy", "rebounds", 6.5, "over",
                      self.recent, 21)])
        full = self._scored("full")
        sg = full[full["player_name"] == "Sparse Guy"]
        self.assertTrue(sg.empty)
        self.assertFalse(full["model_mu"].isna().any())


class FullModeContractTests(EdgeScannerFullTestBase):
    def test_scored_columns_prefix_and_appended(self):
        self.assertEqual(
            es.SCORED_COLUMNS_FULL[: len(es.SCORED_COLUMNS)], es.SCORED_COLUMNS)
        self.assertEqual(es.SCORED_COLUMNS_FULL[len(es.SCORED_COLUMNS):],
                         ["distribution", "model_mode"])
        full = self._scored("full")
        self.assertEqual(list(full.columns), es.SCORED_COLUMNS_FULL)
        self.assertEqual(
            list(full.columns)[: len(es.SCORED_COLUMNS)], es.SCORED_COLUMNS)

    def test_model_modes_accepts_all_three(self):
        self.assertEqual(es.MODEL_MODES, ("chart_mean", "rolling", "full"))
        for mode in es.MODEL_MODES:
            if mode == "rolling":
                continue  # rolling uses DataLoader (needs network) — skip here.
            out = self._scored(mode)
            self.assertFalse(out.empty)
            self.assertTrue((out["model_mode"] == mode).all())


class FullModeHourlyParityTests(EdgeScannerFullTestBase):
    """Scanner full mode must reproduce the numbers the hourly recompute
    (_build_board_lines → _persist_predictions) stores for identical inputs."""

    def test_scanner_full_matches_prop_board(self):
        full = self._scored("full")
        srow = full[(full["player_name"] == "LeBron James") &
                    (full["stat_type"] == "points")].iloc[0]

        with DatabaseManager(db_path=self.db_path) as db:
            tp_map = db.get_team_prior_inputs_map()
            games = db.get_player_games(LEBRON_ID, n_games=25)
        latest = prop_board.build_history_from_games(games, rolling_window=10)

        board_rows = pd.DataFrame([{
            "game_date": "2025-04-16",
            "player_id": LEBRON_ID,
            "player_name": "LeBron James",
            "team": "LAL",
            "book": "Underdog",
            "stat_type": "points",
            "line_value": 17.5,
            "over_odds": None,
            "under_odds": None,
        }])
        board = prop_board._build_board_lines(
            rows=board_rows,
            player_histories={"LeBron James": latest},
            rolling_window=10,
            team_priors=tp_map,
        )
        self.assertEqual(len(board), 1)
        bl = board[0]

        self.assertEqual(round(float(bl.mu), 3), srow["model_mu"])
        self.assertEqual(round(float(bl.sigma), 3), srow["model_sigma"])
        self.assertEqual(round(float(bl.prob_over), 4), srow["p_over"])
        self.assertEqual(bl.distribution, srow["distribution"])


if __name__ == "__main__":
    unittest.main()
