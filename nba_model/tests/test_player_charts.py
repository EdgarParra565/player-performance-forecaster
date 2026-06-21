"""Tests for nba_model/visualization/player_charts.py.

Covers:
  - Canonical stat resolution + alias collapse
  - ``_series_for_stat`` numerical extraction (incl. PRA + RA composites)
  - Market consensus mean across heterogenous book lines
  - Custom-line probe math (P(over), EV per unit, +EV / -EV verdicts)
  - End-to-end DB round-trip for fetch_player_chart_data + the new
    Game Results / Player Stats Browse query helpers
  - Figure builders produce a matplotlib ``Figure`` (no crash on empty
    data, plus a happy path with bars + book-mean line)

The DB round-trip seeds a temp SQLite via ``DatabaseManager`` (which
re-applies the schema on init) so tests don't touch ``data/database/``.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend so figures don't try to open windows
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.visualization import player_charts as pc


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CanonicalStatTypeTests(unittest.TestCase):
    def test_alias_collapse(self):
        self.assertEqual(pc._canonical_stat_type("3pm"), "three_pointers_made")
        self.assertEqual(pc._canonical_stat_type("fgm"), "field_goals_made")
        self.assertEqual(pc._canonical_stat_type("Three Pointers"),
                         "three_pointers_made")

    def test_unknown_passthrough(self):
        # Unknown stat returns the cleaned key — callers downstream decide
        # what to do.  We deliberately don't raise here because the function
        # is also used to normalize column lookups, not to validate input.
        self.assertEqual(pc._canonical_stat_type("turnovers"), "turnovers")
        self.assertEqual(pc._canonical_stat_type("  "), "")


class SeriesForStatTests(unittest.TestCase):
    def _df(self):
        return pd.DataFrame({
            "points":   [20, 25, 30],
            "rebounds": [10, 12, 8],
            "assists":  [5, 7, 6],
            "minutes":  [32.0, 28.5, 35.2],
        })

    def test_direct_column(self):
        out = pc._series_for_stat(self._df(), "points")
        np.testing.assert_array_equal(out, [20, 25, 30])

    def test_pra_composite(self):
        # PRA = points + rebounds + assists
        out = pc._series_for_stat(self._df(), "pra")
        np.testing.assert_array_equal(out, [35, 44, 44])

    def test_ra_composite(self):
        # RA = rebounds + assists
        out = pc._series_for_stat(self._df(), "ra")
        np.testing.assert_array_equal(out, [15, 19, 14])

    def test_unknown_stat_returns_empty(self):
        out = pc._series_for_stat(self._df(), "unknown_stat")
        self.assertEqual(out.size, 0)

    def test_missing_columns_returns_empty(self):
        thin = pd.DataFrame({"points": [10, 12]})
        # rebounds + assists not present → empty (not zero-fill)
        self.assertEqual(pc._series_for_stat(thin, "pra").size, 0)
        self.assertEqual(pc._series_for_stat(thin, "ra").size, 0)


class MarketConsensusLineTests(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(pc._market_consensus_line(pd.DataFrame()))

    def test_all_nan_returns_none(self):
        df = pd.DataFrame({"line_value": [None, None]})
        self.assertIsNone(pc._market_consensus_line(df))

    def test_mean_across_books(self):
        df = pd.DataFrame({
            "book": ["a", "b", "c"],
            "line_value": [25.0, 26.0, 27.0],
        })
        # Mean of 25, 26, 27 is 26.0
        self.assertAlmostEqual(pc._market_consensus_line(df), 26.0, places=4)

    def test_string_lines_coerce(self):
        df = pd.DataFrame({"line_value": ["25.5", "26.5"]})
        self.assertAlmostEqual(pc._market_consensus_line(df), 26.0, places=4)


class CustomLineProbeTests(unittest.TestCase):
    """``evaluate_custom_line`` produces EV per unit + +/-EV verdict."""

    def _data_with_history(self, values, line=25.0):
        # Build a PlayerChartData with a known historical distribution.
        return pc.PlayerChartData(
            player_id=1,
            player_name="Test Player",
            stat_type="points",
            games=pd.DataFrame({
                "game_date": [f"2024-01-{d:02d}" for d in range(1, len(values) + 1)],
            }),
            values=np.array(values, dtype=float),
            book_lines=pd.DataFrame(),
            market_consensus_line=line,
        )

    def test_breakeven_math_on_minus_110(self):
        # -110 → break-even 0.5238; an exact match should give EV ~ 0
        self.assertAlmostEqual(
            pc._american_odds_breakeven(-110), 0.5238, places=3,
        )
        self.assertAlmostEqual(
            pc._american_odds_breakeven(+110), 0.4762, places=3,
        )

    def test_expected_value_per_unit(self):
        # +100 (decimal 2.0) with 60% win prob: EV = 0.6 * 1 + 0.4 * -1 = 0.2
        ev = pc.expected_value(0.6, 100)
        self.assertAlmostEqual(ev, 0.2, places=4)

        # -110 with 50% prob: payout for 1 unit = 100/110 = 0.909.
        # EV = 0.5 * 0.909 + 0.5 * -1 = -0.0455
        ev = pc.expected_value(0.5, -110)
        self.assertAlmostEqual(ev, -0.0455, places=3)

    def test_evaluate_custom_line_marks_positive_ev(self):
        # Distribution averaging ~30 points; line at 25 — P(over) very high.
        data = self._data_with_history([28, 29, 30, 31, 32, 29, 30], line=25.0)
        result = pc.evaluate_custom_line(data, line=25.0, american_odds=-110)
        self.assertGreater(result["p_over"], 0.7)
        # A heavy +EV value: EV should clearly be positive at the over.
        self.assertGreater(result["ev_over_per_unit"], 0.0)
        # Historical hit rate matches the values above 25 (all 7 → 1.0).
        self.assertEqual(result["historical_over_rate"], 1.0)


class DBRoundTripTests(unittest.TestCase):
    """Verify the chart pipeline reads from the new tables correctly."""

    def _seed(self, db_path: str):
        with DatabaseManager(db_path=db_path) as db:
            db.upsert_active_players_reference([{
                "player_id": 2544,
                "player_name": "LeBron James",
                "synced_at_utc": _utc_now_iso(),
            }])
            db.insert_game_logs(pd.DataFrame([
                {
                    "player_id": 2544, "game_id": f"g{i}",
                    "game_date": f"2025-04-{i:02d}", "season": "2024-25",
                    "matchup": "LAL vs. DEN" if i % 2 else "LAL @ DEN",
                    "home_away": "home" if i % 2 else "away",
                    "result": "W", "minutes": 32.0,
                    "points": 20 + i, "rebounds": 8, "assists": 7,
                    "fgm": 8, "fga": 16,
                    "fg3m": 2, "fg3a": 6,
                    "ftm": 4, "fta": 5,
                    "oreb": 2, "dreb": 6,
                    "steals": 1, "blocks": 0, "turnovers": 3,
                    "plus_minus": 5,
                }
                for i in range(1, 11)
            ]))

    def test_fetch_player_chart_data_returns_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            self._seed(db_path)
            data = pc.fetch_player_chart_data(
                db_path=db_path,
                player_id=2544,
                player_name="LeBron James",
                stat_type="points",
                n_games=10,
            )

        self.assertEqual(data.player_id, 2544)
        self.assertEqual(data.player_name, "LeBron James")
        self.assertGreater(data.values.size, 0)
        # Points seeded as 20+i for i=1..10 → mean ~25.5
        self.assertAlmostEqual(float(np.mean(data.values)), 25.5, places=2)

    def test_fetch_recent_games_returns_matchups(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                db.insert_games([
                    {
                        "game_id": "0022400001", "season": "2024-25",
                        "season_type": "Regular Season",
                        "game_date": "2025-04-15",
                        "team_id": 1610612747, "team_abbrev": "LAL",
                        "team_name": "Los Angeles Lakers",
                        "matchup": "LAL vs. DEN", "home_away": "home",
                        "opponent_abbrev": "DEN", "result": "W",
                        "pts": 120, "opp_pts": 110,
                    },
                    {
                        "game_id": "0022400001", "season": "2024-25",
                        "season_type": "Regular Season",
                        "game_date": "2025-04-15",
                        "team_id": 1610612743, "team_abbrev": "DEN",
                        "team_name": "Denver Nuggets",
                        "matchup": "DEN @ LAL", "home_away": "away",
                        "opponent_abbrev": "LAL", "result": "L",
                        "pts": 110, "opp_pts": 120,
                    },
                ])

            df = pc.fetch_recent_games(db_path=db_path, n=10)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["away_abbrev"], "DEN")
        self.assertEqual(row["home_abbrev"], "LAL")
        self.assertEqual(row["winner"], "LAL")

    def test_fetch_player_recent_results_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            self._seed(db_path)

            # Filter to "last game with at least 25 points" → 6 games
            df = pc.fetch_player_recent_results(
                db_path=db_path, n=20, stat="points", min_value=25,
            )

        self.assertGreater(len(df), 0)
        for v in df["points"]:
            self.assertGreaterEqual(v, 25)


class UpcomingGameLineTests(unittest.TestCase):
    """_fetch_latest_book_lines should prefer the next (today/upcoming) game's
    line, not simply max(game_date) or yesterday's slate."""

    def _dates(self):
        from datetime import timedelta
        today = datetime.now(timezone.utc).date()
        return {
            "today": today.isoformat(),
            "tomorrow_wk": (today + timedelta(days=7)).isoformat(),
            "yesterday": (today - timedelta(days=1)).isoformat(),
            "last_week": (today - timedelta(days=7)).isoformat(),
        }

    def _fetch(self, db_path, lines):
        with DatabaseManager(db_path=db_path) as db:
            db.insert_betting_lines_records(lines)
            return pc._fetch_latest_book_lines(db, 2544, "points")

    def _line(self, game_date, value):
        return {
            "player_id": 2544, "game_date": game_date, "book": "FanDuel",
            "stat_type": "points", "line_value": value,
            "over_odds": -110, "under_odds": -110,
        }

    def test_nearest_upcoming_beats_further_future(self):
        d = self._dates()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            df = self._fetch(db_path, [
                self._line(d["today"], 26.5),
                self._line(d["tomorrow_wk"], 30.5),
            ])
        self.assertEqual(len(df), 1)
        self.assertEqual(float(df.iloc[0]["line_value"]), 26.5)

    def test_today_beats_yesterday(self):
        d = self._dates()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            df = self._fetch(db_path, [
                self._line(d["yesterday"], 24.5),
                self._line(d["today"], 26.5),
            ])
        self.assertEqual(float(df.iloc[0]["line_value"]), 26.5)

    def test_falls_back_to_most_recent_past(self):
        d = self._dates()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            df = self._fetch(db_path, [
                self._line(d["last_week"], 20.5),
                self._line(d["yesterday"], 24.5),
            ])
        # No upcoming line → most recent past (yesterday) wins.
        self.assertEqual(float(df.iloc[0]["line_value"]), 24.5)


class LineMovementSnapshotTests(unittest.TestCase):
    """fetch_line_movement_snapshots reads betting_line_snapshots correctly."""

    def _seed(self, db_path: str):
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        recent = now.strftime("%Y-%m-%d %H:%M:%S")
        earlier = (now - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
        stale = (now - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
        with DatabaseManager(db_path=db_path) as db:
            db.insert_betting_line_snapshots([
                # Two books, drift over time, within the lookback window.
                {"snapshot_ts_utc": earlier, "game_date": "2025-04-15",
                 "player_id": 2544, "book": "fanduel", "market_key": "pts",
                 "stat_type": "points", "line_value": 25.5,
                 "over_odds": -110, "under_odds": -110},
                {"snapshot_ts_utc": recent, "game_date": "2025-04-15",
                 "player_id": 2544, "book": "fanduel", "market_key": "pts",
                 "stat_type": "points", "line_value": 26.5,
                 "over_odds": -115, "under_odds": -105},
                {"snapshot_ts_utc": recent, "game_date": "2025-04-15",
                 "player_id": 2544, "book": "draftkings", "market_key": "pts",
                 "stat_type": "points", "line_value": 26.0,
                 "over_odds": -110, "under_odds": -110},
                # Outside the 168h lookback window — must be excluded.
                {"snapshot_ts_utc": stale, "game_date": "2025-04-01",
                 "player_id": 2544, "book": "fanduel", "market_key": "pts",
                 "stat_type": "points", "line_value": 24.0,
                 "over_odds": -110, "under_odds": -110},
                # Different stat — must not leak in.
                {"snapshot_ts_utc": recent, "game_date": "2025-04-15",
                 "player_id": 2544, "book": "fanduel", "market_key": "reb",
                 "stat_type": "rebounds", "line_value": 8.5,
                 "over_odds": -110, "under_odds": -110},
            ])

    def test_returns_snapshots_within_window_for_stat(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            self._seed(db_path)
            df = pc.fetch_line_movement_snapshots(
                db_path=db_path, player_id=2544, stat_type="points",
            )
        self.assertEqual(
            set(df.columns),
            {"snapshot_ts_utc", "book", "stat_type",
             "line_value", "over_odds", "under_odds"},
        )
        # 2 fanduel (recent + earlier) + 1 draftkings; stale + rebounds excluded.
        self.assertEqual(len(df), 3)
        self.assertEqual(set(df["book"]), {"fanduel", "draftkings"})
        self.assertNotIn(24.0, set(df["line_value"]))
        self.assertNotIn("rebounds", set(df["stat_type"]))

    def test_alias_stat_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            self._seed(db_path)
            # "3pm" canonicalizes to three_pointers_made (no snapshots) → empty.
            df = pc.fetch_line_movement_snapshots(
                db_path=db_path, player_id=2544, stat_type="3pm",
            )
        self.assertTrue(df.empty)
        # Empty frame still carries the figure's expected columns.
        self.assertIn("snapshot_ts_utc", df.columns)

    def test_empty_when_no_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path):
                pass
            df = pc.fetch_line_movement_snapshots(
                db_path=db_path, player_id=999, stat_type="points",
            )
        self.assertTrue(df.empty)


class Workstream3SplitsAndDecisionTests(unittest.TestCase):
    """Win/loss + starter/bench splits, calendar perf, Kelly, box/calendar figs."""

    def _rich_data(self):
        # 2024-01-01 = Monday, 2024-01-08 = Monday, 2024-02-06 = Tuesday.
        games = pd.DataFrame({
            "game_date": ["2024-01-01", "2024-01-08", "2024-02-06"],
            "result": ["W", "W", "L"],
            "minutes": [35.0, 35.0, 10.0],
        })
        return pc.PlayerChartData(
            player_id=1, player_name="Test Player", stat_type="points",
            games=games, values=np.array([10.0, 20.0, 30.0]),
            market_consensus_line=18.0,
        )

    def test_win_loss_split(self):
        wl = pc.compute_win_loss_split(self._rich_data())
        self.assertAlmostEqual(wl["W"]["mean"], 15.0)
        self.assertEqual(wl["W"]["n"], 2)
        self.assertAlmostEqual(wl["L"]["mean"], 30.0)
        self.assertEqual(wl["L"]["n"], 1)

    def test_starter_bench_split(self):
        sb = pc.compute_starter_bench_split(self._rich_data())
        self.assertAlmostEqual(sb["starter"]["mean"], 15.0)  # 35-min games
        self.assertEqual(sb["starter"]["n"], 2)
        self.assertAlmostEqual(sb["bench"]["mean"], 30.0)    # 10-min game
        self.assertEqual(sb["bench"]["n"], 1)

    def test_splits_empty_when_columns_missing(self):
        bare = pc.PlayerChartData(
            player_id=1, player_name="X", stat_type="points",
            games=pd.DataFrame({"game_date": ["2024-01-01"]}),
            values=np.array([10.0]),
        )
        self.assertEqual(pc.compute_win_loss_split(bare), {"W": None, "L": None})
        self.assertEqual(pc.compute_starter_bench_split(bare),
                         {"starter": None, "bench": None})

    def test_calendar_performance_matrix(self):
        cal = pc.compute_calendar_performance(self._rich_data())
        self.assertIsNotNone(cal)
        self.assertEqual(cal["month_labels"], ["Jan", "Feb"])
        self.assertEqual(cal["mean_matrix"].shape, (7, 2))
        # Mon (row 0) / Jan (col 0) → mean of 10 and 20.
        self.assertAlmostEqual(cal["mean_matrix"][0][0], 15.0)
        self.assertEqual(cal["count_matrix"][0][0], 2)
        # Tue (row 1) / Feb (col 1) → single game of 30.
        self.assertAlmostEqual(cal["mean_matrix"][1][1], 30.0)
        # Mon / Feb has no game → NaN.
        self.assertTrue(np.isnan(cal["mean_matrix"][0][1]))

    def test_calendar_none_without_dates(self):
        bare = pc.PlayerChartData(
            player_id=1, player_name="X", stat_type="points",
            games=pd.DataFrame(), values=np.array([], dtype=float),
        )
        self.assertIsNone(pc.compute_calendar_performance(bare))

    def test_kelly_stake_positive_edge(self):
        # +100 (dec 2.0), p=0.6 → f = (1·0.6 − 0.4)/1 = 0.2
        self.assertAlmostEqual(pc.kelly_stake(0.6, 100), 0.2, places=6)
        # Half-Kelly via fraction.
        self.assertAlmostEqual(pc.kelly_stake(0.6, 100, fraction=0.5), 0.1,
                               places=6)

    def test_kelly_stake_no_edge_is_zero(self):
        # p=0.5 at -110 has negative expectation → no stake.
        self.assertEqual(pc.kelly_stake(0.5, -110), 0.0)

    def test_kelly_stake_bad_inputs(self):
        self.assertEqual(pc.kelly_stake(None, -110), 0.0)
        self.assertEqual(pc.kelly_stake(0.6, 0), 0.0)      # odds 0 → invalid
        self.assertEqual(pc.kelly_stake(1.5, 100), 0.0)    # prob out of range

    def test_evaluate_custom_line_includes_kelly(self):
        data = pc.PlayerChartData(
            player_id=1, player_name="X", stat_type="points",
            games=pd.DataFrame({"game_date": [f"2024-01-{d:02d}"
                                              for d in range(1, 11)]}),
            values=np.arange(20, 30, dtype=float),
        )
        res = pc.evaluate_custom_line(data, 22.0, american_odds=-110)
        self.assertIn("kelly_over", res)
        self.assertIn("kelly_under", res)
        self.assertGreaterEqual(res["kelly_over"], 0.0)

    def test_box_quantile_figure(self):
        fig = pc.build_box_quantile_figure(self._rich_data())
        self.assertIsInstance(fig, Figure)

    def test_box_quantile_empty_safe(self):
        empty = pc.PlayerChartData(
            player_id=0, player_name="X", stat_type="points",
            games=pd.DataFrame(), values=np.array([], dtype=float),
        )
        self.assertIsInstance(pc.build_box_quantile_figure(empty), Figure)

    def test_calendar_heatmap_figure(self):
        fig = pc.build_calendar_heatmap_figure(self._rich_data())
        self.assertIsInstance(fig, Figure)

    def test_calendar_heatmap_empty_safe(self):
        empty = pc.PlayerChartData(
            player_id=0, player_name="X", stat_type="points",
            games=pd.DataFrame(), values=np.array([], dtype=float),
        )
        self.assertIsInstance(pc.build_calendar_heatmap_figure(empty), Figure)


class Workstream3Batch2Tests(unittest.TestCase):
    """vs-opponent split, correlation matrix, line-vs-actual ribbon, CLV proxy."""

    def _seed_games(self, db_path, pts=(16, 18, 20, 22, 16, 18, 20, 22, 18, 20)):
        with DatabaseManager(db_path=db_path) as db:
            db.upsert_active_players_reference([
                {"player_id": 2544, "player_name": "LeBron James",
                 "synced_at_utc": "2025-04-10T00:00:00Z"}])
            rows = []
            for i, p in enumerate(pts, start=1):
                rows.append({
                    "player_id": 2544, "game_id": f"g{i}",
                    "game_date": f"2025-04-{i:02d}", "season": "2024-25",
                    "matchup": "LAL vs. DEN" if i % 2 else "LAL @ BOS",
                    "home_away": "home" if i % 2 else "away", "result": "W",
                    "minutes": 34.0, "points": p, "rebounds": 8 + (i % 3),
                    "assists": 7 + (i % 2), "fgm": 8, "fga": 16, "fg3m": 2,
                    "fg3a": 6, "ftm": 4, "fta": 5, "oreb": 2, "dreb": 6,
                    "steals": 1, "blocks": 0, "turnovers": 3, "plus_minus": 5,
                })
            db.insert_game_logs(pd.DataFrame(rows))

    def test_opponent_split(self):
        games = pd.DataFrame({
            "matchup": ["LAL vs. DEN", "LAL @ DEN", "LAL vs. BOS"],
        })
        data = pc.PlayerChartData(
            player_id=1, player_name="P", stat_type="points",
            games=games, values=np.array([10.0, 20.0, 30.0]),
        )
        split = pc.compute_opponent_split(data)
        self.assertAlmostEqual(split["DEN"]["mean"], 15.0)
        self.assertEqual(split["DEN"]["n"], 2)
        self.assertAlmostEqual(split["BOS"]["mean"], 30.0)

    def test_correlation_matrix_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            self._seed_games(db_path)
            corr = pc.compute_correlation_matrix(
                db_path, 2544, ["points", "rebounds", "assists"],
            )
        self.assertEqual(corr.shape, (3, 3))
        for s in ("points", "rebounds", "assists"):
            self.assertAlmostEqual(float(corr.loc[s, s]), 1.0, places=6)

    def test_correlation_matrix_needs_two_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            self._seed_games(db_path)
            self.assertTrue(
                pc.compute_correlation_matrix(db_path, 2544, ["points"]).empty)

    def test_cumulative_line_vs_actual(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            self._seed_games(db_path)
            with DatabaseManager(db_path=db_path) as db:
                db.insert_betting_line_snapshots([
                    {"snapshot_ts_utc": "2025-04-01T12:00:00Z",
                     "game_date": "2025-04-01", "player_id": 2544,
                     "book": "fanduel", "market_key": "pts",
                     "stat_type": "points", "line_value": 15.0,
                     "over_odds": -110, "under_odds": -110},
                    {"snapshot_ts_utc": "2025-04-02T12:00:00Z",
                     "game_date": "2025-04-02", "player_id": 2544,
                     "book": "fanduel", "market_key": "pts",
                     "stat_type": "points", "line_value": 25.0,
                     "over_odds": -110, "under_odds": -110},
                ])
            df = pc.fetch_cumulative_line_vs_actual(db_path, 2544, "points")
        self.assertEqual(len(df), 2)
        # Game 1: actual 16 > line 15 → over; game 2: actual 18 < 25 → under.
        self.assertEqual(list(df["result"]), ["over", "under"])

    def test_cumulative_empty_without_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            self._seed_games(db_path)
            df = pc.fetch_cumulative_line_vs_actual(db_path, 2544, "points")
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns),
                         ["game_date", "actual", "line", "delta", "result"])

    def test_clv_proxy_by_book(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                db.insert_betting_line_snapshots([
                    {"snapshot_ts_utc": "2025-04-01T08:00:00Z",
                     "game_date": "2025-04-01", "player_id": 2544,
                     "book": "fanduel", "market_key": "pts",
                     "stat_type": "points", "line_value": 25.0,
                     "over_odds": -110, "under_odds": -110},
                    {"snapshot_ts_utc": "2025-04-01T18:00:00Z",
                     "game_date": "2025-04-01", "player_id": 2544,
                     "book": "fanduel", "market_key": "pts",
                     "stat_type": "points", "line_value": 27.0,
                     "over_odds": -110, "under_odds": -110},
                ])
            df = pc.fetch_clv_proxy_by_book(db_path, 2544, "points")
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["open_line"], 25.0)
        self.assertEqual(row["close_line"], 27.0)
        self.assertAlmostEqual(row["line_delta"], 2.0)
        self.assertEqual(int(row["n_snapshots"]), 2)


class MinutesAndModelEvTests(unittest.TestCase):
    def test_minutes_efficiency(self):
        games = pd.DataFrame({
            "game_date": ["2025-04-01", "2025-04-02", "2025-04-03"],
            "minutes": [30.0, 20.0, 0.0],
        })
        data = pc.PlayerChartData(
            player_id=1, player_name="P", stat_type="points",
            games=games, values=np.array([30.0, 10.0, 0.0]),
        )
        eff = pc.compute_minutes_efficiency(data)
        self.assertIsNotNone(eff)
        self.assertAlmostEqual(eff["per_minute"][0], 1.0)   # 30/30
        self.assertAlmostEqual(eff["per_minute"][1], 0.5)   # 10/20
        self.assertTrue(np.isnan(eff["per_minute"][2]))     # 0 minutes → NaN
        self.assertAlmostEqual(eff["avg_minutes"], 50.0 / 3, places=4)

    def test_minutes_efficiency_none_without_minutes(self):
        data = pc.PlayerChartData(
            player_id=1, player_name="P", stat_type="points",
            games=pd.DataFrame({"game_date": ["2025-04-01"]}),
            values=np.array([10.0]),
        )
        self.assertIsNone(pc.compute_minutes_efficiency(data))

    def test_model_vs_fitted_ev(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                db.conn.execute(
                    "INSERT INTO predictions (player_id, game_date, stat_type, "
                    "predicted_mean, predicted_std, prob_over, line_value, "
                    "book_odds, expected_value) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (2544, "2025-04-01", "points", 26.0, 5.0, 0.58, 24.5,
                     -110, 0.12),
                )
                db.conn.commit()
            data = pc.PlayerChartData(
                player_id=2544, player_name="LeBron James", stat_type="points",
                games=pd.DataFrame({"game_date": [f"2025-03-{d:02d}"
                                                  for d in range(1, 11)]}),
                values=np.arange(22, 32, dtype=float),
            )
            df = pc.fetch_model_vs_fitted_ev(db_path, data)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["line_value"], 24.5)
        self.assertAlmostEqual(df.iloc[0]["model_ev"], 0.12)
        self.assertIsNotNone(df.iloc[0]["fitted_ev"])

    def test_model_vs_fitted_ev_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path):
                pass
            data = pc.PlayerChartData(
                player_id=999, player_name="X", stat_type="points",
                games=pd.DataFrame(), values=np.array([], dtype=float),
            )
            df = pc.fetch_model_vs_fitted_ev(db_path, data)
        self.assertTrue(df.empty)


class FigureBuilderTests(unittest.TestCase):
    """Figure builders should return ``Figure`` objects without raising."""

    def _data(self, n=10, line=25.0):
        return pc.PlayerChartData(
            player_id=1,
            player_name="Test Player",
            stat_type="points",
            games=pd.DataFrame({
                "game_date": [f"2024-01-{d:02d}" for d in range(1, n + 1)],
            }),
            values=np.arange(20, 20 + n, dtype=float),
            book_lines=pd.DataFrame({
                "book": ["a", "b"],
                "line_value": [line, line + 1],
            }),
            market_consensus_line=line + 0.5,
        )

    def test_recent_games_figure(self):
        fig = pc.build_recent_games_figure(self._data())
        self.assertIsInstance(fig, Figure)
        # The figure should have the book-mean line in its legend.
        legend_labels = [
            t.get_text() for t in fig.axes[0].get_legend().get_texts()
        ] if fig.axes[0].get_legend() else []
        self.assertTrue(any("book mean" in lbl.lower() for lbl in legend_labels))

    def test_recent_games_figure_empty_data_is_safe(self):
        empty = pc.PlayerChartData(
            player_id=0, player_name="Nobody", stat_type="points",
            games=pd.DataFrame(), values=np.array([], dtype=float),
        )
        fig = pc.build_recent_games_figure(empty)
        self.assertIsInstance(fig, Figure)

    def test_distribution_figure(self):
        fig = pc.build_distribution_figure(self._data())
        self.assertIsInstance(fig, Figure)

    def test_hit_rate_figure(self):
        fig = pc.build_hit_rate_figure(self._data())
        self.assertIsInstance(fig, Figure)

    def test_rolling_ci_figure(self):
        fig = pc.build_rolling_ci_figure(self._data(), rolling_window=3,
                                         n_bootstrap=50)
        self.assertIsInstance(fig, Figure)
        # Legend should mention the CI band + rolling-mean line.
        legend_labels = (
            [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
            if fig.axes[0].get_legend() else []
        )
        self.assertTrue(any("CI" in lbl for lbl in legend_labels))
        self.assertTrue(any("rolling" in lbl.lower() for lbl in legend_labels))

    def test_trend_form_figure(self):
        fig = pc.build_trend_form_figure(self._data())
        self.assertIsInstance(fig, Figure)

    def test_trend_form_figure_empty_data_is_safe(self):
        empty = pc.PlayerChartData(
            player_id=0, player_name="Nobody", stat_type="points",
            games=pd.DataFrame(), values=np.array([], dtype=float),
        )
        fig = pc.build_trend_form_figure(empty)
        self.assertIsInstance(fig, Figure)


class StalenessSummaryTests(unittest.TestCase):
    def test_none_when_no_book_lines(self):
        data = pc.PlayerChartData(
            player_id=1, player_name="P", stat_type="points",
            games=pd.DataFrame(), values=np.array([], dtype=float),
        )
        self.assertIsNone(pc.book_lines_staleness_summary(data))

    def test_computes_median_hours(self):
        # Two books posted 1h ago and 5h ago → median 3h.
        now = pd.Timestamp.now(tz="UTC")
        data = pc.PlayerChartData(
            player_id=1, player_name="P", stat_type="points",
            games=pd.DataFrame(), values=np.array([], dtype=float),
            book_lines=pd.DataFrame({
                "book": ["a", "b"],
                "line_value": [25.5, 26.0],
                "game_date": [
                    (now - pd.Timedelta(hours=1)).isoformat(),
                    (now - pd.Timedelta(hours=5)).isoformat(),
                ],
            }),
        )
        out = pc.book_lines_staleness_summary(data)
        self.assertIsNotNone(out)
        self.assertEqual(out["n_books"], 2)
        # Allow ±0.05h slack so the test isn't wall-clock-flaky.
        self.assertAlmostEqual(out["hours_min"], 1.0, delta=0.05)
        self.assertAlmostEqual(out["hours_max"], 5.0, delta=0.05)
        self.assertAlmostEqual(out["hours_median"], 3.0, delta=0.05)

    def test_format_staleness_human(self):
        self.assertEqual(pc._format_staleness(0.5), "30 m ago")
        self.assertEqual(pc._format_staleness(2.4), "2.4 h ago")
        self.assertEqual(pc._format_staleness(72.0), "3.0 d ago")


if __name__ == "__main__":
    unittest.main()
