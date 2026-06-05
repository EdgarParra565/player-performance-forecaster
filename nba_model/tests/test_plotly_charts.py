"""Smoke + behavior tests for the upgraded Plotly chart builders.

We don't try to validate the visual output (that's what manual review is
for), but we do verify that:

  - Every public builder returns a non-empty Plotly Figure for a populated
    player + stat, with the expected trace count growing with the inputs.
  - The "line ladder" view is selectable via ``view_mode='ladder'`` and
    produces at least one trace when book lines exist.
  - The fitted-overlay path honors ``simulation.SUPPORTED_DISTRIBUTIONS``
    so the UI selector doesn't fall behind the model vocabulary.
  - Empty inputs degrade gracefully (no exceptions; an explanatory
    annotation trace instead of a crash).
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nba_model.model.simulation import SUPPORTED_DISTRIBUTIONS
from nba_model.visualization import plotly_charts as plc
from nba_model.visualization import player_charts as pc


def _build_synthetic_data(
    *, n_games: int = 40, n_books: int = 6, stat_type: str = "points",
) -> pc.PlayerChartData:
    """Build a populated ``PlayerChartData`` without touching the DB.

    Using random-but-deterministic values lets the test verify the builders
    work for the "6+ books, lots of history" case without depending on the
    bundled SQLite fixtures.
    """
    rng = np.random.default_rng(seed=1234)
    values = rng.normal(loc=25.0, scale=6.0, size=n_games).clip(min=0).round(0)
    dates = pd.date_range("2025-01-01", periods=n_games, freq="3D")
    games = pd.DataFrame({
        "game_date": dates,
        "matchup": ["LAL vs DEN"] * n_games,
        "home_away": ["home" if i % 2 == 0 else "away" for i in range(n_games)],
    })

    book_names = [
        "fanduel", "draftkings", "betmgm", "caesars", "betrivers", "fanatics",
        "bovada", "prizepicks", "underdog",
    ][:n_books]
    book_rows = []
    for i, b in enumerate(book_names):
        book_rows.append({
            "book": b,
            "line_value": 24.5 + 0.5 * (i - n_books / 2),
            "over_odds": -110 if i % 2 == 0 else -120,
            "under_odds": -110 if i % 2 == 0 else +100,
            "game_date": "2025-04-01",
        })
    book_lines = pd.DataFrame(book_rows)

    median_line = float(np.median(book_lines["line_value"]))
    return pc.PlayerChartData(
        player_id=2544,
        player_name="Test Player",
        stat_type=stat_type,
        values=values,
        games=games,
        book_lines=book_lines,
        market_consensus_line=median_line,
    )


class DistributionFigureTests(unittest.TestCase):
    def test_returns_nonempty_figure(self):
        data = _build_synthetic_data()
        fig = plc.build_distribution_figure(data, distributions=("normal",))
        # Histogram + at least the normal-fit overlay = at least 2 traces.
        self.assertGreaterEqual(len(fig.data), 2)

    def test_six_books_render_per_book_markers(self):
        data = _build_synthetic_data(n_books=6)
        fig = plc.build_distribution_figure(data, distributions=("normal",))
        # Each book contributes a vertical line + a top-rail triangle, so
        # we expect at minimum 12 book-related traces in addition to the
        # histogram + the normal fit.
        self.assertGreaterEqual(len(fig.data), 12 + 2)

    def test_handles_every_supported_distribution(self):
        data = _build_synthetic_data()
        # Should not raise for any name in SUPPORTED_DISTRIBUTIONS.
        fig = plc.build_distribution_figure(
            data, distributions=tuple(SUPPORTED_DISTRIBUTIONS) + ("negative_binomial",),
        )
        self.assertGreater(len(fig.data), 1)

    def test_ladder_view_returns_figure(self):
        data = _build_synthetic_data(n_books=6)
        fig = plc.build_distribution_figure(data, view_mode="ladder")
        # The ladder is a single scatter trace + per-row connector shapes.
        self.assertGreaterEqual(len(fig.data), 1)

    def test_empty_data_renders_placeholder_not_crash(self):
        empty = pc.PlayerChartData(
            player_id=0, player_name="Nobody", stat_type="points",
            values=np.array([], dtype=float),
            games=pd.DataFrame(),
            book_lines=pd.DataFrame(),
            market_consensus_line=None,
        )
        fig = plc.build_distribution_figure(empty)
        self.assertEqual(len(fig.data), 0)
        # The "No game data" annotation should be present.
        self.assertEqual(len(fig.layout.annotations), 1)


class RecentGamesFigureTests(unittest.TestCase):
    def test_bar_plus_rolling_mean_traces(self):
        data = _build_synthetic_data()
        fig = plc.build_recent_games_figure(data, rolling_window=5)
        # Bar + rolling-mean overlay = 2 traces (book-mean is a layout shape,
        # not a trace).
        self.assertEqual(len(fig.data), 2)

    def test_empty_returns_placeholder(self):
        empty = pc.PlayerChartData(
            player_id=0, player_name="Nobody", stat_type="points",
            values=np.array([], dtype=float),
            games=pd.DataFrame(),
            book_lines=pd.DataFrame(),
            market_consensus_line=None,
        )
        fig = plc.build_recent_games_figure(empty)
        self.assertEqual(len(fig.data), 0)


class HitRateFigureTests(unittest.TestCase):
    def test_one_bar_per_book(self):
        data = _build_synthetic_data(n_books=4)
        fig = plc.build_hit_rate_figure(data)
        self.assertEqual(len(fig.data), 1)
        # Each bar's `y` carries len == n_books labels.
        self.assertEqual(len(fig.data[0].y), 4)


class SplitsFigureTests(unittest.TestCase):
    def test_two_subplot_traces(self):
        data = _build_synthetic_data()
        fig = plc.build_splits_figure(data)
        # 1 trace per subplot (home/away + rest-days).
        self.assertGreaterEqual(len(fig.data), 1)


class MultiPlayerOverlayTests(unittest.TestCase):
    def test_two_players_render_overlay(self):
        a = _build_synthetic_data()
        b = _build_synthetic_data()
        fig = plc.build_multi_player_distribution_figure([a, b])
        # 2 histograms + 2 fitted-normal overlays.
        self.assertGreaterEqual(len(fig.data), 4)


class LineMovementFigureTests(unittest.TestCase):
    def test_one_trace_per_book(self):
        snapshots = pd.DataFrame([
            {"snapshot_ts_utc": "2025-04-01T00:00:00Z", "book": "fanduel",
             "stat_type": "points", "line_value": 25.5},
            {"snapshot_ts_utc": "2025-04-01T06:00:00Z", "book": "fanduel",
             "stat_type": "points", "line_value": 25.0},
            {"snapshot_ts_utc": "2025-04-01T00:00:00Z", "book": "draftkings",
             "stat_type": "points", "line_value": 25.5},
        ])
        fig = plc.build_line_movement_figure(snapshots, stat_type="points")
        self.assertEqual(len(fig.data), 2)

    def test_empty_snapshots_renders_placeholder(self):
        fig = plc.build_line_movement_figure(pd.DataFrame(), stat_type="points")
        self.assertEqual(len(fig.data), 0)
        self.assertEqual(len(fig.layout.annotations), 1)


if __name__ == "__main__":
    unittest.main()
