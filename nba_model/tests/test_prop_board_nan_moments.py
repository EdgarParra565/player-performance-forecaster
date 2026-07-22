"""Regression tests for NaN μ/σ when a NULL stat value lands in the window.

Root cause: ``add_rolling_stats`` uses ``min_periods == window``, so a single
NULL box score inside the trailing window drops the valid-observation count
below the threshold and pandas returns NaN for that stat's rolling mean/std --
even though ``rolling_mean_minutes`` (the dropna gate in
``build_history_from_games``) is finite. ``_repair_nan_stat_moments`` recomputes
those NaN aggregates over the surviving games (available-case), skipping only
when too few valid games remain.

These tests pin:
  * a single NULL in the window -> finite μ/σ over the surviving games,
  * clean windows -> byte-for-byte unchanged (no drift),
  * a NULL in one stat doesn't disturb another stat's μ,
  * newest-first (get_player_games) and oldest-first input agree,
  * a mostly-NULL window -> μ/σ stays NaN so the row is skipped downstream.
"""

import math
import unittest

import numpy as np
import pandas as pd

from nba_model.model.prop_board import (
    build_history_from_games,
    project_stat_moments,
)

WINDOW = 10
# 12 games so the trailing-10 window is full; points vary, rebounds constant 8.
_POINTS = [16, 18, 20, 22, 16, 18, 20, 22, 18, 20, 24, 26]


def _games(rebounds_per_game, *, order="asc"):
    """Build a raw game_logs-style frame (oldest-first by default)."""
    rows = []
    for i, (p, r) in enumerate(zip(_POINTS, rebounds_per_game), start=1):
        rows.append({
            "player_id": 1, "game_id": f"g{i}",
            "game_date": f"2025-04-{i:02d}", "minutes": 34.0,
            "points": float(p), "rebounds": r, "assists": 7.0,
        })
    df = pd.DataFrame(rows)
    if order == "desc":  # mimic db.get_player_games (ORDER BY game_date DESC)
        df = df.iloc[::-1].reset_index(drop=True)
    return df


class RepairNanMomentsTests(unittest.TestCase):
    def test_single_null_in_window_recomputes_finite_moments(self):
        # Newest game has NULL rebounds; other 11 are 8.0 -> trailing-10 window
        # has 9 valid 8.0s -> μ=8, σ=0 (floored downstream), all finite.
        rebounds = [8.0] * 11 + [None]  # game 12 (newest) is NULL
        latest = build_history_from_games(_games(rebounds), rolling_window=WINDOW)
        self.assertIsNotNone(latest)
        moments = project_stat_moments(latest, "rebounds")
        self.assertTrue(math.isfinite(moments["mu"]))
        self.assertTrue(math.isfinite(moments["sigma"]))
        self.assertAlmostEqual(moments["mu"], 8.0, places=6)

    def test_null_does_not_disturb_other_stats(self):
        # A NULL rebounds must not corrupt the points projection.
        rebounds = [8.0] * 11 + [None]
        latest = build_history_from_games(_games(rebounds), rolling_window=WINDOW)
        pts = project_stat_moments(latest, "points")
        # Trailing-10 points mean (games 3..12): mean of _POINTS[2:].
        expected = float(np.mean(_POINTS[-WINDOW:]))
        self.assertAlmostEqual(pts["mu"], expected, places=6)

    def test_clean_history_moments_unchanged(self):
        # No NULLs: μ/σ must equal the plain trailing-window statistics (the
        # recompute path is inert, so there is no drift for clean windows).
        rebounds = [7.0, 9.0, 8.0, 6.0, 10.0, 8.0, 7.0, 9.0, 8.0, 6.0, 10.0, 9.0]
        latest = build_history_from_games(_games(rebounds), rolling_window=WINDOW)
        moments = project_stat_moments(latest, "rebounds")
        window_vals = pd.Series(rebounds[-WINDOW:])
        self.assertAlmostEqual(moments["mu"], float(window_vals.mean()), places=9)
        self.assertAlmostEqual(
            moments["sigma"], float(window_vals.std(ddof=1)), places=9)

    def test_desc_and_asc_input_agree(self):
        # Scanner full mode feeds DESC (get_player_games); hourly/CLI can feed
        # ASC. Both must land on identical repaired μ/σ.
        rebounds = [8.0] * 11 + [None]
        asc = project_stat_moments(
            build_history_from_games(_games(rebounds, order="asc"), WINDOW),
            "rebounds")
        desc = project_stat_moments(
            build_history_from_games(_games(rebounds, order="desc"), WINDOW),
            "rebounds")
        self.assertAlmostEqual(asc["mu"], desc["mu"], places=9)
        self.assertAlmostEqual(asc["sigma"], desc["sigma"], places=9)

    def test_too_few_valid_in_window_stays_nan(self):
        # 6 of the newest 10 rebounds are NULL -> 4 valid < floor(0.5*10)=5 ->
        # μ/σ stays NaN so the caller drops the row (no fabricated projection).
        rebounds = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0,  # games 1..6 valid
                    None, None, None, None, None, None]  # games 7..12 NULL
        latest = build_history_from_games(_games(rebounds), rolling_window=WINDOW)
        moments = project_stat_moments(latest, "rebounds")
        self.assertTrue(math.isnan(moments["mu"]))


if __name__ == "__main__":
    unittest.main()
