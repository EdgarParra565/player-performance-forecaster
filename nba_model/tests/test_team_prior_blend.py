"""Tests for ``simulation.blend_team_prior`` + the DB-side avg-total helper.

The blender is the "next step" called out in 2026-05-11 notes — it lets the
player projection pull from cross-book consensus without abandoning the
player's own historical baseline.  Tests pin three things:

1. No prior inputs → mu/sigma unchanged (callable from anywhere safely).
2. Pace prior scales mu/sigma proportionally with the supplied ``alpha``.
3. Team-total prior scales by the ratio of implied vs. recent team average.
4. Output is clamped to [-50 %, +50 %] of the input so extreme priors can't
   blow the projection up.
"""

import tempfile
import unittest
from pathlib import Path

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model.simulation import blend_team_prior


class BlendTeamPriorTests(unittest.TestCase):
    def test_no_prior_returns_inputs(self):
        mu, sigma = blend_team_prior(25.0, 6.0)
        self.assertEqual(mu, 25.0)
        self.assertEqual(sigma, 6.0)

    def test_pace_factor_above_baseline_scales_up(self):
        # pace_factor=1.10 → 10 % above league avg → factor = 1 + 0.3 * 0.10 = 1.03
        mu, sigma = blend_team_prior(25.0, 6.0, pace_factor=1.10, alpha=0.3)
        self.assertAlmostEqual(mu, 25.0 * 1.03, places=3)
        self.assertAlmostEqual(sigma, 6.0 * 1.03, places=3)

    def test_pace_factor_below_baseline_scales_down(self):
        mu, sigma = blend_team_prior(25.0, 6.0, pace_factor=0.90, alpha=0.3)
        # factor = 1 + 0.3 * (0.9 - 1) = 0.97
        self.assertAlmostEqual(mu, 25.0 * 0.97, places=3)
        self.assertAlmostEqual(sigma, 6.0 * 0.97, places=3)

    def test_implied_team_total_relative_to_baseline(self):
        # Team usually averages 110; tonight implied 115 → ratio 1.045
        # factor = 1 + 0.3 * (115/110 - 1) = 1.01364
        mu, _ = blend_team_prior(
            25.0, 6.0,
            implied_team_total=115.0,
            team_recent_avg_total=110.0,
            alpha=0.3,
        )
        self.assertAlmostEqual(mu, 25.0 * (1 + 0.3 * (115.0 / 110.0 - 1.0)),
                               places=4)

    def test_combined_priors_multiply(self):
        # Both signals applied: pace 1.05 + team total ratio 1.04
        mu, _ = blend_team_prior(
            25.0, 6.0,
            pace_factor=1.05,
            implied_team_total=110.0,
            team_recent_avg_total=105.0,
            alpha=0.3,
        )
        expected_factor = (
            (1 + 0.3 * 0.05)
            * (1 + 0.3 * (110.0 / 105.0 - 1.0))
        )
        self.assertAlmostEqual(mu, 25.0 * expected_factor, places=4)

    def test_clamped_to_half_range(self):
        # Absurd pace input should not blow mu up beyond 1.5x or below 0.5x.
        mu_hi, _ = blend_team_prior(25.0, 6.0, pace_factor=100.0, alpha=1.0)
        self.assertAlmostEqual(mu_hi, 25.0 * 1.5, places=3)
        mu_lo, _ = blend_team_prior(25.0, 6.0, pace_factor=0.001, alpha=1.0)
        self.assertAlmostEqual(mu_lo, 25.0 * 0.5, places=3)

    def test_alpha_zero_disables_blend(self):
        mu, sigma = blend_team_prior(
            25.0, 6.0, pace_factor=1.30,
            implied_team_total=200.0, team_recent_avg_total=100.0,
            alpha=0.0,
        )
        self.assertEqual(mu, 25.0)
        self.assertEqual(sigma, 6.0)

    def test_bad_inputs_silently_ignored(self):
        # ``None`` and string nonsense should leave mu/sigma untouched.
        mu, sigma = blend_team_prior(
            25.0, 6.0, pace_factor="bad",  # type: ignore[arg-type]
            implied_team_total=None, team_recent_avg_total=110.0,
        )
        self.assertEqual(mu, 25.0)
        self.assertEqual(sigma, 6.0)


class TeamRecentAvgTotalTests(unittest.TestCase):
    def test_average_across_recent_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                db.insert_games([
                    {
                        "game_id": "g1", "season": "2024-25",
                        "season_type": "Regular Season",
                        "game_date": "2025-04-01",
                        "team_id": 1, "team_abbrev": "LAL",
                        "matchup": "LAL vs. DEN", "home_away": "home",
                        "pts": 120, "opp_pts": 110,  # total 230
                    },
                    {
                        "game_id": "g2", "season": "2024-25",
                        "season_type": "Regular Season",
                        "game_date": "2025-04-03",
                        "team_id": 1, "team_abbrev": "LAL",
                        "matchup": "LAL @ DEN", "home_away": "away",
                        "pts": 100, "opp_pts": 105,  # total 205
                    },
                ])
                avg = db.get_team_recent_avg_total("LAL", n_games=5)
        # Mean of (230, 205) = 217.5
        self.assertAlmostEqual(avg, 217.5, places=3)

    def test_missing_team_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                self.assertIsNone(db.get_team_recent_avg_total("LAL"))


if __name__ == "__main__":
    unittest.main()
