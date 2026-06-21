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

import pandas as pd

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


def _seed_team_priors_and_games(db, *, home_tt, away_tt, pace, lal_points):
    """Seed a DEN@LAL prior + recent LAL games (for the per-team baseline)."""
    db.upsert_team_priors([{
        "away_team": "DEN", "home_team": "LAL",
        "computed_at_utc": "2025-04-10T00:00:00Z",
        "consensus_total": home_tt + away_tt,
        "home_spread": -3.0, "away_spread": 3.0,
        "home_team_total": home_tt, "away_team_total": away_tt,
        "home_win_prob_devig": 0.6, "away_win_prob_devig": 0.4,
        "pace_factor": pace, "n_books": 5,
        "latest_observed_at": "2025-04-10T00:00:00Z",
    }])
    db.insert_games([
        {
            "game_id": f"g{i}", "season": "2024-25",
            "season_type": "Regular Season",
            "game_date": f"2025-04-0{i}",
            "team_id": 1610612747, "team_abbrev": "LAL",
            "matchup": "LAL vs. DEN", "home_away": "home",
            "pts": lal_points, "opp_pts": 100,
        }
        for i in range(1, 6)
    ])


class TeamRecentAvgPointsTests(unittest.TestCase):
    def test_per_team_points_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                db.insert_games([
                    {"game_id": "g1", "season": "2024-25",
                     "season_type": "Regular Season", "game_date": "2025-04-01",
                     "team_id": 1, "team_abbrev": "LAL", "matchup": "LAL vs. DEN",
                     "home_away": "home", "pts": 120, "opp_pts": 110},
                    {"game_id": "g2", "season": "2024-25",
                     "season_type": "Regular Season", "game_date": "2025-04-03",
                     "team_id": 1, "team_abbrev": "LAL", "matchup": "LAL @ DEN",
                     "home_away": "away", "pts": 100, "opp_pts": 105},
                ])
                # avg(pts) = (120+100)/2 = 110 — per-team, NOT the game total.
                self.assertAlmostEqual(
                    db.get_team_recent_avg_points("LAL"), 110.0, places=3)


class TeamPriorInputsTests(unittest.TestCase):
    def test_home_orientation_picks_home_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                _seed_team_priors_and_games(
                    db, home_tt=118.0, away_tt=112.0, pace=1.05, lal_points=110)
                inputs = db.get_team_prior_inputs("LAL", "DEN")
        self.assertAlmostEqual(inputs["implied_team_total"], 118.0)  # LAL home
        self.assertAlmostEqual(inputs["pace_factor"], 1.05)
        self.assertAlmostEqual(inputs["team_recent_avg_total"], 110.0)

    def test_away_orientation_picks_away_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                _seed_team_priors_and_games(
                    db, home_tt=118.0, away_tt=112.0, pace=1.05, lal_points=110)
                # DEN is the away side of the stored DEN@LAL matchup.
                inputs = db.get_team_prior_inputs("DEN", "LAL")
        self.assertAlmostEqual(inputs["implied_team_total"], 112.0)

    def test_no_prior_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                self.assertEqual(db.get_team_prior_inputs("LAL", "DEN"), {})
                self.assertEqual(db.get_team_prior_inputs("", "DEN"), {})

    def test_inputs_map_covers_both_sides(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                _seed_team_priors_and_games(
                    db, home_tt=118.0, away_tt=112.0, pace=1.05, lal_points=110)
                m = db.get_team_prior_inputs_map()
        self.assertIn("LAL", m)
        self.assertIn("DEN", m)
        self.assertAlmostEqual(m["LAL"]["implied_team_total"], 118.0)
        self.assertAlmostEqual(m["DEN"]["implied_team_total"], 112.0)


class RunSinglePropTeamPriorTests(unittest.TestCase):
    """Integration: team prior actually shifts run_single_prop's projection."""

    def _seed_lebron(self, db):
        # LeBron resolves to player_id 2544 via the bundled nba_api static list.
        rows = []
        for i in range(1, 13):
            rows.append({
                "player_id": 2544, "game_id": f"lj{i}",
                "game_date": f"2025-03-{i:02d}", "season": "2024-25",
                "matchup": "LAL vs. DEN", "home_away": "home", "result": "W",
                "minutes": 35.0, "points": 28, "rebounds": 8, "assists": 7,
                "fgm": 10, "fga": 20, "fg3m": 2, "fg3a": 6, "ftm": 6, "fta": 7,
                "oreb": 1, "dreb": 7, "steals": 1, "blocks": 1, "turnovers": 3,
                "plus_minus": 6,
            })
        db.insert_game_logs(pd.DataFrame(rows))

    def test_prior_shifts_mu_up(self):
        from nba_model.run_model import run_single_prop
        # DataLoader keeps its sqlite connection open, so on Windows the temp
        # dir can't be unlinked until GC — tolerate that in teardown.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                self._seed_lebron(db)
                _seed_team_priors_and_games(
                    db, home_tt=130.0, away_tt=120.0, pace=1.30, lal_points=110)

            result = run_single_prop(
                player_name="LeBron James", line=27.5, rolling_window=3,
                american_odds=-110, opp_def_rating=113.0, vegas_spread=3.0,
                n_games=12, player_team="LAL", opponent_team="DEN",
                db_path=db_path,
            )
        self.assertTrue(result["team_prior_applied"])
        # High pace (1.30) + implied 130 vs baseline 110 → mu nudged UP.
        self.assertGreater(result["mu"], result["mu_pre_prior"])

    def test_no_team_context_leaves_mu_unchanged(self):
        from nba_model.run_model import run_single_prop
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                self._seed_lebron(db)
                _seed_team_priors_and_games(
                    db, home_tt=130.0, away_tt=120.0, pace=1.30, lal_points=110)

            result = run_single_prop(
                player_name="LeBron James", line=27.5, rolling_window=3,
                american_odds=-110, opp_def_rating=113.0, vegas_spread=3.0,
                n_games=12, db_path=db_path,  # no player_team/opponent_team
            )
        self.assertFalse(result["team_prior_applied"])
        self.assertAlmostEqual(result["mu"], result["mu_pre_prior"], places=6)


if __name__ == "__main__":
    unittest.main()
