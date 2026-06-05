"""End-to-end tests asserting every SUPPORTED_DISTRIBUTIONS family flows
through run_single_prop and the board-level distribution selection without
errors and yields a probability in [0, 1]."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from nba_model.model.simulation import (
    DEFAULT_DISTRIBUTION_BY_STAT,
    SUPPORTED_DISTRIBUTIONS,
    get_default_distribution,
    monte_carlo_over,
)


def _synthetic_player_log(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "game_date": dates,
            "points": rng.normal(24, 6, n).clip(0).round(),
            "assists": rng.normal(6, 2, n).clip(0).round(),
            "rebounds": rng.normal(7, 2.5, n).clip(0).round(),
            "minutes": rng.normal(33, 3, n).clip(15, 45),
        }
    )


class RunSinglePropDistributionEndToEndTests(unittest.TestCase):
    """run_single_prop must successfully evaluate every supported family."""

    @patch("nba_model.run_model.DataLoader")
    def test_every_supported_distribution_runs(self, mock_loader_cls):
        from nba_model.run_model import run_single_prop

        loader_instance = MagicMock()
        loader_instance.load_player_data.return_value = _synthetic_player_log()
        mock_loader_cls.return_value = loader_instance

        for distribution in SUPPORTED_DISTRIBUTIONS:
            result = run_single_prop(
                player_name="LeBron James",
                line=24.5,
                rolling_window=10,
                american_odds=-110,
                opp_def_rating=112.0,
                vegas_spread=5.0,
                distribution=distribution,
                n_games=80,
            )
            self.assertEqual(result["distribution"], distribution)
            prob = result["prob_over"]
            self.assertGreaterEqual(prob, 0.0, f"{distribution} returned p={prob}")
            self.assertLessEqual(prob, 1.0, f"{distribution} returned p={prob}")


class BoardDistributionDefaultsTests(unittest.TestCase):
    """get_default_distribution must be wired and produce valid probs for the
    four primary stats."""

    def test_defaults_cover_primary_stats(self):
        for stat in ("points", "assists", "rebounds", "pra"):
            self.assertIn(stat, DEFAULT_DISTRIBUTION_BY_STAT)
            self.assertIn(DEFAULT_DISTRIBUTION_BY_STAT[stat], SUPPORTED_DISTRIBUTIONS)

    def test_default_distribution_runs_for_each_primary_stat(self):
        # mu/sigma typical of each stat — verifies the chosen default produces
        # a valid probability without raising.
        scenarios = {
            "points": (24.0, 6.0, 25.5),
            "assists": (6.0, 2.0, 6.5),
            "rebounds": (7.5, 2.5, 8.5),
            "pra": (38.0, 8.0, 39.5),
        }
        for stat, (mu, sigma, line) in scenarios.items():
            dist = get_default_distribution(stat)
            prob = monte_carlo_over(
                mu=mu, sigma=sigma, line=line, n=2000,
                distribution=dist, sample_size=10,
            )
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)


if __name__ == "__main__":
    unittest.main()
