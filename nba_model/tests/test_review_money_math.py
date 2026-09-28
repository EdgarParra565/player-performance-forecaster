"""Regression tests for money-math fixes from the 2026-09-25 code review.

Each class pins one finding in review_findings_data.md:
  * defense adjustment direction (higher DRtg = worse defense = higher mu),
  * binomial fit no longer collapses the mean when var >= mean,
  * push mass on integer lines for discrete distributions (P(under) and EV),
  * invalid American odds (|odds| < 100) cannot fabricate a two-way arb,
  * "significant" means significantly BETTER than breakeven,
  * backtest pushes are neither wins nor losses.
"""

import math
import unittest

import numpy as np
import pandas as pd
from scipy.stats import binom, poisson

from nba_model.evaluation.backtest import Backtester
from nba_model.evaluation.significance import win_rate_significance_summary
from nba_model.model import cross_book_arb as cba
from nba_model.model import edge_scanner as es
from nba_model.model.defense_adjustment import adjust_mu_for_defense
from nba_model.model.odds import expected_value
from nba_model.model.probability import (
    prob_over_distribution,
    prob_push_distribution,
)
from nba_model.model.simulation import _draw_samples
from nba_model.tests.test_edge_scanner_full import EdgeScannerFullTestBase, _card
from nba_model.data.database.db_manager import DatabaseManager
from nba_model.visualization import player_charts as pc


class DefenseDirectionTests(unittest.TestCase):
    def test_bad_defense_raises_and_good_defense_lowers_mu(self):
        self.assertGreater(adjust_mu_for_defense(20.0, 121.0), 20.0)
        self.assertLess(adjust_mu_for_defense(20.0, 106.3), 20.0)  # OKC-grade D
        self.assertEqual(adjust_mu_for_defense(20.0, 113.0), 20.0)


class BinomialFitTests(unittest.TestCase):
    def test_overdispersed_input_falls_back_instead_of_collapsing(self):
        # var 9 > mean 8: binomial moments are impossible. Old code gave ~6e-08.
        p = prob_over_distribution(7.5, 8.0, 3.0, "binomial")
        nb = prob_over_distribution(7.5, 8.0, 3.0, "negative_binomial")
        self.assertAlmostEqual(p, nb, places=12)
        self.assertGreater(p, 0.3)

    def test_underdispersed_input_uses_exact_binomial(self):
        # mean 8, var 4 -> p = 0.5, n = 16.
        p = prob_over_distribution(7.5, 8.0, 2.0, "binomial")
        self.assertAlmostEqual(p, float(binom.sf(7, 16, 0.5)), places=12)

    def test_sampler_mean_preserved(self):
        draws = _draw_samples(8.0, 3.0, 200_000, "binomial")
        self.assertAlmostEqual(float(np.mean(draws)), 8.0, delta=0.1)


class PushMassTests(unittest.TestCase):
    def test_poisson_integer_line_push_is_pmf(self):
        push = prob_push_distribution(7.0, 7.0, math.sqrt(7.0), "poisson")
        self.assertAlmostEqual(push, float(poisson.pmf(7, 7.0)), places=12)
        over = prob_over_distribution(7.0, 7.0, math.sqrt(7.0), "poisson")
        under_true = float(poisson.cdf(6, 7.0))
        self.assertAlmostEqual(1.0 - over - push, under_true, places=12)

    def test_no_push_for_half_lines_or_continuous(self):
        self.assertEqual(prob_push_distribution(7.5, 7.0, 2.6, "poisson"), 0.0)
        self.assertEqual(prob_push_distribution(7.0, 7.0, 2.6, "normal"), 0.0)

    def test_push_aware_ev(self):
        # p_win .4, push .2 -> loss .4 at +100: EV = .4 - .4 = 0 (was -0.2).
        self.assertAlmostEqual(expected_value(0.4, 100, push_prob=0.2), 0.0)
        self.assertAlmostEqual(pc.expected_value(0.4, 100, push_prob=0.2), 0.0)
        self.assertAlmostEqual(expected_value(0.4, 100), -0.2)  # default unchanged


class ScannerPushTests(EdgeScannerFullTestBase):
    def setUp(self):
        super().setUp()
        with DatabaseManager(db_path=self.db_path) as db:
            db.insert_web_prop_cards([
                _card("PrizePicks", "LeBron James", "rebounds", 8.0, "over",
                      self.recent, 99),
            ])

    def test_full_mode_integer_rebounds_line_excludes_push(self):
        scored = self._scored("full")
        row = scored[(scored["stat_type"] == "rebounds")
                     & (scored["book_line"] == 8.0)].iloc[0]
        self.assertEqual(row["distribution"], "poisson")
        # Every game had 8 rebounds, so a push at 8 carries real mass.
        self.assertLess(row["p_over"] + row["p_under"], 0.99)


class InvalidOddsArbTests(unittest.TestCase):
    def test_zero_or_sub_100_odds_never_flag_arb(self):
        for bad in (0, -50, 50):
            self.assertIsNone(cba._safe_implied_prob(bad))
        self.assertAlmostEqual(cba._safe_implied_prob(-110), 110 / 210)
        lines = pd.DataFrame([
            {"player_id": 1, "player_name": "X", "game_date": "2026-01-01",
             "stat_type": "points", "book": "b1", "line_value": 20.5,
             "over_odds": 0, "under_odds": -110},
            {"player_id": 1, "player_name": "X", "game_date": "2026-01-01",
             "stat_type": "points", "book": "b2", "line_value": 20.5,
             "over_odds": -110, "under_odds": -110},
        ])
        self.assertTrue(cba.detect_two_way_arb(lines).empty)


class SignificanceDirectionTests(unittest.TestCase):
    def test_significant_loser_not_flagged(self):
        res = win_rate_significance_summary(40, 100)
        self.assertLess(res["z_score_vs_breakeven"], 0)
        self.assertFalse(res["significant_at_5pct"])

    def test_significant_winner_flagged(self):
        self.assertTrue(win_rate_significance_summary(70, 100)["significant_at_5pct"])


class BacktestPushTests(unittest.TestCase):
    def test_pushes_are_not_losses(self):
        bt = Backtester.__new__(Backtester)
        bt.distribution = "normal"
        bt.results = [
            {"bet_recommendation": "over", "outcome": "over", "correct": True,
             "profit": 100.0, "actual_value": 25, "line": 20, "prob_over": 0.7},
            {"bet_recommendation": "over", "outcome": "push", "correct": False,
             "profit": 0.0, "actual_value": 20, "line": 20, "prob_over": 0.7},
            {"bet_recommendation": "under", "outcome": "push", "correct": False,
             "profit": 0.0, "actual_value": 20, "line": 20, "prob_over": 0.3},
        ]
        m = bt._calculate_metrics()
        self.assertEqual(int(m["wins"]), 1)
        self.assertEqual(int(m["losses"]), 0)
        self.assertEqual(m["win_rate"], 1.0)
        self.assertEqual(int(m["pushes"]), 2)
        # Brier over the one resolved row only: (0.7 - 1)^2.
        self.assertAlmostEqual(m["brier_score"], 0.09)


if __name__ == "__main__":
    unittest.main()
