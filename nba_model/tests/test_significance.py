import unittest

from nba_model.evaluation.significance import (
    breakeven_probability,
    wilson_interval,
    win_rate_significance_summary,
    z_test_proportion,
)


class SignificanceTests(unittest.TestCase):
    def test_breakeven_probability_minus_110(self):
        p = breakeven_probability(-110)
        self.assertAlmostEqual(p, 0.5238095, places=5)

    def test_wilson_interval_bounds(self):
        interval = wilson_interval(successes=55, trials=100, confidence=0.95)
        self.assertLess(interval["lower"], 0.55)
        self.assertGreater(interval["upper"], 0.55)
        self.assertGreaterEqual(interval["lower"], 0.0)
        self.assertLessEqual(interval["upper"], 1.0)

    def test_z_test_two_sided(self):
        result = z_test_proportion(successes=60, trials=100, p0=0.5)
        self.assertGreater(result["z_score"], 0)
        self.assertLess(result["p_value"], 0.1)

    def test_summary_fields(self):
        summary = win_rate_significance_summary(wins=60, bets=100)
        self.assertIn("win_rate_ci_lower", summary)
        self.assertIn("win_rate_ci_upper", summary)
        self.assertIn("p_value_vs_breakeven", summary)
        self.assertIn("significant_at_5pct", summary)


if __name__ == "__main__":
    unittest.main()
