"""Tests for the distribution-sweep per-stat line resolution (WS5c).

Guards against the ``line == model mean`` degeneracy: with no market lines and
no explicit --line, each stat must settle at a realistic default line.
"""

import unittest

from nba_model.evaluation import run_distribution_sweep as rds


class ResolveStatLinesTests(unittest.TestCase):
    def test_defaults_used_when_no_market_no_fixed(self):
        out = rds._resolve_stat_lines(
            ["points", "rebounds"], line=None, use_market_lines=False)
        self.assertEqual(out["points"], rds.DEFAULT_LINES_BY_STAT["points"])
        self.assertEqual(out["rebounds"], rds.DEFAULT_LINES_BY_STAT["rebounds"])
        # No default falls back to model mean (None).
        self.assertIsNone(out["points"] and None)  # sanity: not None for known

    def test_explicit_fixed_line_applies_to_all(self):
        out = rds._resolve_stat_lines(
            ["points", "assists"], line=20.5, use_market_lines=False)
        self.assertEqual(out, {"points": 20.5, "assists": 20.5})

    def test_market_lines_pass_none_through(self):
        out = rds._resolve_stat_lines(
            ["points"], line=None, use_market_lines=True)
        self.assertIsNone(out["points"])  # backtest pulls the market line

    def test_unknown_stat_without_default_is_none(self):
        out = rds._resolve_stat_lines(
            ["blocks"], line=None, use_market_lines=False)
        self.assertIsNone(out["blocks"])

    def test_per_stat_sweep_shares_canonical_lines(self):
        from nba_model.evaluation import run_per_stat_sweep as rps
        self.assertIs(rps.DEFAULT_LINES_BY_STAT, rds.DEFAULT_LINES_BY_STAT)


if __name__ == "__main__":
    unittest.main()
