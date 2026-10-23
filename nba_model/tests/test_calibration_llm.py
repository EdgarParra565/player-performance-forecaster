"""calibration_report --source llm (WS11): Brier decomposition + ECE, and the
LLM-vs-model-vs-market-vs-0.5 report on the exact same props. Offline, temp
DBs only (FakeProvider answers)."""

import contextlib
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from nba_model.evaluation import calibration_report as cr
from nba_model.evaluation import llm_forecast as lf
from nba_model.tests.llm_fixtures import (
    EXPECTED_MARKET_PROB, LEBRON, NEXT, TARGET, add_real_prices, seed_llm_db,
)


class DecompositionTests(unittest.TestCase):
    def test_hand_computed_example(self):
        d = cr.brier_decomposition([0.15, 0.15, 0.85, 0.85, 0.85], [0, 1, 1, 1, 0])
        self.assertEqual(d["n"], 5)
        self.assertAlmostEqual(d["brier"], 0.3025)
        self.assertAlmostEqual(d["reliability"], (2 * 0.35 ** 2 + 3 * (0.85 - 2 / 3) ** 2) / 5)
        self.assertAlmostEqual(d["resolution"], (2 * 0.1 ** 2 + 3 * (2 / 3 - 0.6) ** 2) / 5)
        self.assertAlmostEqual(d["uncertainty"], 0.24)
        self.assertAlmostEqual(d["within_bin"], 0.0)       # one value per bucket
        self.assertAlmostEqual(d["ece"], (2 * 0.35 + 3 * (0.85 - 2 / 3)) / 5)
        self.assertAlmostEqual(d["brier"],
                               d["reliability"] - d["resolution"] + d["uncertainty"])

    def test_hedging_at_half_is_calibrated_but_has_no_resolution(self):
        d = cr.brier_decomposition([0.5] * 4, [1, 1, 0, 1])
        self.assertAlmostEqual(d["brier"], 0.25)
        self.assertAlmostEqual(d["resolution"], 0.0)
        self.assertAlmostEqual(d["reliability"], 0.0625)
        self.assertAlmostEqual(d["uncertainty"], 0.1875)
        self.assertAlmostEqual(d["ece"], 0.25)

    def test_perfect_forecaster_resolves_all_uncertainty(self):
        d = cr.brier_decomposition([1, 0, 1, 1, 0], [1, 0, 1, 1, 0])
        self.assertAlmostEqual(d["brier"], 0.0)
        self.assertAlmostEqual(d["reliability"], 0.0)
        self.assertAlmostEqual(d["resolution"], d["uncertainty"])
        self.assertAlmostEqual(d["ece"], 0.0)

    def test_within_bin_is_negative_when_ranking_inside_a_bucket_is_right(self):
        d = cr.brier_decomposition([0.41, 0.49], [0, 1])
        self.assertAlmostEqual(d["brier"], 0.2141)
        self.assertAlmostEqual(d["reliability"], 0.0025)
        self.assertAlmostEqual(d["resolution"], 0.0)
        self.assertAlmostEqual(d["within_bin"], 0.2141 - 0.2525)

    def test_within_bin_term_closes_the_identity(self):
        rng = np.random.default_rng(7)
        pred = rng.uniform(size=200)
        hit = (rng.uniform(size=200) < pred).astype(float)
        d = cr.brier_decomposition(pred, hit, n_buckets=5)
        self.assertNotAlmostEqual(d["within_bin"], 0.0)
        self.assertAlmostEqual(
            d["brier"],
            d["reliability"] - d["resolution"] + d["uncertainty"] + d["within_bin"])

    def test_nan_rows_dropped_and_empty_input(self):
        d = cr.brier_decomposition([0.5, np.nan, 0.5], [1, 1, np.nan])
        self.assertEqual(d["n"], 1)
        empty = cr.brier_decomposition([], [])
        self.assertEqual(empty["n"], 0)
        self.assertTrue(np.isnan(empty["brier"]))


class LlmReportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.db_path = str(self.dir / "nba.db")
        self.ids = seed_llm_db(self.db_path)

    def tearDown(self):
        self._tmp.cleanup()

    def collect(self, provider, run_id, **kwargs):
        return lf.run_llm_forecasts(
            self.db_path, provider, retro=True, run_id=run_id, limit=None,
            cache_dir=self.dir / "cache", artifact_dir=self.dir / "runs", **kwargs)

    def two_models(self, run_id="r1"):
        self.collect(lf.FakeProvider(model="fake-a", responder=lambda s, u: "0.7"), run_id,
                     training_cutoff="2024-06-30")
        self.collect(lf.FakeProvider(model="fake-b", responder=lambda s, u: (
            "maybe" if "PRA" in u else "0.6")), run_id)

    def report(self, run_id="r1"):
        return cr.run_calibration_report(self.db_path, source="llm", run_id=run_id,
                                         artifact_dir=str(self.dir / "artifacts"))

    def test_llm_vs_baselines_on_the_common_prop_set(self):
        self.two_models()
        result = self.report()
        for key in ("summary_csv", "brier_csv", "reliability_csv", "coverage_csv", "md_path"):
            self.assertTrue(Path(result[key]).is_file(), key)
        self.assertEqual((result["props_in_run"], result["scored_props"]), (4, 3))
        self.assertEqual(result["leak_risk_props"], 3)     # fake-b: cutoff unknown
        self.assertEqual(result["market_real"], 0)

        summary = pd.read_csv(result["summary_csv"])
        self.assertEqual(set(summary["section"]), {"leak_risk"})
        self.assertEqual(list(summary["forecaster"]),
                         ["fake-a", "fake-b", "model", "market", "constant 0.5"])
        self.assertEqual(set(summary["n"]), {3})            # PRA dropped: fake-b invalid
        by = summary.set_index("forecaster")
        self.assertAlmostEqual(by.loc["fake-a", "brier"], (0.09 + 0.09 + 0.49) / 3)
        self.assertAlmostEqual(by.loc["model", "brier"],     # production-config rows
                               (0.45 ** 2 + 0.52 ** 2 + 0.45 ** 2) / 3)
        self.assertAlmostEqual(by.loc["constant 0.5", "brier"], 0.25)
        self.assertAlmostEqual(by.loc["constant 0.5", "resolution"], 0.0)
        assumed = lf.ASSUMED_PRICE_PROB
        self.assertAlmostEqual(by.loc["market", "mean_pred"], assumed)
        self.assertAlmostEqual(by.loc["market", "brier"],
                               (2 * (1 - assumed) ** 2 + assumed ** 2) / 3)

        coverage = pd.read_csv(result["coverage_csv"]).set_index("forecaster")
        self.assertEqual(coverage.loc["fake-b", ["rows", "ok", "invalid", "scored"]].tolist(),
                         [4, 3, 1, 3])
        self.assertEqual(coverage.loc["fake-a", "leak_risk_rows"], 0)
        self.assertEqual(coverage.loc["fake-b", "leak_risk_rows"], 4)

        md = Path(result["md_path"]).read_text(encoding="utf-8")
        self.assertIn("not a headline result", md)
        self.assertIn("## Headline (leak-safe) — 0 props", md)
        self.assertIn("## Reported separately — possible training-data leakage — 3 props", md)
        self.assertIn("| constant 0.5 | 3 | 0.2500 |", md)
        self.assertIn("-110 assumption", md)
        self.assertIn("limited by fake-b (3 scored)", md)
        self.assertEqual(result["model_lookahead_props"], 2)   # the two points props
        self.assertIn("Model baseline caveat:** 2 points props", md)

    def test_known_pre_game_cutoff_is_headline(self):
        self.collect(lf.FakeProvider(model="fake-a"), "r2", training_cutoff="2024-06-30")
        result = self.report("r2")
        self.assertEqual((result["scored_props"], result["leak_risk_props"]), (4, 0))
        summary = pd.read_csv(result["summary_csv"])
        self.assertEqual(set(summary["section"]), {"headline"})
        md = Path(result["md_path"]).read_text(encoding="utf-8")
        self.assertIn("## Headline (leak-safe) — 4 props", md)
        self.assertNotIn("not a headline result", md)

    def test_market_baseline_uses_real_prices_when_posted(self):
        add_real_prices(self.db_path)
        self.collect(lf.FakeProvider(model="fake-a"), "r1")
        loaded = cr.load_llm_calibration_frame(self.db_path, "r1")
        self.assertEqual(loaded["market_real"], 1)
        long = loaded["long"]
        market = long[long["forecaster"] == "market"].set_index(["game_date", "stat_type"])
        self.assertAlmostEqual(market.loc[(TARGET, "points"), "pred_prob"],
                               EXPECTED_MARKET_PROB)
        self.assertAlmostEqual(market.loc[(NEXT, "points"), "pred_prob"],
                               lf.ASSUMED_PRICE_PROB)
        model = long[long["forecaster"] == "model"].set_index(["game_date", "stat_type"])
        self.assertAlmostEqual(model.loc[(TARGET, "rebounds"), "pred_prob"], 0.48)
        self.assertEqual(set(long["player_id"]), {LEBRON})

    def test_newest_run_by_default(self):
        self.collect(lf.FakeProvider(model="fake-a"), "old")
        self.collect(lf.FakeProvider(model="fake-a"), "new")
        self.assertEqual(cr.load_llm_calibration_frame(self.db_path)["run_id"], "new")

    def test_db_without_the_table_reports_empty(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("DROP TABLE llm_forecasts")
        conn.commit()
        conn.close()
        result = self.report(None)
        self.assertEqual((result["forecast_rows"], result["scored_props"]), (0, 0))
        md = Path(result["md_path"]).read_text(encoding="utf-8")
        self.assertIn("_No forecasts in this run._", md)
        conn = sqlite3.connect(self.db_path)          # read-only: nothing recreated
        self.assertIsNone(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'llm_forecasts'").fetchone())
        conn.close()

    def test_cli(self):
        self.two_models()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cr.main(["--db", self.db_path, "--source", "llm", "--run-id", "r1",
                            "--artifact-dir", str(self.dir / "artifacts")])
        self.assertEqual(code, 0)
        self.assertIn("- source: llm", out.getvalue())
        self.assertIn("- scored_props: 3", out.getvalue())


if __name__ == "__main__":
    unittest.main()
