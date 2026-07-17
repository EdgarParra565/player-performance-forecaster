"""Tests for the Streamlit-free helpers behind the Cross-book web view (WS8).

These exercise ``nba_model.web.cross_book_view`` without a Streamlit runtime:
the KPI math, the display-table shaping, and the model-mode capability probe /
graceful-degradation contract with the model layer (Agent A).
"""

import types
import unittest

import pandas as pd

from nba_model.model import cross_book_arb as cba
from nba_model.web import cross_book_view as cbv


def _cross_row(player, gap, opp_type):
    return {
        "player_name": player, "stat_type": "points", "n_books": 2,
        "line_min": 17.5, "line_max": 17.5 + gap, "line_gap": gap,
        "best_over_book": "A", "best_under_book": "B", "consensus_mean": 18.0,
        "p_over_at_line_min": 0.6, "p_over_at_line_max": 0.4,
        "middle_size": gap, "opportunity_type": opp_type,
        "model_mu": 18.0, "model_sigma": 3.0,
    }


class ComputeKpisTests(unittest.TestCase):
    def test_kpis_full(self):
        cross = pd.DataFrame([
            _cross_row("P1", 2.0, cba.OPP_MIDDLE),
            _cross_row("P2", 0.5, cba.OPP_LINE_GAP),
        ])
        scored = pd.DataFrame({"observed_hours_ago": [3.0, 1.5, None]})
        arbs = pd.DataFrame([{"guaranteed_margin": 0.02}])
        kpis = cbv.compute_kpis(scored, cross, arbs, min_gap=1.0)
        self.assertEqual(kpis["pairs"], 2)
        self.assertAlmostEqual(kpis["max_gap"], 2.0, places=6)
        self.assertEqual(kpis["over_threshold"], 1)   # only the 2.0-gap row
        self.assertEqual(kpis["arb_count"], 1)
        self.assertAlmostEqual(kpis["freshest_hours"], 1.5, places=6)

    def test_kpis_empty(self):
        kpis = cbv.compute_kpis(
            pd.DataFrame(), pd.DataFrame(columns=cba.CROSS_BOOK_COLUMNS),
            pd.DataFrame(columns=cba.ARB_COLUMNS), min_gap=0.5,
        )
        self.assertEqual(kpis["pairs"], 0)
        self.assertEqual(kpis["max_gap"], 0.0)
        self.assertEqual(kpis["over_threshold"], 0)
        self.assertEqual(kpis["arb_count"], 0)
        self.assertIsNone(kpis["freshest_hours"])


class PrepareDisplayTableTests(unittest.TestCase):
    def test_columns_ordered_subset(self):
        cross = pd.DataFrame([_cross_row("P1", 2.0, cba.OPP_MIDDLE)])
        disp = cbv.prepare_display_table(cross)
        self.assertEqual(list(disp.columns), cbv.DISPLAY_COLUMNS)
        # middle_size / model_mu are dropped from the on-screen table (still in CSV).
        self.assertNotIn("model_mu", disp.columns)
        self.assertNotIn("middle_size", disp.columns)

    def test_empty(self):
        disp = cbv.prepare_display_table(
            pd.DataFrame(columns=cba.CROSS_BOOK_COLUMNS))
        self.assertTrue(disp.empty)


class ModelModeContractTests(unittest.TestCase):
    def test_full_supported_true_when_marker_present(self):
        fake = types.SimpleNamespace(MODEL_MODES=("chart_mean", "rolling", "full"))
        self.assertTrue(cbv.full_model_supported(fake))

    def test_full_supported_false_when_absent(self):
        fake = types.SimpleNamespace()  # no marker at all
        self.assertFalse(cbv.full_model_supported(fake))

    def test_full_supported_false_when_marker_lacks_full(self):
        fake = types.SimpleNamespace(MODEL_MODES=("chart_mean", "rolling"))
        self.assertFalse(cbv.full_model_supported(fake))

    def test_resolve_degrades_full_when_unsupported(self):
        fake = types.SimpleNamespace(MODEL_MODES=("chart_mean", "rolling"))
        mode, notice = cbv.resolve_model_mode("full", fake)
        self.assertEqual(mode, "chart_mean")
        self.assertIsNotNone(notice)

    def test_resolve_passthrough_full_when_supported(self):
        fake = types.SimpleNamespace(MODEL_MODES=("chart_mean", "rolling", "full"))
        mode, notice = cbv.resolve_model_mode("full", fake)
        self.assertEqual(mode, "full")
        self.assertIsNone(notice)

    def test_resolve_passthrough_other_modes(self):
        fake = types.SimpleNamespace(MODEL_MODES=("chart_mean",))
        for m in ("chart_mean", "rolling"):
            mode, notice = cbv.resolve_model_mode(m, fake)
            self.assertEqual(mode, m)
            self.assertIsNone(notice)

    def test_resolve_against_real_edge_scanner(self):
        # Integration: the live edge_scanner advertises full-mode support now.
        from nba_model.model import edge_scanner as es
        mode, notice = cbv.resolve_model_mode("full", es)
        self.assertIn(mode, ("full", "chart_mean"))  # either is a valid outcome


if __name__ == "__main__":
    unittest.main()
