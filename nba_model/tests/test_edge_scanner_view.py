"""Tests for the Streamlit-free helpers behind the Line edge scanner web view.

These exercise ``nba_model.web.edge_scanner_view`` without a Streamlit runtime:
the model-mode selector labels, the label→``model_mode`` mapping, and the
capability-gated fallback contract (re-exported from ``cross_book_view``) that
lets the view expose ``full`` mode and degrade gracefully when it is missing.
"""

import types
import unittest

from nba_model.model import edge_scanner as es
from nba_model.web import edge_scanner_view as esv


class LabelToModeTests(unittest.TestCase):
    def test_labels_are_the_three_modes_in_order(self):
        self.assertEqual(
            esv.MODEL_MODE_LABELS,
            [
                "Last-N mean (charts)",
                "Rolling window (prop board)",
                "Full model (beta)",
            ],
        )

    def test_full_label_uses_cross_book_convention(self):
        # Same labeling convention as the Cross-book view's selector.
        self.assertIn("Full model (beta)", esv.MODEL_MODE_LABELS)

    def test_each_label_maps_to_its_mode(self):
        self.assertEqual(esv.label_to_mode("Last-N mean (charts)"), "chart_mean")
        self.assertEqual(
            esv.label_to_mode("Rolling window (prop board)"), "rolling")
        self.assertEqual(esv.label_to_mode("Full model (beta)"), "full")

    def test_every_selector_label_maps_to_a_real_model_mode(self):
        for label in esv.MODEL_MODE_LABELS:
            self.assertIn(esv.label_to_mode(label), es.MODEL_MODES)

    def test_unknown_label_defaults_to_full(self):
        # Matches the Cross-book else-branch: anything not Last-N/Rolling → full.
        self.assertEqual(esv.label_to_mode("Something else"), "full")


class ModelModeFallbackTests(unittest.TestCase):
    def test_full_passes_through_against_real_edge_scanner(self):
        # The live edge_scanner advertises full-mode support, so the view's
        # "Full model (beta)" selection reaches the scorer unchanged.
        mode, notice = esv.resolve_model_mode(
            esv.label_to_mode("Full model (beta)"), es)
        self.assertEqual(mode, "full")
        self.assertIsNone(notice)

    def test_full_degrades_when_support_marker_absent(self):
        fake = types.SimpleNamespace(MODEL_MODES=("chart_mean", "rolling"))
        mode, notice = esv.resolve_model_mode("full", fake)
        self.assertEqual(mode, "chart_mean")
        self.assertIsNotNone(notice)

    def test_full_model_supported_reexported(self):
        # Re-exported helper shares its source of truth with the Cross-book view.
        self.assertTrue(esv.full_model_supported(es))
        self.assertFalse(
            esv.full_model_supported(types.SimpleNamespace()))

    def test_other_modes_pass_through_unchanged(self):
        for m in ("chart_mean", "rolling"):
            mode, notice = esv.resolve_model_mode(m, es)
            self.assertEqual(mode, m)
            self.assertIsNone(notice)


if __name__ == "__main__":
    unittest.main()
