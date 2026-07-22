"""Pure (Streamlit-free) helpers for the Line edge scanner web view.

Mirrors ``cross_book_view.py``: the model-mode selector labels, the
label→``model_mode`` mapping, and the capability-gated fallback live here so they
are unit-testable without a Streamlit runtime. Keep this module import-safe
(no ``import streamlit`` at top level).

The ``full`` mode (WS9: rolling μ/σ + ``blend_team_prior`` + per-stat default
distribution) is scored by ``edge_scanner.score_prop_edges`` exactly as it is for
the CLI and the Cross-book view; the fallback / labeling contract is shared with
``cross_book_view`` so the two views stay consistent.
"""
from __future__ import annotations

# Re-export the shared capability probe + graceful-degradation helper so the
# Line edge scanner view degrades ``full`` → ``chart_mean`` with the same notice
# and single source of truth the Cross-book view uses.
from nba_model.web.cross_book_view import (  # noqa: F401
    full_model_supported,
    resolve_model_mode,
)

# Model-mode selector labels, in display order. Same labeling convention as the
# Cross-book view ("Full model (beta)").
MODEL_MODE_LABELS = [
    "Last-N mean (charts)",
    "Rolling window (prop board)",
    "Full model (beta)",
]


def label_to_mode(label: str) -> str:
    """Map a selector label to the ``score_prop_edges`` ``model_mode`` string.

    Uses the same prefix convention as the Cross-book view so both selectors
    agree: ``Last-N…`` → ``chart_mean``, ``Rolling…`` → ``rolling``, anything
    else (``Full model (beta)``) → ``full``.
    """
    return (
        "chart_mean" if label.startswith("Last-N")
        else "rolling" if label.startswith("Rolling")
        else "full"
    )
