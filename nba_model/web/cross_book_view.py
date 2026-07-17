"""Pure (Streamlit-free) helpers for the Cross-book web view (WS8).

The rendering lives in ``app._cross_book_view``; the number-crunching that feeds
the KPI row and the table lives here so it is unit-testable without a Streamlit
runtime. Keep this module import-safe (no ``import streamlit`` at top level).
"""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

# Columns shown in the main cross-book table, in order. The full frame (with
# model_mu / model_sigma / middle_size) is still what the CSV export ships.
DISPLAY_COLUMNS = [
    "player_name", "stat_type", "n_books",
    "line_min", "line_max", "line_gap",
    "best_over_book", "best_under_book", "consensus_mean",
    "p_over_at_line_min", "p_over_at_line_max", "opportunity_type",
]


def _row_count(df: Optional[pd.DataFrame]) -> int:
    return 0 if df is None or df.empty else int(len(df))


def compute_kpis(
    scored_df: Optional[pd.DataFrame],
    cross_df: Optional[pd.DataFrame],
    arb_df: Optional[pd.DataFrame],
    min_gap: float,
) -> dict:
    """KPI values for the Cross-book header row.

    ``pairs`` = cross-book (player, stat) opportunities; ``max_gap`` = widest
    line gap; ``over_threshold`` = how many meet the min-gap slider; ``arb_count``
    = confirmed two-way arbs (real odds only); ``freshest_hours`` = freshest
    scraped line age from the scored slate (None when unknown).
    """
    pairs = _row_count(cross_df)
    max_gap = (
        float(cross_df["line_gap"].max())
        if pairs and "line_gap" in cross_df.columns else 0.0
    )
    over_threshold = (
        int((cross_df["line_gap"] >= float(min_gap)).sum())
        if pairs and "line_gap" in cross_df.columns else 0
    )
    freshest = None
    if (scored_df is not None and not scored_df.empty
            and "observed_hours_ago" in scored_df.columns):
        col = scored_df["observed_hours_ago"].dropna()
        if not col.empty:
            freshest = float(col.min())
    return {
        "pairs": pairs,
        "max_gap": max_gap,
        "over_threshold": over_threshold,
        "arb_count": _row_count(arb_df),
        "freshest_hours": freshest,
    }


def prepare_display_table(cross_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Ordered subset of the cross-book frame for on-screen display."""
    if cross_df is None or cross_df.empty:
        return pd.DataFrame(columns=[c for c in DISPLAY_COLUMNS])
    cols = [c for c in DISPLAY_COLUMNS if c in cross_df.columns]
    return cross_df[cols].reset_index(drop=True)


def full_model_supported(es_module) -> bool:
    """Best-effort probe: does ``score_prop_edges`` support ``model_mode='full'``?

    The shared contract with the model-layer agent is append-only, so we look
    for a capability marker (``SUPPORTED_MODEL_MODES`` / ``MODEL_MODES``) rather
    than importing internals or hard-coding a version. Returns ``False`` when the
    marker is absent, so the UI degrades to ``chart_mean`` (always valid).
    """
    for attr in ("SUPPORTED_MODEL_MODES", "MODEL_MODES", "MODEL_MODE_CHOICES"):
        modes = getattr(es_module, attr, None)
        if modes is not None:
            try:
                return "full" in set(modes)
            except TypeError:
                continue
    return False


def resolve_model_mode(requested: str, es_module) -> Tuple[str, Optional[str]]:
    """Map a requested UI mode to one we can actually run, plus a user notice.

    ``"full"`` degrades to ``"chart_mean"`` with a caption notice when the model
    layer hasn't shipped full-mode support yet. Other modes pass through. This is
    the capability-check half of the graceful-degradation contract; the caller
    still wraps the actual ``score_prop_edges`` call in try/except as a backstop.
    """
    if requested == "full" and not full_model_supported(es_module):
        return "chart_mean", (
            "Full model mode isn't available in this build yet (pending the "
            "model-layer update) — scoring with last-N mean (charts) instead."
        )
    return requested, None
