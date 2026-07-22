"""Read-only FastAPI service for the flagship NBA-props web UI.

This package is an ADDITIVE, read-only wrapper around the existing Python
data layer (``nba_model.visualization.player_charts`` fetchers,
``nba_model.model.edge_scanner``, ``nba_model.model.cross_book_arb``, and the
``DatabaseManager`` consensus queries). It never writes to the database and it
never duplicates model/data logic — every endpoint calls the same functions the
Streamlit app uses, validating inputs through
``nba_model.web.input_validation`` exactly like the Streamlit dispatch does.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
