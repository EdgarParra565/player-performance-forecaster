"""NFL results ingestion via ``nfl_data_py`` (WS6 scaffolding).

``nfl_data_py`` wraps the ``nflverse`` Parquet drops — it is the NFL
equivalent of ``nba_api``. It is NOT installed in this environment and live
NFL data isn't available here, so this module is structured so that:

  * importing it never fails (the dependency is imported lazily, per call);
  * the pure transform (`transform_weekly_to_player_logs`) works on an
    in-memory DataFrame fixture with no network and no optional dep;
  * any function that actually needs ``nfl_data_py`` raises a clear,
    actionable ``RuntimeError`` when the dep is absent (not a bare
    ``ImportError`` at import time).

Persistence note: the existing ``game_logs`` schema is NBA-shaped, so this
module deliberately stops at producing normalized per-player-week dicts
(tagged ``sport='nfl'``). Wiring those into a sport-aware table is the
follow-up once a sport column / per-sport schema decision lands (see
``sports/nfl.py`` notes).

CLI (only works once ``pip install nfl_data_py`` is done on a live host):
    .venv/bin/python3 -m nba_model.data.nfl_results_ingestion --seasons 2025 2024
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
from typing import Optional, Sequence

import pandas as pd

logger = logging.getLogger("nba_model.nfl_results_ingestion")

DEFAULT_SEASONS = (2025, 2024)

# nflverse weekly column -> our normalized stat key (subset of sports/nfl.py).
_WEEKLY_STAT_COLUMNS: dict[str, str] = {
    "passing_yards": "passing_yards",
    "passing_tds": "passing_touchdowns",
    "completions": "passing_completions",
    "attempts": "passing_attempts",
    "interceptions": "interceptions",
    "rushing_yards": "rushing_yards",
    "carries": "rushing_attempts",
    "rushing_tds": "rushing_touchdowns",
    "receiving_yards": "receiving_yards",
    "receptions": "receptions",
    "receiving_tds": "receiving_touchdowns",
}


def nfl_data_py_available() -> bool:
    """True when the optional ``nfl_data_py`` dependency can be imported."""
    return importlib.util.find_spec("nfl_data_py") is not None


def _import_nfl_data_py():
    """Import ``nfl_data_py`` or raise an actionable error.

    Kept out of module scope so importing this module (e.g. in CI / tests)
    never requires the optional dependency.
    """
    if not nfl_data_py_available():
        raise RuntimeError(
            "nfl_data_py is not installed in this environment. "
            "Install it on a host with network access: "
            "`.venv/bin/python3 -m pip install nfl_data_py`. "
            "It wraps the nflverse Parquet drops (NFL equivalent of nba_api)."
        )
    return importlib.import_module("nfl_data_py")


def fetch_weekly_player_stats(seasons: Sequence[int]) -> pd.DataFrame:
    """Fetch nflverse weekly player stats for ``seasons`` (needs nfl_data_py)."""
    nfl = _import_nfl_data_py()
    years = [int(s) for s in seasons]
    logger.info("Fetching nflverse weekly data for seasons: %s", years)
    return nfl.import_weekly_data(years)


def fetch_schedules(seasons: Sequence[int]) -> pd.DataFrame:
    """Fetch nflverse schedules for ``seasons`` (needs nfl_data_py)."""
    nfl = _import_nfl_data_py()
    years = [int(s) for s in seasons]
    logger.info("Fetching nflverse schedules for seasons: %s", years)
    return nfl.import_schedules(years)


def transform_weekly_to_player_logs(df: pd.DataFrame) -> list[dict]:
    """Convert an nflverse weekly-stats frame → normalized per-player-week dicts.

    Pure + dependency-free (operates on a DataFrame), so it is unit-testable
    against a small fixture. Output rows carry ``sport='nfl'`` and only the
    stat columns present in the frame (nflverse omits columns for positions
    that never record them).
    """
    if df is None or df.empty:
        return []

    out: list[dict] = []
    present_stats = {
        col: key for col, key in _WEEKLY_STAT_COLUMNS.items() if col in df.columns
    }
    for _, row in df.iterrows():
        player = row.get("player_display_name") or row.get("player_name")
        if not player:
            continue
        record = {
            "sport": "nfl",
            "player_name": str(player),
            "player_id": (str(row.get("player_id")) if row.get("player_id") is not None else None),
            "season": (int(row["season"]) if pd.notna(row.get("season")) else None),
            "week": (int(row["week"]) if pd.notna(row.get("week")) else None),
            "team": (str(row.get("recent_team")) if row.get("recent_team") else None),
            "position": (str(row.get("position")) if row.get("position") else None),
        }
        for col, key in present_stats.items():
            val = row.get(col)
            record[key] = float(val) if pd.notna(val) else None
        out.append(record)
    return out


def ingest_all(seasons: Optional[Sequence[int]] = None) -> dict:
    """Fetch + transform NFL weekly stats. Requires ``nfl_data_py`` at runtime."""
    seasons = list(seasons or DEFAULT_SEASONS)
    weekly = fetch_weekly_player_stats(seasons)
    records = transform_weekly_to_player_logs(weekly)
    logger.info("Transformed %s NFL player-week rows", len(records))
    return {
        "sport": "nfl",
        "seasons": seasons,
        "player_week_rows": len(records),
        "persisted": 0,  # persistence deferred until a sport-aware schema lands
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest NFL weekly player stats via nfl_data_py (nflverse).",
    )
    parser.add_argument("--seasons", nargs="+", type=int, default=list(DEFAULT_SEASONS))
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args()
    if not nfl_data_py_available():
        print(
            "nfl_data_py is not installed. Install it on a live host:\n"
            "  .venv/bin/python3 -m pip install nfl_data_py"
        )
        return
    summary = ingest_all(seasons=args.seasons)
    print("NFL ingestion summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
