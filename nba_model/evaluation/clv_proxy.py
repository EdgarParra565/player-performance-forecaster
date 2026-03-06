"""Simple CLV-style proxy using historical line snapshots.

For each (player, game_date, stat_type, book, market_key) combination:
  - identify opening and closing snapshots
  - compute changes in line and implied probabilities

This does not require knowing the actual bet; it describes how markets moved
between open and close and can be joined to model signals offline.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model.odds import american_to_implied_prob

ARTIFACT_DIR = Path("nba_model/evaluation/artifacts")


def _safe_implied_prob(odds) -> Optional[float]:
    """Convert American odds to implied probability; return None if invalid."""
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return None
    try:
        return float(american_to_implied_prob(int(odds)))
    except (TypeError, ValueError):
        return None


def load_snapshots(
    db_path: str,
    start_date: Optional[str],
    end_date: Optional[str],
    stat_types: Optional[list[str]],
) -> pd.DataFrame:
    """Load betting_line_snapshots from DB, optionally filtered by date and stat_type."""
    with DatabaseManager(db_path=db_path) as db:
        df = pd.read_sql_query("SELECT * FROM betting_line_snapshots", db.conn)

    if df.empty:
        return df

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    df["snapshot_ts_utc"] = pd.to_datetime(
        df["snapshot_ts_utc"], errors="coerce", utc=True)
    if start_date:
        start = pd.to_datetime(start_date).date()
        df = df[df["game_date"] >= start]
    if end_date:
        end = pd.to_datetime(end_date).date()
        df = df[df["game_date"] <= end]
    if stat_types:
        norm_stats = {str(s).strip().lower() for s in stat_types}
        df["stat_type"] = df["stat_type"].astype(str).str.lower()
        df = df[df["stat_type"].isin(norm_stats)]

    return df.reset_index(drop=True)


def build_clv_proxy(
    snapshots_df: pd.DataFrame,
    min_snapshots_per_market: int = 2,
) -> pd.DataFrame:
    """Build open/close CLV-style metrics per (player, game_date, stat_type, book, market)."""
    if snapshots_df.empty:
        return pd.DataFrame()

    key_cols = [
        "player_id",
        "game_date",
        "stat_type",
        "book",
        "market_key",
    ]
    snapshots_df = snapshots_df.dropna(
        subset=key_cols + ["snapshot_ts_utc"]).copy()
    if snapshots_df.empty:
        return pd.DataFrame()

    rows = []
    for key, group in snapshots_df.groupby(key_cols):
        if len(group) < max(2, int(min_snapshots_per_market)):
            continue
        group = group.sort_values("snapshot_ts_utc")
        open_row = group.iloc[0]
        close_row = group.iloc[-1]

        def _build_side_metrics(row, prefix: str):
            over_odds = row.get("over_odds")
            under_odds = row.get("under_odds")
            return {
                f"{prefix}_line": float(row["line_value"]),
                f"{prefix}_over_odds": int(over_odds) if pd.notna(over_odds) else None,
                f"{prefix}_under_odds": int(under_odds) if pd.notna(under_odds) else None,
                f"{prefix}_over_implied": _safe_implied_prob(over_odds),
                f"{prefix}_under_implied": _safe_implied_prob(under_odds),
            }

        row = {
            "player_id": key[0],
            "game_date": str(key[1]),
            "stat_type": key[2],
            "book": key[3],
            "market_key": key[4],
            "snapshots_count": int(len(group)),
            "open_snapshot_ts_utc": open_row["snapshot_ts_utc"].isoformat(),
            "close_snapshot_ts_utc": close_row["snapshot_ts_utc"].isoformat(),
        }
        row.update(_build_side_metrics(open_row, "open"))
        row.update(_build_side_metrics(close_row, "close"))

        # Deltas: line move and implied-prob move for over/under.
        row["delta_line"] = float(row["close_line"] - row["open_line"])
        for side in ("over", "under"):
            open_prob = row.get(f"open_{side}_implied")
            close_prob = row.get(f"close_{side}_implied")
            if open_prob is not None and close_prob is not None:
                row[f"delta_{side}_implied"] = float(close_prob - open_prob)
            else:
                row[f"delta_{side}_implied"] = None

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(
        ["game_date", "player_id", "stat_type", "book"]
    ).reset_index(drop=True)
    return out


def run_clv_proxy(
    db_path: str = "data/database/nba_data.db",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stat_types: Optional[list[str]] = None,
    min_snapshots_per_market: int = 2,
    output_prefix: str = "clv_proxy",
) -> dict:
    """Load snapshots, build CLV proxy table, write CSV and return summary."""
    snapshots_df = load_snapshots(
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        stat_types=stat_types,
    )
    clv_df = build_clv_proxy(
        snapshots_df=snapshots_df,
        min_snapshots_per_market=min_snapshots_per_market,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = ARTIFACT_DIR / f"{output_prefix}_{ts}.csv"
    clv_df.to_csv(csv_path, index=False)

    return {
        "csv_path": str(csv_path),
        "rows": int(len(clv_df)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute simple CLV-style proxies from line snapshots."
    )
    parser.add_argument("--db-path", default="data/database/nba_data.db")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--stat-types", nargs="*", default=None)
    parser.add_argument("--min-snapshots-per-market", type=int, default=2)
    parser.add_argument("--output-prefix", default="clv_proxy")
    return parser


def main() -> None:
    """CLI entry point: run CLV proxy and print summary."""
    args = _build_parser().parse_args()
    result = run_clv_proxy(
        db_path=args.db_path,
        start_date=args.start_date,
        end_date=args.end_date,
        stat_types=args.stat_types,
        min_snapshots_per_market=args.min_snapshots_per_market,
        output_prefix=args.output_prefix,
    )
    print("CLV proxy summary:")
    for key, value in result.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
