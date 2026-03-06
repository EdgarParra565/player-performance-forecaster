"""Market data quality checks for betting_lines and betting_line_snapshots.

Runs lightweight diagnostics for:
  - missing/NULL critical fields
  - potential duplicate snapshots
  - snapshot freshness (time since last snapshot)
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from nba_model.data.database.db_manager import DatabaseManager

ARTIFACT_DIR = Path("nba_model/evaluation/artifacts")


def _utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def run_market_data_quality_checks(
    db_path: str = "data/database/nba_data.db",
    max_snapshot_age_hours: float = 6.0,
    output_prefix: str = "market_data_quality",
) -> dict:
    """Run basic data-quality checks on betting lines and snapshots."""
    with DatabaseManager(db_path=db_path) as db:
        lines_df = pd.read_sql_query("SELECT * FROM betting_lines", db.conn)
        snapshots_df = pd.read_sql_query(
            "SELECT * FROM betting_line_snapshots", db.conn)

    summary: dict = {
        "betting_lines_rows": int(len(lines_df)),
        "snapshots_rows": int(len(snapshots_df)),
    }

    # Missing field checks on betting_lines.
    if not lines_df.empty:
        summary["betting_lines_missing_player_id"] = int(
            lines_df["player_id"].isna().sum())
        summary["betting_lines_missing_game_date"] = int(
            lines_df["game_date"].isna().sum())
        summary["betting_lines_missing_stat_type"] = int(
            lines_df["stat_type"].isna().sum())
        summary["betting_lines_missing_line_value"] = int(
            lines_df["line_value"].isna().sum())
    else:
        summary.update(
            {
                "betting_lines_missing_player_id": 0,
                "betting_lines_missing_game_date": 0,
                "betting_lines_missing_stat_type": 0,
                "betting_lines_missing_line_value": 0,
            }
        )

    # Snapshot-specific checks.
    if not snapshots_df.empty:
        snapshots_df["snapshot_ts_utc"] = pd.to_datetime(
            snapshots_df["snapshot_ts_utc"], errors="coerce", utc=True
        )
        summary["snapshots_missing_timestamp"] = int(
            snapshots_df["snapshot_ts_utc"].isna().sum())
        summary["snapshots_missing_player_id"] = int(
            snapshots_df["player_id"].isna().sum())
        summary["snapshots_missing_book"] = int(
            snapshots_df["book"].isna().sum())
        summary["snapshots_missing_stat_type"] = int(
            snapshots_df["stat_type"].isna().sum())
        summary["snapshots_missing_line_value"] = int(
            snapshots_df["line_value"].isna().sum())

        # Potential duplicate snapshots: same key within identical timestamp.
        dup_key_cols = [
            "snapshot_ts_utc",
            "event_id",
            "game_date",
            "player_id",
            "book",
            "market_key",
            "stat_type",
            "line_value",
            "over_odds",
            "under_odds",
        ]
        if all(col in snapshots_df.columns for col in dup_key_cols):
            dup_counts = (
                snapshots_df.groupby(dup_key_cols)
                .size()
                .reset_index(name="count")
            )
            duplicates = dup_counts[dup_counts["count"] > 1]["count"].sum()
            summary["snapshots_exact_duplicate_rows"] = int(duplicates)
        else:
            summary["snapshots_exact_duplicate_rows"] = 0

        # Freshness: time since latest snapshot.
        latest_ts = snapshots_df["snapshot_ts_utc"].max()
        if pd.isna(latest_ts):
            summary["latest_snapshot_ts_utc"] = None
            summary["hours_since_latest_snapshot"] = None
            summary["snapshots_stale_flag"] = False
        else:
            now = _utc_now()
            hours_since = (now - latest_ts).total_seconds() / 3600.0
            summary["latest_snapshot_ts_utc"] = latest_ts.isoformat()
            summary["hours_since_latest_snapshot"] = float(hours_since)
            summary["snapshots_stale_flag"] = bool(
                max_snapshot_age_hours > 0.0 and hours_since > max_snapshot_age_hours
            )
    else:
        summary.update(
            {
                "snapshots_missing_timestamp": 0,
                "snapshots_missing_player_id": 0,
                "snapshots_missing_book": 0,
                "snapshots_missing_stat_type": 0,
                "snapshots_missing_line_value": 0,
                "snapshots_exact_duplicate_rows": 0,
                "latest_snapshot_ts_utc": None,
                "hours_since_latest_snapshot": None,
                "snapshots_stale_flag": False,
            }
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = ARTIFACT_DIR / f"{output_prefix}_{ts}.json"
    output_path.write_text(
        pd.Series(summary).to_json(indent=2), encoding="utf-8")
    summary["report_path"] = str(output_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run market data quality checks.")
    parser.add_argument("--db-path", default="data/database/nba_data.db")
    parser.add_argument(
        "--max-snapshot-age-hours",
        type=float,
        default=6.0,
        help="Flag market snapshots as stale when latest snapshot is older than this many hours.",
    )
    parser.add_argument("--output-prefix", default="market_data_quality")
    return parser


def main() -> None:
    """CLI entry point: run market data quality checks and print summary."""
    args = _build_parser().parse_args()
    summary = run_market_data_quality_checks(
        db_path=args.db_path,
        max_snapshot_age_hours=args.max_snapshot_age_hours,
        output_prefix=args.output_prefix,
    )
    print("Market data quality summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
