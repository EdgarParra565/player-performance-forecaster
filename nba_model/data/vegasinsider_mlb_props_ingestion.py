"""Ingest VegasInsider's MLB player-props grid into ``mlb_prop_lines``.

Reads the newest stored ``web_text_snapshots`` row for
``vegasinsider.com/mlb/odds/player-props``, parses it with
``nba_model.scrapers.vegasinsider.extract_mlb_odds_rows`` (real American odds,
one column per underlying book), validates each row against the MLB sport
registry (``sports/mlb.py``), and writes change-only rows to ``mlb_prop_lines``.

Why a dedicated table and not ``betting_lines`` / ``web_prop_cards``:
``betting_lines.player_id`` is an NBA ``players`` FK and its NBA readers
(cross_book_arb, prop_board, line_comparison, ...) never filter on ``sport``;
``web_prop_cards`` feeds NBA freshness/book counts. Writing MLB rows to either
would leak them into NBA views. ``mlb_prop_lines`` keeps MLB out of every NBA
query by construction (same rationale as ``mlb_game_logs``).

Row contract:
  * ``sport='mlb'``, ``source='vegasinsider'``, ``book`` = the UNDERLYING book.
  * ``stat_type`` must be in ``sports.mlb.SPORT.stat_types``; lines inside the
    registry's ``stat_line_ranges`` (yes/no markets must sit at 0.5); odds pass
    ``validate_american_odds``. Everything else is counted as rejected.
  * ``stat_group`` ('hitting' | 'pitching' | 'combined') from
    ``mlb_props.stat_group`` so hitter and pitcher markets stay separate.
  * Over-only grid: ``under_odds`` is always NULL (never invented).

CLI::

    .venv/bin/python3 -m nba_model.data.vegasinsider_mlb_props_ingestion \\
        --db-path data/database/nba_data.db
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from typing import Optional

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.data.vegasinsider_odds_ingestion import _game_date_from_fetched
from nba_model.scrapers.mlb_props import stat_group
from nba_model.scrapers.vegasinsider import MLB_PARSER_VERSION, extract_mlb_odds_rows
from nba_model.web.input_validation import ValidationError, validate_american_odds
from sports import get_sport

logger = logging.getLogger("nba_model.vegasinsider_mlb_props_ingestion")

DEFAULT_DB_PATH = "data/database/nba_data.db"
DEFAULT_SOURCE_URL_LIKE = "%vegasinsider.com/mlb/odds/player-props%"
SOURCE_TAG = "vegasinsider"

_MLB = get_sport("mlb")
_YES_NO_STATS = frozenset({"anytime_home_run", "first_run_scorer"})


def _latest_snapshot(db: DatabaseManager, source_url_like: str) -> Optional[dict]:
    """Newest ``web_text_snapshots`` row whose URL matches ``source_url_like``."""
    row = db.conn.execute(
        """
        SELECT snapshot_id, source_url, fetched_at_utc, text_content
        FROM web_text_snapshots
        WHERE source_url LIKE ?
        ORDER BY fetched_at_utc DESC, snapshot_id DESC
        LIMIT 1
        """,
        (source_url_like,),
    ).fetchone()
    if not row:
        return None
    return {
        "snapshot_id": row[0],
        "source_url": row[1],
        "fetched_at_utc": row[2],
        "text_content": row[3] or "",
    }


def is_valid_mlb_prop_row(stat_type: str, line_value, odds) -> bool:
    """Registry + plausibility check for one parsed MLB prop cell."""
    stat = str(stat_type or "").strip().lower()
    if stat not in _MLB.stat_types:
        return False
    try:
        line = float(line_value)
    except (TypeError, ValueError):
        return False
    if stat in _YES_NO_STATS:
        if line != 0.5:
            return False
    else:
        lo_hi = _MLB.stat_line_ranges.get(stat)
        if lo_hi is None or not (lo_hi[0] <= line <= lo_hi[1]):
            return False
    try:
        return validate_american_odds(odds) is not None
    except ValidationError:
        return False


def _record_sha256(snapshot_id, rec: dict) -> str:
    key = "|".join(str(v) for v in (
        snapshot_id, rec["book"], rec["player_name"].lower(), rec["stat_type"],
        rec["side"], rec["line_value"], rec["over_odds"], rec["market_shape"],
    ))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def ingest_vegasinsider_mlb_props(
    db_path: str = DEFAULT_DB_PATH,
    source_url_like: str = DEFAULT_SOURCE_URL_LIKE,
    snapshot_text: Optional[str] = None,
    observed_at_utc: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Parse the latest VegasInsider MLB props snapshot into ``mlb_prop_lines``.

    ``snapshot_text`` / ``observed_at_utc`` override the DB lookup (tests).
    Returns a summary dict (parsed / valid / rejected / inserted / unchanged).
    An empty grid or missing snapshot is a normal 0-row return, not an error.
    """
    with DatabaseManager(db_path=db_path) as db:
        snapshot_id = None
        source_url = None
        if snapshot_text is None:
            snap = _latest_snapshot(db, source_url_like)
            if snap is None:
                return {
                    "source": SOURCE_TAG, "sport": "mlb", "status": "no_snapshot",
                    "parsed_rows": 0, "valid_rows": 0, "inserted": 0,
                }
            snapshot_text = snap["text_content"]
            snapshot_id = snap["snapshot_id"]
            source_url = snap["source_url"]
            observed_at_utc = observed_at_utc or snap["fetched_at_utc"]
        game_date = _game_date_from_fetched(observed_at_utc)

        parsed = extract_mlb_odds_rows(snapshot_text)
        records: list[dict] = []
        rejected = 0
        for r in parsed:
            if not is_valid_mlb_prop_row(r["stat_type"], r["line_value"], r["odds"]):
                rejected += 1
                continue
            rec = {
                "snapshot_id": snapshot_id,
                "source_url": source_url,
                "source": SOURCE_TAG,
                "book": r["book"],
                "observed_at_utc": observed_at_utc,
                "game_date": game_date,
                "player_name": r["player_name"],
                "stat_type": r["stat_type"],
                "stat_group": stat_group(r["stat_type"]),
                "market_shape": r["market_shape"],
                "line_value": float(r["line_value"]),
                "side": r["side"],
                "over_odds": int(r["odds"]),
                "under_odds": None,  # over-only grid — never invent an under
                "parser_version": MLB_PARSER_VERSION,
            }
            rec["record_sha256"] = _record_sha256(snapshot_id, rec)
            records.append(rec)

        summary = {
            "source": SOURCE_TAG,
            "sport": "mlb",
            "status": "success",
            "snapshot_id": snapshot_id,
            "game_date": game_date,
            "parsed_rows": len(parsed),
            "valid_rows": len(records),
            "rejected_invalid": rejected,
            "distinct_players": len({r["player_name"] for r in records}),
            "dry_run": bool(dry_run),
        }
        if dry_run or not records:
            summary.update({"inserted": 0, "skipped_unchanged": 0})
            logger.info("VegasInsider MLB props ingest (%s): %s",
                        "dry-run" if dry_run else "no records", summary)
            return summary

        insert = db.insert_mlb_prop_lines(records)

    summary.update({
        "inserted": int(insert.get("inserted", 0)),
        "skipped_unchanged": int(insert.get("skipped_unchanged", 0)),
    })
    logger.info("VegasInsider MLB props ingest complete: %s", summary)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest VegasInsider's MLB player-props grid into mlb_prop_lines.",
    )
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--source-url-like", default=DEFAULT_SOURCE_URL_LIKE,
        help="LIKE pattern selecting the VegasInsider MLB snapshot to parse.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args()
    summary = ingest_vegasinsider_mlb_props(
        db_path=args.db_path,
        source_url_like=args.source_url_like,
        dry_run=args.dry_run,
    )
    print("VegasInsider MLB props ingestion summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
