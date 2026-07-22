"""Ingest VegasInsider's cross-book odds grid into ``betting_lines``.

VegasInsider republishes a public NBA player-prop odds grid — real American
odds from ~11 books. This module reads the stored ``web_text_snapshots`` row
for the VegasInsider props page, parses it with
``nba_model.scrapers.vegasinsider.extract_odds_rows``, resolves each player to
a ``player_id``, and writes one ``betting_lines`` row per (player, book, stat)
over-cell.

Design mirrors ``historical_odds_backfill``:
  * player_id resolution reuses that module's tolerant name→id map;
  * rows go through ``DatabaseManager.insert_betting_lines_records``, which
    applies ``is_plausible_betting_line`` (data-poisoning defense) and skips
    exact duplicates — re-running never double-counts.

Provenance: every row is attributed to the UNDERLYING book (``book``) but
tagged ``source='vegasinsider'`` so an aggregator-lifted line stays
distinguishable from a direct scrape. Cells are over-only, so ``under_odds``
is always NULL (we never invent an under price) — meaning these rows feed
``--use-market-lines`` and cross-book line-shopping, but cannot by themselves
flag a two-way arb.

CLI::

    .venv/bin/python3 -m nba_model.data.vegasinsider_odds_ingestion \\
        --db-path data/database/nba_data.db
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.data.historical_odds_backfill import _build_name_to_player_id
from nba_model.scrapers.player_names import normalize_name_key
from nba_model.scrapers.vegasinsider import extract_odds_rows

logger = logging.getLogger("nba_model.vegasinsider_odds_ingestion")

DEFAULT_DB_PATH = "data/database/nba_data.db"
DEFAULT_SOURCE_URL_LIKE = "%vegasinsider.com/nba/odds/player-props%"
SOURCE_TAG = "vegasinsider"


def _latest_snapshot(
    db: DatabaseManager,
    source_url_like: str,
) -> Optional[dict]:
    """Newest ``web_text_snapshots`` row whose URL matches ``source_url_like``."""
    row = db.conn.execute(
        """
        SELECT source_url, fetched_at_utc, text_content
        FROM web_text_snapshots
        WHERE source_url LIKE ?
        ORDER BY fetched_at_utc DESC
        LIMIT 1
        """,
        (source_url_like,),
    ).fetchone()
    if not row:
        return None
    return {
        "source_url": row[0],
        "fetched_at_utc": row[1],
        "text_content": row[2] or "",
    }


def _game_date_from_fetched(fetched_at_utc: Optional[str]) -> Optional[str]:
    """The slate day = the ``YYYY-MM-DD`` prefix of the snapshot timestamp.

    The grid has no per-row date; it's a single day's board, so the fetch day
    is the game day (same convention as ``historical_odds_backfill``, which
    uses ``DATE(observed_at_utc)``)."""
    if not fetched_at_utc:
        return None
    raw = str(fetched_at_utc).strip()
    # ISO ("2026-05-08T19:41:30+00:00") or SQLite ("2026-05-08 19:41:30").
    head = raw.replace("T", " ").split(" ", 1)[0]
    return head or None


def ingest_vegasinsider_odds(
    db_path: str = DEFAULT_DB_PATH,
    source_url_like: str = DEFAULT_SOURCE_URL_LIKE,
    snapshot_text: Optional[str] = None,
    game_date: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Parse the latest VegasInsider snapshot and store real odds.

    ``snapshot_text`` / ``game_date`` override the DB lookup (used by tests).
    Returns a summary dict with parsed / resolved / inserted counts.
    """
    with DatabaseManager(db_path=db_path) as db:
        if snapshot_text is None:
            snap = _latest_snapshot(db, source_url_like)
            if snap is None:
                return {
                    "source": SOURCE_TAG,
                    "status": "no_snapshot",
                    "parsed_rows": 0, "resolved": 0, "inserted": 0,
                }
            snapshot_text = snap["text_content"]
            if game_date is None:
                game_date = _game_date_from_fetched(snap["fetched_at_utc"])

        parsed = extract_odds_rows(snapshot_text)
        name_to_id = _build_name_to_player_id(db)

        records: list[dict] = []
        unresolved: set[str] = set()
        for r in parsed:
            pid = name_to_id.get(normalize_name_key(r["player_name"]))
            if pid is None:
                unresolved.add(r["player_name"])
                continue
            records.append({
                "player_id": pid,
                "game_date": game_date,
                "book": r["book"],
                "stat_type": r["stat_type"],
                "line_value": float(r["line_value"]),
                "over_odds": int(r["over_odds"]),
                "under_odds": None,  # over-only grid — never invent an under
                "source": SOURCE_TAG,
            })

        summary = {
            "source": SOURCE_TAG,
            "status": "success",
            "game_date": game_date,
            "parsed_rows": len(parsed),
            "resolved": len(records),
            "unresolved_players": len(unresolved),
            "distinct_players": len({r["player_name"] for r in parsed}),
            "dry_run": bool(dry_run),
        }
        if dry_run or not records:
            summary.update({"inserted": 0, "duplicates_ignored": 0})
            logger.info("VegasInsider ingest (%s): %s",
                        "dry-run" if dry_run else "no records", summary)
            return summary

        insert = db.insert_betting_lines_records(records)

    summary.update({
        "inserted": int(insert.get("inserted", 0)),
        "duplicates_ignored": int(insert.get("duplicates_ignored", 0)),
        "rejected_implausible": int(insert.get("rejected_implausible", 0)),
    })
    logger.info("VegasInsider ingest complete: %s", summary)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest VegasInsider's cross-book odds grid into betting_lines.",
    )
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--source-url-like", default=DEFAULT_SOURCE_URL_LIKE,
        help="LIKE pattern selecting the VegasInsider snapshot to parse.",
    )
    parser.add_argument(
        "--game-date", default=None,
        help="Override the slate day (default: the snapshot's fetch date).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args()
    summary = ingest_vegasinsider_odds(
        db_path=args.db_path,
        source_url_like=args.source_url_like,
        game_date=args.game_date,
        dry_run=args.dry_run,
    )
    print("VegasInsider odds ingestion summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
