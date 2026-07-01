"""Historical odds backfill — populate ``betting_lines`` from data we already have.

``betting_lines`` is the table the evaluation backtest reads when run with
``--use-market-lines``. Live odds-API history is not available here, so the
data-driven source is the props we already scrape into ``web_prop_cards``:
for each (player, stat, book, game-day) we take the latest observed line and
write it as a ``betting_lines`` row. DFS pickem boards have no posted odds, so
over/under default to the -110 breakeven (matching how the edge scanner prices
DFS lines).

Idempotent: rows go through ``DatabaseManager.insert_betting_lines_records``,
which skips exact duplicates and drops implausible lines — re-running the
backfill never double-counts.

CLI:
    .venv/bin/python3 -m nba_model.data.historical_odds_backfill \\
        --db-path data/database/nba_data.db --lookback-hours 168

See README "Production defaults" for the re-run + sweep workflow.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.scrapers.player_names import normalize_name_key

logger = logging.getLogger("nba_model.historical_odds_backfill")

DEFAULT_DB_PATH = "data/database/nba_data.db"
DEFAULT_ODDS = -110


def _build_name_to_player_id(db: DatabaseManager) -> dict[str, int]:
    """Map a normalized player-name key → player_id.

    Prefers ``nba_active_players_ref`` (scraper-synced, ~530 rows) and falls
    back to ``players``. Keyed by ``normalize_name_key`` for tolerant matching.
    """
    mapping: dict[str, int] = {}
    # players first, ref second so the synced ref wins on conflict.
    for sql in (
        "SELECT player_id, name AS player_name FROM players WHERE player_id IS NOT NULL",
        "SELECT player_id, player_name FROM nba_active_players_ref WHERE player_id IS NOT NULL",
    ):
        try:
            rows = db.conn.execute(sql).fetchall()
        except Exception:  # noqa: BLE001 — table may be absent on a bare DB
            continue
        for pid, name in rows:
            key = normalize_name_key(str(name or ""))
            if key and pid is not None:
                mapping[key] = int(pid)
    return mapping


def _latest_web_prop_lines(
    db: DatabaseManager,
    lookback_hours: Optional[float],
    books: Optional[list[str]],
) -> list[dict]:
    """Latest line per (player, stat, book, game-day) from ``web_prop_cards``."""
    where = ["line_value IS NOT NULL"]
    params: list = []
    if lookback_hours is not None:
        where.append("observed_at_utc >= datetime('now', ?)")
        params.append(f"-{float(lookback_hours)} hours")
    if books:
        placeholders = ",".join("?" * len(books))
        where.append(f"lower(book) IN ({placeholders})")
        params.extend([b.strip().lower() for b in books])
    where_sql = " AND ".join(where)
    sql = f"""
        WITH ranked AS (
            SELECT
                player_name, stat_type, book, line_value,
                DATE(observed_at_utc) AS game_date,
                ROW_NUMBER() OVER (
                    PARTITION BY lower(player_name), lower(stat_type),
                                 lower(book), DATE(observed_at_utc)
                    ORDER BY observed_at_utc DESC, card_id DESC
                ) AS rn
            FROM web_prop_cards
            WHERE {where_sql}
        )
        SELECT player_name, stat_type, book, line_value, game_date
        FROM ranked WHERE rn = 1
    """
    cur = db.conn.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def backfill_betting_lines_from_web_prop_cards(
    db_path: str = DEFAULT_DB_PATH,
    lookback_hours: Optional[float] = None,
    default_odds: int = DEFAULT_ODDS,
    books: Optional[list[str]] = None,
    dry_run: bool = False,
) -> dict:
    """Derive ``betting_lines`` rows from scraped ``web_prop_cards``.

    Returns a summary dict with candidate / resolved / inserted counts.
    """
    with DatabaseManager(db_path=db_path) as db:
        name_to_id = _build_name_to_player_id(db)
        candidates = _latest_web_prop_lines(db, lookback_hours, books)

        records: list[dict] = []
        unresolved: set[str] = set()
        for cand in candidates:
            pid = name_to_id.get(normalize_name_key(cand["player_name"]))
            if pid is None:
                unresolved.add(cand["player_name"])
                continue
            records.append(
                {
                    "player_id": pid,
                    "game_date": cand["game_date"],
                    "book": cand["book"],
                    "stat_type": str(cand["stat_type"]).lower(),
                    "line_value": float(cand["line_value"]),
                    "over_odds": int(default_odds),
                    "under_odds": int(default_odds),
                }
            )

        summary = {
            "source": "web_prop_cards",
            "candidates": len(candidates),
            "resolved": len(records),
            "unresolved_players": len(unresolved),
            "dry_run": bool(dry_run),
        }
        if dry_run:
            summary.update({"inserted": 0, "duplicates_ignored": 0})
            logger.info("Backfill dry-run: %s", summary)
            return summary

        insert = db.insert_betting_lines_records(records)

    summary.update(
        {
            "inserted": int(insert.get("inserted", 0)),
            "duplicates_ignored": int(insert.get("duplicates_ignored", 0)),
            "rejected_implausible": int(insert.get("rejected_implausible", 0)),
        }
    )
    logger.info("Backfill complete: %s", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill betting_lines from scraped web_prop_cards.",
    )
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--lookback-hours", type=float, default=None,
        help="Only use web_prop_cards observed within this window (default: all).",
    )
    parser.add_argument(
        "--default-odds", type=int, default=DEFAULT_ODDS,
        help="American odds to assign over/under (DFS boards post none).",
    )
    parser.add_argument(
        "--books", nargs="*", default=None,
        help="Restrict to these books (default: all books present).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args()
    summary = backfill_betting_lines_from_web_prop_cards(
        db_path=args.db_path,
        lookback_hours=args.lookback_hours,
        default_odds=args.default_odds,
        books=args.books,
        dry_run=args.dry_run,
    )
    print("Historical odds backfill summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
