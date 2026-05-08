"""Generate a human-readable inventory of what's actually in the SQLite DB.

Run any time to refresh `data/DATABASE_INVENTORY.txt`:

    .venv/bin/python3 -m nba_model.data.audit_db

The output is overwrite-in-place; commit the file if you want a snapshot of
DB state at a point in time. Sections:

- File metadata + generated_at timestamp
- Tables + row counts
- game_logs coverage by team (parsed from `matchup`) and date range
- Top players by game count (useful for chart UI)
- betting_lines + betting_line_snapshots: book x stat matrix
- predictions: per stat_type
- team_defense: seasons covered
- web_text_snapshots: per URL
- web_prop_cards: per book
- Known data-quality flags (e.g., empty `players.team`)
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

DEFAULT_DB_PATH = "data/database/nba_data.db"
DEFAULT_OUTPUT_PATH = "data/DATABASE_INVENTORY.txt"

LINE_WIDTH = 80


def _section(title: str) -> list[str]:
    return ["", "=" * LINE_WIDTH, f"== {title}", "=" * LINE_WIDTH]


def _h2(title: str) -> list[str]:
    return ["", f"-- {title}"]


def _table_lines(con: sqlite3.Connection, query: str,
                 headers: tuple[str, ...],
                 widths: tuple[int, ...] | None = None,
                 params: tuple = ()) -> list[str]:
    rows = list(con.execute(query, params))
    if not rows:
        return ["  (no rows)"]
    if widths is None:
        widths = tuple(max(len(h), 12) for h in headers)
    out = []
    header = "  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    out.append(header)
    out.append("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        cells = []
        for v, w in zip(row, widths):
            text = "" if v is None else str(v)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                cells.append(text.rjust(w))
            else:
                cells.append(text.ljust(w))
        out.append("  " + "  ".join(cells))
    return out


def _scalar(con: sqlite3.Connection, query: str, params: tuple = ()):
    row = con.execute(query, params).fetchone()
    return row[0] if row else None


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    return _scalar(
        con,
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ) is not None


def build_report(db_path: str) -> str:
    """Return the full inventory report as a single string."""
    p = Path(db_path)
    if not p.exists():
        return f"DB not found: {db_path}\n"

    out: list[str] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    file_size = p.stat().st_size
    out.append("# NBA Probability Model - Database Inventory")
    out.append(
        "# Auto-generated. Re-run with: "
        ".venv/bin/python3 -m nba_model.data.audit_db"
    )
    out.append("")
    out.append(f"db_path:        {db_path}")
    out.append(f"size_bytes:     {file_size:,}")
    out.append(f"size_mb:        {file_size / (1024 * 1024):.2f}")
    out.append(f"generated_utc:  {now}")

    con = sqlite3.connect(db_path)
    try:
        # ----- Tables + row counts ------------------------------------
        out.extend(_section("Tables and row counts"))
        out.extend(_table_lines(
            con,
            """
            SELECT name,
                   (SELECT COUNT(*) FROM sqlite_master AS s2
                      WHERE s2.type='index' AND s2.tbl_name=s.name) AS idx
            FROM sqlite_master AS s
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """,
            headers=("table", "indexes"),
            widths=(30, 8),
        ))
        out.append("")
        out.append("  row counts:")
        for (name,) in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ):
            try:
                n = _scalar(con, f'SELECT COUNT(*) FROM "{name}"')
                out.append(f"    {name:<32} {int(n or 0):>10,}")
            except sqlite3.Error as exc:
                out.append(f"    {name:<32} ERROR: {exc}")

        # ----- game_logs ---------------------------------------------
        if _table_exists(con, "game_logs"):
            out.extend(_section("game_logs: coverage"))
            n_logs = _scalar(con, "SELECT COUNT(*) FROM game_logs") or 0
            n_players = _scalar(
                con, "SELECT COUNT(DISTINCT player_id) FROM game_logs") or 0
            d_min = _scalar(con, "SELECT MIN(game_date) FROM game_logs")
            d_max = _scalar(con, "SELECT MAX(game_date) FROM game_logs")
            out.append(f"  total game-rows:   {int(n_logs):,}")
            out.append(f"  distinct players:  {int(n_players):,}")
            out.append(f"  date range:        {d_min}  ->  {d_max}")

            out.extend(_h2("By team (parsed from matchup column)"))
            out.extend(_table_lines(
                con,
                """
                SELECT upper(trim(substr(matchup, 1,
                                         instr(matchup, ' ') - 1))) AS team,
                       COUNT(*) AS games,
                       COUNT(DISTINCT player_id) AS players,
                       MIN(game_date) AS earliest,
                       MAX(game_date) AS latest
                FROM game_logs
                WHERE matchup IS NOT NULL AND instr(matchup, ' ') > 0
                GROUP BY team
                ORDER BY games DESC
                """,
                headers=("team", "games", "players", "earliest", "latest"),
                widths=(6, 7, 8, 22, 22),
            ))

            out.extend(_h2("Top 30 players by games logged"))
            out.extend(_table_lines(
                con,
                """
                SELECT g.player_id,
                       COALESCE(p.name, '?') AS player_name,
                       COUNT(*) AS games,
                       MIN(g.game_date) AS earliest,
                       MAX(g.game_date) AS latest
                FROM game_logs g
                LEFT JOIN players p ON p.player_id = g.player_id
                GROUP BY g.player_id, p.name
                ORDER BY games DESC
                LIMIT 30
                """,
                headers=("player_id", "name", "games", "earliest", "latest"),
                widths=(10, 28, 6, 22, 22),
            ))

        # ----- betting_lines ------------------------------------------
        if _table_exists(con, "betting_lines"):
            out.extend(_section("betting_lines"))
            d_min = _scalar(con, "SELECT MIN(game_date) FROM betting_lines")
            d_max = _scalar(con, "SELECT MAX(game_date) FROM betting_lines")
            n_rows = _scalar(con, "SELECT COUNT(*) FROM betting_lines") or 0
            n_books = _scalar(
                con, "SELECT COUNT(DISTINCT book) FROM betting_lines") or 0
            n_players = _scalar(
                con, "SELECT COUNT(DISTINCT player_id) FROM betting_lines") or 0
            out.append(f"  rows:              {int(n_rows):,}")
            out.append(f"  distinct books:    {int(n_books)}")
            out.append(f"  distinct players:  {int(n_players)}")
            out.append(f"  date range:        {d_min}  ->  {d_max}")

            out.extend(_h2("Rows per book"))
            out.extend(_table_lines(
                con,
                "SELECT book, COUNT(*) AS rows FROM betting_lines "
                "GROUP BY book ORDER BY rows DESC",
                headers=("book", "rows"),
                widths=(20, 8),
            ))

            out.extend(_h2("Rows per stat_type"))
            out.extend(_table_lines(
                con,
                "SELECT stat_type, COUNT(*) AS rows FROM betting_lines "
                "GROUP BY stat_type ORDER BY rows DESC",
                headers=("stat_type", "rows"),
                widths=(15, 8),
            ))

            out.extend(_h2("book x stat_type matrix"))
            out.extend(_table_lines(
                con,
                """
                SELECT book, stat_type, COUNT(*) AS rows
                FROM betting_lines
                GROUP BY book, stat_type
                ORDER BY book, rows DESC
                """,
                headers=("book", "stat_type", "rows"),
                widths=(20, 15, 8),
            ))

        # ----- betting_line_snapshots --------------------------------
        if _table_exists(con, "betting_line_snapshots"):
            out.extend(_section("betting_line_snapshots"))
            ts_min = _scalar(
                con, "SELECT MIN(snapshot_ts_utc) FROM betting_line_snapshots")
            ts_max = _scalar(
                con, "SELECT MAX(snapshot_ts_utc) FROM betting_line_snapshots")
            n_rows = _scalar(
                con, "SELECT COUNT(*) FROM betting_line_snapshots") or 0
            n_books = _scalar(
                con, "SELECT COUNT(DISTINCT book) FROM betting_line_snapshots") or 0
            out.append(f"  rows:              {int(n_rows):,}")
            out.append(f"  distinct books:    {int(n_books)}")
            out.append(f"  snapshot ts range: {ts_min}  ->  {ts_max}")

            out.extend(_h2("Rows per book"))
            out.extend(_table_lines(
                con,
                "SELECT book, COUNT(*) AS rows FROM betting_line_snapshots "
                "GROUP BY book ORDER BY rows DESC",
                headers=("book", "rows"),
                widths=(20, 8),
            ))

        # ----- predictions ------------------------------------------
        if _table_exists(con, "predictions"):
            out.extend(_section("predictions"))
            n_rows = _scalar(con, "SELECT COUNT(*) FROM predictions") or 0
            n_players = _scalar(
                con, "SELECT COUNT(DISTINCT player_id) FROM predictions") or 0
            out.append(f"  rows:              {int(n_rows):,}")
            out.append(f"  distinct players:  {int(n_players)}")
            out.extend(_h2("Per stat_type"))
            out.extend(_table_lines(
                con,
                "SELECT stat_type, COUNT(*) AS rows FROM predictions "
                "GROUP BY stat_type ORDER BY rows DESC",
                headers=("stat_type", "rows"),
                widths=(15, 8),
            ))

        # ----- team_defense -----------------------------------------
        if _table_exists(con, "team_defense"):
            out.extend(_section("team_defense"))
            n_rows = _scalar(con, "SELECT COUNT(*) FROM team_defense") or 0
            n_seasons = _scalar(
                con, "SELECT COUNT(DISTINCT season) FROM team_defense") or 0
            out.append(f"  rows:              {int(n_rows):,}")
            out.append(f"  distinct seasons:  {int(n_seasons)}")
            out.extend(_h2("By season"))
            out.extend(_table_lines(
                con,
                "SELECT season, COUNT(*) AS teams FROM team_defense "
                "GROUP BY season ORDER BY season DESC",
                headers=("season", "teams"),
                widths=(10, 8),
            ))

        # ----- web_text_snapshots -----------------------------------
        if _table_exists(con, "web_text_snapshots"):
            out.extend(_section("web_text_snapshots"))
            n_rows = _scalar(
                con, "SELECT COUNT(*) FROM web_text_snapshots") or 0
            n_urls = _scalar(
                con, "SELECT COUNT(DISTINCT source_url) "
                "FROM web_text_snapshots") or 0
            out.append(f"  rows:              {int(n_rows):,}")
            out.append(f"  distinct urls:     {int(n_urls)}")
            out.extend(_h2("Rows per source URL"))
            out.extend(_table_lines(
                con,
                "SELECT source_url, COUNT(*) AS rows, "
                "MAX(fetched_at_utc) AS last_fetch "
                "FROM web_text_snapshots GROUP BY source_url ORDER BY rows DESC",
                headers=("url", "rows", "last_fetch"),
                widths=(60, 6, 22),
            ))

        # ----- web_prop_cards ---------------------------------------
        if _table_exists(con, "web_prop_cards"):
            out.extend(_section("web_prop_cards"))
            n_rows = _scalar(con, "SELECT COUNT(*) FROM web_prop_cards") or 0
            out.append(f"  rows:              {int(n_rows):,}")
            out.extend(_h2("Per book"))
            out.extend(_table_lines(
                con,
                "SELECT book, COUNT(*) AS rows FROM web_prop_cards "
                "GROUP BY book ORDER BY rows DESC",
                headers=("book", "rows"),
                widths=(20, 8),
            ))
            out.extend(_h2("Per stat_type"))
            out.extend(_table_lines(
                con,
                "SELECT stat_type, COUNT(*) AS rows FROM web_prop_cards "
                "GROUP BY stat_type ORDER BY rows DESC",
                headers=("stat_type", "rows"),
                widths=(15, 8),
            ))

        # ----- players ----------------------------------------------
        if _table_exists(con, "players"):
            out.extend(_section("players"))
            n_rows = _scalar(con, "SELECT COUNT(*) FROM players") or 0
            n_with_team = _scalar(
                con, "SELECT COUNT(*) FROM players WHERE team IS NOT NULL "
                "AND TRIM(team) <> ''") or 0
            out.append(f"  rows:              {int(n_rows):,}")
            out.append(f"  with team set:     {int(n_with_team):,}")
            if n_with_team:
                out.extend(_h2("Players per team (where set)"))
                out.extend(_table_lines(
                    con,
                    "SELECT team, COUNT(*) AS players FROM players "
                    "WHERE team IS NOT NULL AND TRIM(team) <> '' "
                    "GROUP BY team ORDER BY players DESC",
                    headers=("team", "players"),
                    widths=(6, 8),
                ))

        # ----- nba_active_players_ref -------------------------------
        if _table_exists(con, "nba_active_players_ref"):
            out.extend(_section("nba_active_players_ref"))
            n_rows = _scalar(
                con, "SELECT COUNT(*) FROM nba_active_players_ref") or 0
            last_synced = _scalar(
                con, "SELECT MAX(synced_at_utc) FROM nba_active_players_ref")
            out.append(f"  rows:              {int(n_rows):,}")
            out.append(f"  last synced:       {last_synced}")

        # ----- Data-quality flags -----------------------------------
        out.extend(_section("Known data-quality flags"))
        flags = []
        if _table_exists(con, "players"):
            n_total = _scalar(con, "SELECT COUNT(*) FROM players") or 0
            n_with_team = _scalar(
                con, "SELECT COUNT(*) FROM players "
                "WHERE team IS NOT NULL AND TRIM(team) <> ''") or 0
            if n_with_team < n_total:
                flags.append(
                    f"players.team is empty for {n_total - n_with_team} of "
                    f"{n_total} rows. The Player Charts team-filter dropdown "
                    "reads from this column, so almost every team appears empty. "
                    "Backfill from game_logs.matchup."
                )
        if _table_exists(con, "betting_lines"):
            n_with_pid = _scalar(
                con, "SELECT COUNT(*) FROM betting_lines "
                "WHERE player_id IS NOT NULL") or 0
            n_total = _scalar(con, "SELECT COUNT(*) FROM betting_lines") or 0
            if n_with_pid < n_total:
                flags.append(
                    f"betting_lines.player_id NULL for {n_total - n_with_pid} of "
                    f"{n_total} rows -- those lines won't join to game_logs."
                )
        if _table_exists(con, "predictions"):
            stat_rows = list(con.execute(
                "SELECT DISTINCT stat_type FROM predictions"))
            stat_types = {r[0] for r in stat_rows}
            expected_stats = {
                "points", "assists", "rebounds", "pra",
                "three_pointers_made", "field_goals_made",
            }
            missing = expected_stats - stat_types
            if missing:
                flags.append(
                    f"predictions covers only {sorted(stat_types)} -- missing "
                    f"{sorted(missing)}. line_comparison + monthly diagnostics "
                    "will be sparse for those stats."
                )
        if not flags:
            out.append("  (none detected)")
        for i, msg in enumerate(flags, 1):
            out.append(f"  {i}. {msg}")

    finally:
        con.close()

    out.append("")
    out.append("# end of inventory")
    return "\n".join(out) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH,
                        help="SQLite DB path (default: %(default)s)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH,
                        help="Output txt file (default: %(default)s)")
    parser.add_argument("--stdout", action="store_true",
                        help="Also print the full report to stdout")
    parser.add_argument("--no-write", action="store_true",
                        help="Skip writing the output file (with --stdout)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = build_report(args.db_path)

    if not args.no_write:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        size = os.path.getsize(out_path)
        print(f"Wrote {out_path}  ({size:,} bytes)")
    if args.stdout:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
