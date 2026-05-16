"""SQLite database manager for NBA data, betting lines, and predictions."""
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_int(value):
    """Coerce ``value`` to int, returning None on failure or NaN."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return int(f)


def _safe_float(value):
    """Coerce ``value`` to float, returning None on failure or NaN."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f


def _american_to_implied_prob(odds):
    """Convert American odds to implied probability in (0, 1).

    + odds → 100 / (odds + 100)        e.g. +150 → 0.40
    - odds → -odds / (-odds + 100)     e.g. -110 → 0.524
    """
    o = _safe_float(odds)
    if o is None or o == 0:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return -o / (-o + 100.0)


def _implied_prob_to_american(prob):
    """Convert implied probability in (0, 1) back to American odds.

    p <= 0.5 → +(100 * (1-p)/p)          underdog
    p >  0.5 → -(100 * p/(1-p))          favorite
    """
    p = _safe_float(prob)
    if p is None or not (0.0 < p < 1.0):
        return None
    if p <= 0.5:
        return int(round(100.0 * (1.0 - p) / p))
    return int(round(-100.0 * p / (1.0 - p)))


class DatabaseManager:
    """Manages all database operations for NBA data."""

    def __init__(self, db_path='data/database/nba_data.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._initialize_database()

    def _initialize_database(self):
        """Create database and tables if they don't exist."""
        self.conn = sqlite3.connect(self.db_path)

        # Read and execute schema
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, "r", encoding="utf-8") as f:
            self.conn.executescript(f.read())

        logger.info("Database initialized at %s", self.db_path)

    def get_team_recent_avg_total(
        self,
        team_abbrev: str,
        n_games: int = 20,
    ) -> Optional[float]:
        """Average game total (the team's pts + opp_pts) over the last N games.

        Used by ``simulation.blend_team_prior`` as the baseline against which
        the cross-book ``implied_team_total`` is compared.  Returns ``None``
        when the team has no recent games in ``games``.
        """
        if not team_abbrev:
            return None
        rows = self.conn.execute(
            """
            SELECT pts, opp_pts FROM games
            WHERE upper(team_abbrev) = upper(?)
              AND pts IS NOT NULL AND opp_pts IS NOT NULL
            ORDER BY game_date DESC
            LIMIT ?
            """,
            (team_abbrev, int(max(1, n_games))),
        ).fetchall()
        if not rows:
            return None
        totals = [float(p) + float(o) for p, o in rows
                  if p is not None and o is not None]
        if not totals:
            return None
        return sum(totals) / len(totals)

    def upsert_team_priors(self, records):
        """Upsert reverse-engineered priors (one row per matchup)."""
        if not records:
            return {"upserted": 0, "attempted": 0}
        query = """
            INSERT OR REPLACE INTO team_priors (
                away_team, home_team, computed_at_utc,
                consensus_total, home_spread, away_spread,
                home_team_total, away_team_total,
                home_win_prob_devig, away_win_prob_devig,
                pace_factor, n_books, latest_observed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        payload = []
        for r in records:
            payload.append((
                r.get("away_team"), r.get("home_team"),
                r.get("computed_at_utc"),
                _safe_float(r.get("consensus_total")),
                _safe_float(r.get("home_spread")),
                _safe_float(r.get("away_spread")),
                _safe_float(r.get("home_team_total")),
                _safe_float(r.get("away_team_total")),
                _safe_float(r.get("home_win_prob_devig")),
                _safe_float(r.get("away_win_prob_devig")),
                _safe_float(r.get("pace_factor")),
                _safe_int(r.get("n_books")),
                r.get("latest_observed_at"),
            ))
        if not payload:
            return {"upserted": 0, "attempted": 0}
        before = self.conn.total_changes
        self.conn.executemany(query, payload)
        self.conn.commit()
        upserted = self.conn.total_changes - before
        return {"upserted": int(upserted), "attempted": int(len(payload))}

    def get_team_prior(self, away_team: str, home_team: str):
        """Look up the latest prior row for a matchup; returns None if absent."""
        row = self.conn.execute(
            """
            SELECT away_team, home_team, computed_at_utc,
                   consensus_total, home_spread, away_spread,
                   home_team_total, away_team_total,
                   home_win_prob_devig, away_win_prob_devig,
                   pace_factor, n_books, latest_observed_at
            FROM team_priors
            WHERE lower(away_team) = lower(?)
              AND lower(home_team) = lower(?)
            """,
            (away_team, home_team),
        ).fetchone()
        if row is None:
            return None
        keys = ["away_team", "home_team", "computed_at_utc",
                "consensus_total", "home_spread", "away_spread",
                "home_team_total", "away_team_total",
                "home_win_prob_devig", "away_win_prob_devig",
                "pace_factor", "n_books", "latest_observed_at"]
        return dict(zip(keys, row))

    def backfill_predictions_outcomes(self):
        """Settle every pending ``predictions`` row whose game now has logs.

        Looks up each pending prediction's ``(player_id, game_date)`` in
        ``game_logs``, computes ``actual_result`` for the stat the
        prediction was made on (points / rebounds / assists / pra / ra /
        three_pointers_made / field_goals_made / minutes), then assigns
        ``outcome`` ∈ {'over', 'under', 'push'} based on ``line_value``.

        Returns counts of how many were settled and how many remain pending.
        Idempotent — re-running just settles any new games that landed
        since the last call.
        """
        # Map stat_type → SQL expression against game_logs columns. PRA / RA
        # are computed from the raw columns, others are direct.
        stat_expr = {
            "points":              "g.points",
            "assists":             "g.assists",
            "rebounds":            "g.rebounds",
            "three_pointers_made": "g.fg3m",
            "field_goals_made":    "g.fgm",
            "minutes":             "g.minutes",
            "pra":                 ("COALESCE(g.points, 0) + "
                                    "COALESCE(g.rebounds, 0) + "
                                    "COALESCE(g.assists, 0)"),
            "ra":                  ("COALESCE(g.rebounds, 0) + "
                                    "COALESCE(g.assists, 0)"),
            "steals":              "g.steals",
            "blocks":              "g.blocks",
            "turnovers":           "g.turnovers",
        }

        update_stmt = """
            UPDATE predictions
               SET actual_result = ?,
                   outcome = ?
             WHERE prediction_id = ?
        """

        settled = 0
        scanned = 0
        unsupported_stats: set[str] = set()

        # Pull pending rows up front so the UPDATEs don't fight a live cursor.
        pending = self.conn.execute(
            """
            SELECT prediction_id, player_id, game_date, stat_type, line_value
            FROM predictions
            WHERE actual_result IS NULL
            """
        ).fetchall()

        for pred_id, player_id, game_date, stat_type, line_value in pending:
            scanned += 1
            stat = (stat_type or "").strip().lower()
            expr = stat_expr.get(stat)
            if expr is None:
                unsupported_stats.add(stat or "<empty>")
                continue
            row = self.conn.execute(
                f"""
                SELECT {expr} AS actual
                FROM game_logs g
                WHERE g.player_id = ?
                  AND DATE(g.game_date) = DATE(?)
                LIMIT 1
                """,
                (int(player_id), str(game_date)),
            ).fetchone()
            if not row or row[0] is None:
                continue
            actual = float(row[0])
            if line_value is None:
                outcome = None
            else:
                lv = float(line_value)
                if actual > lv:
                    outcome = "over"
                elif actual < lv:
                    outcome = "under"
                else:
                    outcome = "push"
            self.conn.execute(update_stmt, (actual, outcome, int(pred_id)))
            settled += 1

        self.conn.commit()
        result = {
            "scanned": int(scanned),
            "settled": int(settled),
            "remaining_pending": int(scanned - settled),
        }
        if unsupported_stats:
            result["unsupported_stats"] = sorted(unsupported_stats)
        logger.info(
            "Backfilled predictions outcomes: %s settled, %s still pending "
            "(scanned %s)",
            settled, result["remaining_pending"], scanned,
        )
        return result

    def sync_players_table(self):
        """Backfill the ``players`` table from authoritative sources.

        Why this exists: the snapshot DB had only ~94 rows in ``players``
        with ``players.team`` set for just 1 row.  The chart layer and team
        dropdowns used to read directly from there, which collapsed every
        view to a single team.  The new bulk NBA-API ingest (8K+ team-games,
        91K+ player game logs, 530 active-player reference rows) gives us a
        much richer source — this method fills ``players`` from the union.

        For each ``nba_active_players_ref`` entry we add a row with the
        canonical name and a derived ``team`` (the first whitespace token
        of the player's most-recent ``game_logs.matchup``, if any).
        Existing rows are updated via ``ON CONFLICT(player_id)``.
        """
        # Each player's team derived from their most-recent game_logs row.
        rows = self.conn.execute(
            """
            WITH ranked AS (
                SELECT
                    g.player_id,
                    upper(trim(substr(g.matchup, 1, instr(g.matchup, ' ') - 1))) AS team,
                    ROW_NUMBER() OVER (
                        PARTITION BY g.player_id
                        ORDER BY g.game_date DESC, g.game_log_id DESC
                    ) AS rn
                FROM game_logs g
                WHERE g.matchup IS NOT NULL AND instr(g.matchup, ' ') > 0
            ),
            player_team AS (
                SELECT player_id, team FROM ranked WHERE rn = 1
            )
            SELECT
                r.player_id,
                r.player_name,
                pt.team
            FROM nba_active_players_ref r
            LEFT JOIN player_team pt ON pt.player_id = r.player_id
            """
        ).fetchall()

        if not rows:
            return {"upserted": 0, "attempted": 0}

        # Upsert: insert when new, overwrite name + team on conflict.  The
        # existing ``insert_player`` runs one row at a time; for ~530 rows
        # batching gives no perf win but the SQL is identical.
        query = """
            INSERT INTO players (player_id, name, team)
            VALUES (?, ?, ?)
            ON CONFLICT(player_id) DO UPDATE SET
                name = excluded.name,
                team = excluded.team,
                last_updated = CURRENT_TIMESTAMP
        """
        before = self.conn.total_changes
        self.conn.executemany(query, rows)
        self.conn.commit()
        upserted = self.conn.total_changes - before
        logger.info(
            "sync_players_table: upserted %s rows (%s with team derived)",
            upserted, sum(1 for r in rows if r[2]),
        )
        return {"upserted": int(upserted), "attempted": int(len(rows))}

    def insert_player(self, player_id, name, team=None, position=None):
        """Insert or update player record."""
        # noinspection SqlNoDataSourceInspection
        query = """
            INSERT INTO players (player_id, name, team, position)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(player_id) DO UPDATE SET
                team = excluded.team,
                position = excluded.position,
                last_updated = CURRENT_TIMESTAMP
        """
        self.conn.execute(query, (player_id, name, team, position))
        self.conn.commit()

    def insert_team_defense_records(self, records):
        """
        Upsert team defense records.

        Args:
            records: Iterable of dicts with keys:
                team_abbrev, season, def_rating, opp_ppg, pace
        """
        if not records:
            return

        query = """
            INSERT INTO team_defense (team_abbrev, season, def_rating, opp_ppg, pace)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(team_abbrev) DO UPDATE SET
                season = excluded.season,
                def_rating = excluded.def_rating,
                opp_ppg = excluded.opp_ppg,
                pace = excluded.pace,
                last_updated = CURRENT_TIMESTAMP
        """
        payload = [
            (
                rec.get("team_abbrev"),
                rec.get("season"),
                rec.get("def_rating"),
                rec.get("opp_ppg"),
                rec.get("pace"),
            )
            for rec in records
            if rec.get("team_abbrev")
        ]
        if not payload:
            return
        self.conn.executemany(query, payload)
        self.conn.commit()
        logger.info("Upserted %s team_defense rows", len(payload))

    def insert_betting_lines_records(self, records):
        """
        Insert betting lines while skipping exact duplicates.

        Args:
            records: Iterable of dicts with keys:
                player_id, game_date, book, stat_type, line_value, over_odds, under_odds

        Returns:
            dict: Insert summary with keys inserted, duplicates_ignored, attempted.
        """
        if not records:
            return {
                "inserted": 0,
                "duplicates_ignored": 0,
                "attempted": 0,
            }

        query = """
            INSERT INTO betting_lines
                (player_id, game_date, book, stat_type, line_value, over_odds, under_odds)
            SELECT ?, ?, ?, ?, ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1
                FROM betting_lines bl
                WHERE bl.player_id = ?
                  AND bl.game_date = ?
                  AND bl.book = ?
                  AND bl.stat_type = ?
                  AND bl.line_value = ?
                  AND IFNULL(bl.over_odds, -99999) = IFNULL(?, -99999)
                  AND IFNULL(bl.under_odds, -99999) = IFNULL(?, -99999)
            )
        """

        # SECURITY (data poisoning defense): drop scraper-poisoned rows that
        # claim a structurally implausible line/odds. The scrapers are the
        # only third-party-influenced surface; even one bad row pollutes the
        # consensus mean + EV math for every user. The validator's range is
        # deliberately generous - real outliers pass, deliberate bad data
        # (negative lines, 9999.5 points, nan, inf, odds=0) gets dropped.
        from nba_model.web.input_validation import is_plausible_betting_line

        payload = []
        rejected_implausible = 0
        for rec in records:
            row = (
                rec.get("player_id"),
                rec.get("game_date"),
                rec.get("book"),
                rec.get("stat_type"),
                rec.get("line_value"),
                rec.get("over_odds"),
                rec.get("under_odds"),
            )
            if not all([row[0], row[1], row[2], row[3]]) or row[4] is None:
                continue
            if not is_plausible_betting_line(
                row[3], row[4], over_odds=row[5], under_odds=row[6],
            ):
                rejected_implausible += 1
                continue
            payload.append(row + row)

        if not payload:
            return {
                "inserted": 0,
                "duplicates_ignored": 0,
                "attempted": 0,
            }

        before_changes = self.conn.total_changes
        self.conn.executemany(query, payload)
        self.conn.commit()
        inserted = self.conn.total_changes - before_changes
        ignored = len(payload) - inserted
        if rejected_implausible:
            logger.warning(
                "Dropped %s implausible betting_lines rows (likely scraper "
                "poisoning); see input_validation.is_plausible_betting_line",
                rejected_implausible,
            )
        logger.info(
            "Inserted %s betting_lines rows (%s duplicates ignored, "
            "%s implausible)",
            inserted, ignored, rejected_implausible,
        )
        return {
            "inserted": int(inserted),
            "duplicates_ignored": int(ignored),
            "rejected_implausible": int(rejected_implausible),
            "attempted": int(len(payload)),
        }

    def insert_betting_line_snapshots(self, records):
        """
        Insert betting line snapshots (no de-duplication; time-series storage).

        Args:
            records: Iterable of dicts with keys:
                snapshot_ts_utc, event_id, game_date, player_id,
                book, market_key, stat_type, line_value,
                over_odds, under_odds, raw_payload (optional)
        """
        if not records:
            return {
                "inserted": 0,
                "attempted": 0,
            }

        query = """
            INSERT INTO betting_line_snapshots (
                snapshot_ts_utc,
                event_id,
                game_date,
                player_id,
                book,
                market_key,
                stat_type,
                line_value,
                over_odds,
                under_odds,
                raw_payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        payload = []
        for rec in records:
            row = (
                rec.get("snapshot_ts_utc"),
                rec.get("event_id"),
                rec.get("game_date"),
                rec.get("player_id"),
                rec.get("book"),
                rec.get("market_key"),
                rec.get("stat_type"),
                rec.get("line_value"),
                rec.get("over_odds"),
                rec.get("under_odds"),
                rec.get("raw_payload"),
            )
            # Require minimal core fields.
            if not all([row[0], row[2], row[3], row[4], row[5], row[6]]) or row[7] is None:
                continue
            payload.append(row)

        if not payload:
            return {
                "inserted": 0,
                "attempted": 0,
            }

        before_changes = self.conn.total_changes
        self.conn.executemany(query, payload)
        self.conn.commit()
        inserted = self.conn.total_changes - before_changes
        logger.info("Inserted %s betting_line_snapshots rows", inserted)
        return {
            "inserted": int(inserted),
            "attempted": int(len(payload)),
        }

    def insert_web_text_snapshots(self, records):
        """
        Insert raw text snapshots fetched from direct web URLs.

        Args:
            records: Iterable of dicts with keys:
                source_url, fetched_at_utc, http_status, content_type,
                text_content, text_length, content_sha256
        """
        if not records:
            return {
                "inserted": 0,
                "attempted": 0,
            }

        query = """
            INSERT INTO web_text_snapshots (
                source_url,
                fetched_at_utc,
                http_status,
                content_type,
                text_content,
                text_length,
                content_sha256
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        payload = []
        for rec in records:
            row = (
                rec.get("source_url"),
                rec.get("fetched_at_utc"),
                rec.get("http_status"),
                rec.get("content_type"),
                rec.get("text_content"),
                rec.get("text_length"),
                rec.get("content_sha256"),
            )
            if not row[0] or not row[1] or not row[4]:
                continue
            payload.append(row)

        if not payload:
            return {
                "inserted": 0,
                "attempted": 0,
            }

        before_changes = self.conn.total_changes
        self.conn.executemany(query, payload)
        self.conn.commit()
        inserted = self.conn.total_changes - before_changes
        logger.info("Inserted %s web_text_snapshots rows", inserted)
        return {
            "inserted": int(inserted),
            "attempted": int(len(payload)),
        }

    def get_latest_web_text_fetch_times(self, source_urls):
        """
        Return latest fetched_at_utc per URL from web_text_snapshots.

        Args:
            source_urls: iterable of URLs

        Returns:
            dict[str, str]: source_url -> latest fetched_at_utc
        """
        urls = [
            str(url).strip()
            for url in (source_urls or [])
            if str(url).strip()
        ]
        if not urls:
            return {}

        placeholders = ", ".join(["?"] * len(urls))
        query = f"""
            SELECT source_url, MAX(fetched_at_utc) AS latest_fetched_at_utc
            FROM web_text_snapshots
            WHERE source_url IN ({placeholders})
            GROUP BY source_url
        """
        rows = self.conn.execute(query, tuple(urls)).fetchall()
        return {
            str(row[0]): str(row[1])
            for row in rows
            if row and row[0] is not None and row[1] is not None
        }

    def get_recent_web_text_snapshots(
        self,
        source_urls=None,
        max_snapshots_per_url=1,
        limit_total=250,
    ):
        """
        Return recent web-text snapshots for parser ingestion.

        Args:
            source_urls: optional iterable of URL filters.
            max_snapshots_per_url: max snapshots returned per URL.
            limit_total: hard cap for total snapshots returned.
        """
        per_url_limit = max(1, int(max_snapshots_per_url))
        total_limit = max(1, int(limit_total))
        urls = [
            str(url).strip()
            for url in (source_urls or [])
            if str(url).strip()
        ]

        if urls:
            placeholders = ", ".join(["?"] * len(urls))
            query = f"""
                SELECT snapshot_id, source_url, fetched_at_utc, text_content, text_length, content_sha256
                FROM web_text_snapshots
                WHERE source_url IN ({placeholders})
                ORDER BY source_url ASC, fetched_at_utc DESC, snapshot_id DESC
            """
            rows = self.conn.execute(query, tuple(urls)).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT snapshot_id, source_url, fetched_at_utc, text_content, text_length, content_sha256
                FROM web_text_snapshots
                ORDER BY source_url ASC, fetched_at_utc DESC, snapshot_id DESC
                """
            ).fetchall()

        selected = []
        per_url_counts = {}
        for row in rows:
            source_url = str(row[1]).strip() if row and row[1] is not None else ""
            if not source_url:
                continue
            seen_for_url = per_url_counts.get(source_url, 0)
            if seen_for_url >= per_url_limit:
                continue
            selected.append(
                {
                    "snapshot_id": int(row[0]),
                    "source_url": source_url,
                    "fetched_at_utc": str(row[2]),
                    "text_content": str(row[3]) if row[3] is not None else "",
                    "text_length": int(row[4]) if row[4] is not None else None,
                    "content_sha256": str(row[5]) if row[5] is not None else None,
                }
            )
            per_url_counts[source_url] = seen_for_url + 1
            if len(selected) >= total_limit:
                break

        return selected

    def upsert_active_players_reference(self, records):
        """
        Upsert active NBA players reference rows.

        Args:
            records: Iterable[dict] with keys:
                player_id, player_name, synced_at_utc
        """
        if not records:
            return {
                "attempted": 0,
                "written": 0,
            }

        query = """
            INSERT INTO nba_active_players_ref (player_id, player_name, synced_at_utc)
            VALUES (?, ?, ?)
            ON CONFLICT(player_id) DO UPDATE SET
                player_name = excluded.player_name,
                synced_at_utc = excluded.synced_at_utc
        """
        payload = []
        for rec in records:
            player_id = rec.get("player_id")
            player_name = str(rec.get("player_name", "")).strip()
            synced_at_utc = str(rec.get("synced_at_utc", "")).strip()
            if player_id is None or not player_name or not synced_at_utc:
                continue
            payload.append((player_id, player_name, synced_at_utc))

        if not payload:
            return {
                "attempted": 0,
                "written": 0,
            }

        before_changes = self.conn.total_changes
        self.conn.executemany(query, payload)
        self.conn.commit()
        written = self.conn.total_changes - before_changes
        logger.info("Upserted %s nba_active_players_ref rows", written)
        return {
            "attempted": int(len(payload)),
            "written": int(written),
        }

    def get_active_players_reference_names(self):
        """Return all active NBA player names from reference table."""
        rows = self.conn.execute(
            """
            SELECT player_name
            FROM nba_active_players_ref
            WHERE player_name IS NOT NULL AND trim(player_name) <> ''
            ORDER BY player_name ASC
            """
        ).fetchall()
        return [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]

    def insert_web_prop_cards(self, records):
        """
        Insert parsed web prop cards with dedupe via record_sha256.

        Args:
            records: Iterable[dict] with parser output fields.
        """
        if not records:
            return {
                "inserted": 0,
                "attempted": 0,
            }

        query = """
            INSERT OR IGNORE INTO web_prop_cards (
                snapshot_id,
                source_url,
                book,
                observed_at_utc,
                player_name,
                player_classification,
                stat_type,
                line_value,
                side,
                parse_confidence,
                raw_card_text,
                parser_version,
                record_sha256
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        payload = []
        for rec in records:
            snapshot_id = rec.get("snapshot_id")
            line_value = rec.get("line_value")
            parse_confidence = rec.get("parse_confidence")
            try:
                snapshot_id = int(snapshot_id)
                line_value = float(line_value)
                parse_confidence = float(parse_confidence)
            except (TypeError, ValueError):
                continue

            row = (
                snapshot_id,
                str(rec.get("source_url", "")).strip(),
                str(rec.get("book", "")).strip(),
                str(rec.get("observed_at_utc", "")).strip(),
                str(rec.get("player_name", "")).strip(),
                str(rec.get("player_classification", "")).strip(),
                str(rec.get("stat_type", "")).strip(),
                line_value,
                str(rec.get("side", "")).strip(),
                parse_confidence,
                str(rec.get("raw_card_text", "")).strip() or None,
                str(rec.get("parser_version", "")).strip(),
                str(rec.get("record_sha256", "")).strip(),
            )
            if not all(
                [
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    row[8],
                    row[11],
                    row[12],
                ]
            ):
                continue
            payload.append(row)

        if not payload:
            return {
                "inserted": 0,
                "attempted": 0,
            }

        before_changes = self.conn.total_changes
        self.conn.executemany(query, payload)
        self.conn.commit()
        inserted = self.conn.total_changes - before_changes
        logger.info("Inserted %s web_prop_cards rows", inserted)
        return {
            "inserted": int(inserted),
            "attempted": int(len(payload)),
        }

    def get_consensus_prop_lines(
        self,
        player_name=None,
        stat_type=None,
        side=None,
        since_hours=None,
        min_books=1,
    ):
        """Return cross-book consensus line values from web_prop_cards.

        For each (player, stat, side) the latest line from each book is
        selected (so a single book can't be double-counted), then averaged
        across books.  Returns rows with the mean line, the number of
        contributing books, and the comma-separated book list.

        Args:
            player_name: optional case-insensitive filter on player_name.
            stat_type: optional case-insensitive filter on stat_type.
            side: optional 'over' / 'under' filter; default returns both.
            since_hours: only include cards observed within the last N hours.
            min_books: drop rows with fewer than this many contributing books.
        """
        clauses = ["player_classification = 'active_nba'"]
        params: list = []
        if player_name:
            clauses.append("lower(player_name) = lower(?)")
            params.append(str(player_name).strip())
        if stat_type:
            clauses.append("lower(stat_type) = lower(?)")
            params.append(str(stat_type).strip())
        if side:
            clauses.append("lower(side) = lower(?)")
            params.append(str(side).strip())
        if since_hours is not None:
            try:
                hours = float(since_hours)
            except (TypeError, ValueError):
                hours = None
            if hours and hours > 0:
                clauses.append(
                    "observed_at_utc >= datetime('now', ?)"
                )
                params.append(f"-{hours} hours")
        where_sql = " AND ".join(clauses) if clauses else "1=1"

        query = f"""
            WITH latest_per_book AS (
                SELECT
                    player_name,
                    stat_type,
                    side,
                    book,
                    line_value,
                    observed_at_utc,
                    ROW_NUMBER() OVER (
                        PARTITION BY lower(player_name), lower(stat_type), lower(side), lower(book)
                        ORDER BY observed_at_utc DESC, card_id DESC
                    ) AS rn
                FROM web_prop_cards
                WHERE {where_sql}
            )
            SELECT
                player_name,
                stat_type,
                side,
                AVG(line_value)               AS mean_line,
                MIN(line_value)               AS min_line,
                MAX(line_value)               AS max_line,
                COUNT(DISTINCT lower(book))   AS n_books,
                GROUP_CONCAT(DISTINCT book)   AS books,
                MAX(observed_at_utc)          AS latest_observed_at
            FROM latest_per_book
            WHERE rn = 1
            GROUP BY lower(player_name), lower(stat_type), lower(side)
            HAVING n_books >= ?
            ORDER BY player_name ASC, stat_type ASC, side ASC
        """
        params.append(int(max(1, min_books)))
        rows = self.conn.execute(query, tuple(params)).fetchall()
        return [
            {
                "player_name": row[0],
                "stat_type": row[1],
                "side": row[2],
                "mean_line": float(row[3]) if row[3] is not None else None,
                "min_line": float(row[4]) if row[4] is not None else None,
                "max_line": float(row[5]) if row[5] is not None else None,
                "n_books": int(row[6] or 0),
                "books": str(row[7] or ""),
                "latest_observed_at": row[8],
            }
            for row in rows
        ]

    def insert_web_team_lines(self, records):
        """Insert game-level team lines with dedupe via record_sha256."""
        if not records:
            return {"inserted": 0, "attempted": 0}

        query = """
            INSERT OR IGNORE INTO web_team_lines (
                snapshot_id, source_url, book, observed_at_utc,
                away_team, home_team, market_type, side, team,
                line_value, odds_american,
                parse_confidence, raw_text, parser_version, record_sha256
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        payload = []
        for rec in records:
            try:
                snapshot_id = int(rec.get("snapshot_id"))
                parse_confidence = float(rec.get("parse_confidence"))
            except (TypeError, ValueError):
                continue
            line_value = rec.get("line_value")
            try:
                line_value = float(line_value) if line_value is not None else None
            except (TypeError, ValueError):
                line_value = None
            odds = rec.get("odds_american")
            try:
                odds = int(odds) if odds is not None else None
            except (TypeError, ValueError):
                odds = None

            row = (
                snapshot_id,
                str(rec.get("source_url", "")).strip(),
                str(rec.get("book", "")).strip(),
                str(rec.get("observed_at_utc", "")).strip(),
                str(rec.get("away_team", "")).strip(),
                str(rec.get("home_team", "")).strip(),
                str(rec.get("market_type", "")).strip(),
                str(rec.get("side", "")).strip(),
                (str(rec.get("team", "")).strip() or None),
                line_value,
                odds,
                parse_confidence,
                (str(rec.get("raw_text", "")).strip() or None),
                str(rec.get("parser_version", "")).strip(),
                str(rec.get("record_sha256", "")).strip(),
            )
            # Required: source_url, book, observed_at_utc, both teams,
            # market_type, side, parser_version, record_sha256.
            if not all([row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[13], row[14]]):
                continue
            payload.append(row)

        if not payload:
            return {"inserted": 0, "attempted": 0}

        before = self.conn.total_changes
        self.conn.executemany(query, payload)
        self.conn.commit()
        inserted = self.conn.total_changes - before
        logger.info("Inserted %s web_team_lines rows", inserted)
        return {"inserted": int(inserted), "attempted": int(len(payload))}

    def get_consensus_team_lines(
        self,
        away_team=None,
        home_team=None,
        market_type=None,
        side=None,
        since_hours=None,
        min_books=1,
    ):
        """Return cross-book consensus for game-level markets.

        For each (away_team, home_team, market_type, side) the latest line
        from each book is selected, then averaged across books.  Returns one
        row per (game, market, side) with mean line, mean odds, and the
        list of contributing books.
        """
        clauses: list[str] = []
        params: list = []
        if away_team:
            clauses.append("lower(away_team) = lower(?)")
            params.append(str(away_team).strip())
        if home_team:
            clauses.append("lower(home_team) = lower(?)")
            params.append(str(home_team).strip())
        if market_type:
            clauses.append("lower(market_type) = lower(?)")
            params.append(str(market_type).strip())
        if side:
            clauses.append("lower(side) = lower(?)")
            params.append(str(side).strip())
        if since_hours is not None:
            try:
                hours = float(since_hours)
            except (TypeError, ValueError):
                hours = None
            if hours and hours > 0:
                clauses.append("observed_at_utc >= datetime('now', ?)")
                params.append(f"-{hours} hours")
        where_sql = (" AND ".join(clauses)) if clauses else "1=1"

        # Pull the latest line per (game, market, side, book) and aggregate
        # in Python so we can convert odds → implied probability → mean →
        # American.  Averaging raw American odds is mathematically wrong
        # whenever the sample contains both + and - values (the signs flip
        # the magnitude, distorting the mean).
        query = f"""
            WITH latest_per_book AS (
                SELECT
                    away_team, home_team, market_type, side,
                    book, line_value, odds_american, observed_at_utc,
                    ROW_NUMBER() OVER (
                        PARTITION BY lower(away_team), lower(home_team),
                                     lower(market_type), lower(side), lower(book)
                        ORDER BY observed_at_utc DESC, line_id DESC
                    ) AS rn
                FROM web_team_lines
                WHERE {where_sql}
            )
            SELECT
                away_team, home_team, market_type, side,
                book, line_value, odds_american, observed_at_utc
            FROM latest_per_book
            WHERE rn = 1
            ORDER BY away_team, home_team, market_type, side, book
        """
        rows = self.conn.execute(query, tuple(params)).fetchall()

        # Group rows by (away, home, market, side) and aggregate.
        groups: dict[tuple, dict] = {}
        for away, home, market, side, book, line, odds, obs_at in rows:
            key = (away.lower(), home.lower(), market.lower(), side.lower())
            slot = groups.setdefault(key, {
                "away_team": away, "home_team": home,
                "market_type": market, "side": side,
                "line_values": [], "odds_probs": [], "raw_odds": [],
                "books": set(), "latest_observed_at": obs_at,
            })
            if line is not None:
                slot["line_values"].append(float(line))
            prob = _american_to_implied_prob(odds)
            if prob is not None:
                slot["odds_probs"].append(prob)
                slot["raw_odds"].append(int(odds))
            if book:
                slot["books"].add(book)
            if obs_at and (slot["latest_observed_at"] is None
                           or obs_at > slot["latest_observed_at"]):
                slot["latest_observed_at"] = obs_at

        min_books_int = int(max(1, min_books))
        out: list[dict] = []
        for slot in groups.values():
            n_books = len(slot["books"])
            if n_books < min_books_int:
                continue
            line_vals = slot["line_values"]
            probs = slot["odds_probs"]
            raw_odds = slot["raw_odds"]
            mean_prob = sum(probs) / len(probs) if probs else None
            out.append({
                "away_team": slot["away_team"],
                "home_team": slot["home_team"],
                "market_type": slot["market_type"],
                "side": slot["side"],
                "mean_line": (sum(line_vals) / len(line_vals)
                              if line_vals else None),
                "min_line": min(line_vals) if line_vals else None,
                "max_line": max(line_vals) if line_vals else None,
                # mean_odds is now the American odds corresponding to the
                # *mean implied probability* across books, which is what
                # "average market odds" actually means.  Raw-american mean
                # (e.g. -110, +120, -105 → 35/3) is retained as
                # mean_odds_naive for callers that want to compare.
                "mean_odds": _implied_prob_to_american(mean_prob),
                "mean_implied_prob": mean_prob,
                "mean_odds_naive": (sum(raw_odds) / len(raw_odds)
                                    if raw_odds else None),
                "n_books": n_books,
                "books": ",".join(sorted(slot["books"])),
                "latest_observed_at": slot["latest_observed_at"],
            })
        out.sort(key=lambda r: (
            r["home_team"], r["away_team"], r["market_type"], r["side"],
        ))
        return out

    def insert_games(self, records):
        """Upsert team-game rows (one per team per game).

        ``records`` is an iterable of dicts with the columns from the
        ``games`` table. Existing (game_id, team_id) rows are replaced so
        a fresh nba_api fetch always reflects the latest scores/results.
        """
        if not records:
            return {"inserted": 0, "attempted": 0}
        query = """
            INSERT OR REPLACE INTO games (
                game_id, season, season_type, game_date,
                team_id, team_abbrev, team_name,
                matchup, home_away, opponent_abbrev, result,
                pts, opp_pts, plus_minus,
                fg_pct, fg3_pct, ft_pct,
                rebounds, assists, steals, blocks, turnovers,
                last_updated
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        now = datetime.now(timezone.utc).isoformat()
        payload = []
        for rec in records:
            try:
                game_id = str(rec.get("game_id") or "").strip()
                team_id = int(rec.get("team_id"))
            except (TypeError, ValueError):
                continue
            if not game_id or not team_id:
                continue
            payload.append((
                game_id,
                str(rec.get("season") or "").strip(),
                str(rec.get("season_type") or "").strip(),
                str(rec.get("game_date") or "").strip(),
                team_id,
                str(rec.get("team_abbrev") or "").strip().upper(),
                rec.get("team_name") or None,
                rec.get("matchup") or None,
                rec.get("home_away") or None,
                rec.get("opponent_abbrev") or None,
                rec.get("result") or None,
                _safe_int(rec.get("pts")),
                _safe_int(rec.get("opp_pts")),
                _safe_int(rec.get("plus_minus")),
                _safe_float(rec.get("fg_pct")),
                _safe_float(rec.get("fg3_pct")),
                _safe_float(rec.get("ft_pct")),
                _safe_int(rec.get("rebounds")),
                _safe_int(rec.get("assists")),
                _safe_int(rec.get("steals")),
                _safe_int(rec.get("blocks")),
                _safe_int(rec.get("turnovers")),
                now,
            ))
        if not payload:
            return {"inserted": 0, "attempted": 0}
        before = self.conn.total_changes
        self.conn.executemany(query, payload)
        self.conn.commit()
        inserted = self.conn.total_changes - before
        logger.info("Upserted %s rows into games", inserted)
        return {"inserted": int(inserted), "attempted": int(len(payload))}

    def get_recent_games(
        self,
        n: int = 100,
        season: Optional[str] = None,
        season_type: Optional[str] = None,
        team_abbrev: Optional[str] = None,
    ):
        """Return recent NBA games as one row per matchup (away vs home).

        Joins the two team-rows per game_id back together so callers get a
        single row per game with both team names + scores + winner.
        """
        clauses = []
        params: list = []
        if season:
            clauses.append("a.season = ?")
            params.append(season)
        if season_type:
            clauses.append("a.season_type = ?")
            params.append(season_type)
        if team_abbrev:
            clauses.append("(a.team_abbrev = ? OR h.team_abbrev = ?)")
            tt = team_abbrev.upper()
            params.extend([tt, tt])
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        query = f"""
            SELECT
                a.game_id, a.game_date, a.season, a.season_type,
                a.team_abbrev AS away_abbrev, a.team_name AS away_name,
                a.pts AS away_pts,
                h.team_abbrev AS home_abbrev, h.team_name AS home_name,
                h.pts AS home_pts,
                a.matchup AS matchup,
                CASE
                    WHEN a.pts IS NULL OR h.pts IS NULL THEN NULL
                    WHEN a.pts > h.pts THEN a.team_abbrev
                    WHEN h.pts > a.pts THEN h.team_abbrev
                    ELSE 'TIE'
                END AS winner
            FROM games a
            JOIN games h
              ON h.game_id = a.game_id
             AND h.team_id != a.team_id
             AND a.home_away = 'away'
             AND h.home_away = 'home'
            {where}
            ORDER BY a.game_date DESC, a.game_id DESC
            LIMIT ?
        """
        params.append(int(max(1, n)))
        return [dict(zip(
            ["game_id", "game_date", "season", "season_type",
             "away_abbrev", "away_name", "away_pts",
             "home_abbrev", "home_name", "home_pts",
             "matchup", "winner"],
            row,
        )) for row in self.conn.execute(query, tuple(params)).fetchall()]

    def get_player_recent_results(
        self,
        n: int = 200,
        player_id: Optional[int] = None,
        team_abbrev: Optional[str] = None,
        stat: Optional[str] = None,
        min_value: Optional[float] = None,
        season: Optional[str] = None,
    ):
        """Return recent player game-log rows joined with player names.

        Uses ``nba_active_players_ref`` for the canonical name (530 rows)
        with a fallback to the sparse ``players`` table.  Optional filters
        let the frontend slice by player, team, season, or "show me players
        with at least N points last game".
        """
        clauses = []
        params: list = []
        if player_id is not None:
            clauses.append("g.player_id = ?")
            params.append(int(player_id))
        if team_abbrev:
            clauses.append("upper(trim(substr(g.matchup, 1, instr(g.matchup, ' ') - 1))) = ?")
            params.append(team_abbrev.upper())
        if season:
            clauses.append("g.season = ?")
            params.append(season)
        if stat and min_value is not None:
            allowed = {"points", "rebounds", "assists", "steals", "blocks",
                       "turnovers", "fg3m", "minutes"}
            stat_col = stat.lower()
            if stat_col in allowed:
                clauses.append(f"g.{stat_col} >= ?")
                params.append(float(min_value))
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        query = f"""
            SELECT
                g.player_id,
                COALESCE(r.player_name, p.name, 'Player ' || g.player_id) AS player_name,
                g.game_id, g.game_date, g.season,
                g.matchup, g.home_away, g.result,
                g.minutes, g.points, g.rebounds, g.assists,
                g.steals, g.blocks, g.turnovers,
                g.fgm, g.fga, g.fg3m, g.fg3a, g.ftm, g.fta,
                g.plus_minus
            FROM game_logs g
            LEFT JOIN nba_active_players_ref r ON r.player_id = g.player_id
            LEFT JOIN players p ON p.player_id = g.player_id
            {where}
            ORDER BY g.game_date DESC, g.player_id ASC
            LIMIT ?
        """
        params.append(int(max(1, n)))
        cols = ["player_id", "player_name", "game_id", "game_date", "season",
                "matchup", "home_away", "result",
                "minutes", "points", "rebounds", "assists",
                "steals", "blocks", "turnovers",
                "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "plus_minus"]
        return [dict(zip(cols, row))
                for row in self.conn.execute(query, tuple(params)).fetchall()]

    def insert_game_logs(self, game_logs_df):
        """
        Bulk insert game logs from DataFrame.

        Returns the number of newly-inserted rows (existing duplicates are
        skipped via ``INSERT OR IGNORE``).
        """
        try:
            if game_logs_df is None or game_logs_df.empty:
                return 0

            # Remove duplicates before inserting
            game_logs_df = game_logs_df.drop_duplicates(
                subset=['player_id', 'game_id'])

            # Keep only actual table columns and rely on INSERT OR IGNORE for
            # existing rows already present in SQLite.
            table_columns = {
                row[1]
                for row in self.conn.execute("PRAGMA table_info(game_logs)").fetchall()
            }
            blocked_columns = {'game_log_id', 'created_at'}
            columns = [
                col for col in game_logs_df.columns
                if col in table_columns and col not in blocked_columns
            ]
            if not columns:
                logger.warning("No valid game_logs columns to insert")
                return 0

            payload = game_logs_df[columns].where(
                pd.notna(game_logs_df[columns]), None)
            query = f"""
                INSERT OR IGNORE INTO game_logs ({", ".join(columns)})
                VALUES ({", ".join(["?"] * len(columns))})
            """

            before_changes = self.conn.total_changes
            self.conn.executemany(
                query, payload.itertuples(index=False, name=None))
            self.conn.commit()
            inserted = self.conn.total_changes - before_changes
            ignored = len(payload) - inserted
            logger.info(
                "Inserted %s game logs (%s duplicates ignored)", inserted, ignored
            )
            return int(inserted)
        except Exception as e:
            logger.error("Error inserting game logs: %s", e)
            self.conn.rollback()
            raise

    def get_player_games(self, player_id, n_games=50):
        """Fetch most recent N games for a player."""
        # noinspection SqlNoDataSourceInspection
        query = """
            SELECT *
            FROM game_logs
            WHERE player_id = ?
            ORDER BY game_date DESC
            LIMIT ?
        """
        return pd.read_sql_query(query, self.conn, params=(player_id, n_games))

    def get_games_by_date_range(self, player_id, start_date, end_date):
        """Fetch games within date range (for backtesting)."""
        # noinspection SqlNoDataSourceInspection
        query = """
            SELECT *
            FROM game_logs
            WHERE player_id = ?
              AND game_date BETWEEN ? AND ?
            ORDER BY game_date ASC
        """
        return pd.read_sql_query(query, self.conn, params=(player_id, start_date, end_date))

    def insert_prediction(self, prediction_data):
        """
        Insert a prediction record.

        Args:
            prediction_data: dict with keys:
                player_id, game_date, stat_type, predicted_mean,
                predicted_std, prob_over, line_value, expected_value,
                optional model_config_json
        """
        # noinspection SqlNoDataSourceInspection
        query = """
            INSERT INTO predictions
            (player_id, game_date, stat_type, predicted_mean, predicted_std,
             prob_over, line_value, book_odds, expected_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        values = (
            prediction_data['player_id'],
            prediction_data['game_date'],
            prediction_data['stat_type'],
            prediction_data['predicted_mean'],
            prediction_data['predicted_std'],
            prediction_data['prob_over'],
            prediction_data.get('line_value'),
            prediction_data.get('book_odds'),
            prediction_data.get('expected_value')
        )
        cursor = self.conn.execute(query, values)

        model_config_json = prediction_data.get("model_config_json")
        if model_config_json:
            self.conn.execute(
                """
                INSERT INTO prediction_configs (prediction_id, config_json)
                VALUES (?, ?)
                """,
                (cursor.lastrowid, model_config_json),
            )
        self.conn.commit()

    def get_prediction_config(self, prediction_id):
        """Fetch model configuration JSON for a prediction id."""
        if prediction_id is None:
            return None
        row = self.conn.execute(
            """
            SELECT config_json
            FROM prediction_configs
            WHERE prediction_id = ?
            ORDER BY created_at DESC, config_id DESC
            LIMIT 1
            """,
            (prediction_id,),
        ).fetchone()
        return row[0] if row else None

    def update_prediction_result(self, prediction_id, actual_result, outcome):
        """Update prediction with actual game result."""
        # noinspection SqlNoDataSourceInspection
        query = """
            UPDATE predictions
            SET actual_result = ?,
                outcome = ?
            WHERE prediction_id = ?
        """
        self.conn.execute(query, (actual_result, outcome, prediction_id))
        self.conn.commit()

    def get_backtest_data(self, start_date, end_date):
        """
        Fetch all predictions with actual results for backtesting.

        Returns DataFrame with predictions and outcomes.
        """
        # noinspection SqlNoDataSourceInspection
        query = """
            SELECT 
                p.*,
                gl.points as actual_points,
                gl.assists as actual_assists,
                gl.rebounds as actual_rebounds
            FROM predictions p
            LEFT JOIN game_logs gl
                ON p.player_id = gl.player_id
                AND p.game_date = gl.game_date
            WHERE p.game_date BETWEEN ? AND ?
              AND p.actual_result IS NOT NULL
            ORDER BY p.game_date DESC
        """
        return pd.read_sql_query(query, self.conn, params=(start_date, end_date))

    def get_market_line(self, player_id, game_date, stat_type, book=None, agg="median"):
        """
        Fetch market line for a player/stat/date, optionally scoped to one book.

        Args:
            player_id: NBA player id
            game_date: date-like value; compared on YYYY-MM-DD
            stat_type: points/assists/rebounds/pra
            book: optional sportsbook title to filter
            agg: one of median/mean/min/max for multi-book aggregation

        Returns:
            float | None
        """
        if isinstance(game_date, (pd.Timestamp, datetime)):
            game_date = game_date.strftime("%Y-%m-%d")
        else:
            game_date = str(game_date)[:10]

        if book:
            query = """
                SELECT line_value
                FROM betting_lines
                WHERE player_id = ?
                  AND game_date = ?
                  AND stat_type = ?
                  AND book = ?
            """
            params = (player_id, game_date, stat_type, book)
        else:
            query = """
                SELECT line_value
                FROM betting_lines
                WHERE player_id = ?
                  AND game_date = ?
                  AND stat_type = ?
            """
            params = (player_id, game_date, stat_type)

        rows = [r[0] for r in self.conn.execute(
            query, params).fetchall() if r and r[0] is not None]
        if not rows:
            return None

        series = pd.Series(rows, dtype="float64")
        agg_key = (agg or "median").lower()
        if agg_key == "mean":
            return float(series.mean())
        if agg_key == "min":
            return float(series.min())
        if agg_key == "max":
            return float(series.max())
        return float(series.median())

    def get_market_spread(self, player_id, game_date, book=None, agg="median", stat_types=None):
        """
        Fetch pregame spread value from betting_lines for a player/date.

        Args:
            player_id: NBA player id
            game_date: date-like value; compared on YYYY-MM-DD
            book: optional sportsbook title filter
            agg: one of median/mean/min/max for multi-row aggregation
            stat_types: optional list of stat_type aliases treated as spread fields

        Returns:
            float | None
        """
        if not player_id:
            return None

        if isinstance(game_date, (pd.Timestamp, datetime)):
            game_date = game_date.strftime("%Y-%m-%d")
        else:
            game_date = str(game_date)[:10]

        spread_aliases = stat_types or [
            "spread",
            "game_spread",
            "game spread",
            "line_spread",
            "line spread",
            "vegas_spread",
            "vegas spread",
            "pregame_spread",
            "pregame spread",
            "closing_spread",
            "closing spread",
        ]
        spread_aliases = sorted(
            {
                str(alias).strip().lower()
                for alias in spread_aliases
                if str(alias).strip()
            }
        )
        if not spread_aliases:
            return None

        placeholders = ", ".join(["?"] * len(spread_aliases))
        if book:
            query = f"""
                SELECT line_value
                FROM betting_lines
                WHERE player_id = ?
                  AND game_date = ?
                  AND lower(stat_type) IN ({placeholders})
                  AND book = ?
            """
            params = (player_id, game_date, *spread_aliases, book)
        else:
            query = f"""
                SELECT line_value
                FROM betting_lines
                WHERE player_id = ?
                  AND game_date = ?
                  AND lower(stat_type) IN ({placeholders})
            """
            params = (player_id, game_date, *spread_aliases)

        rows = [r[0] for r in self.conn.execute(
            query, params).fetchall() if r and r[0] is not None]
        if not rows:
            return None

        series = pd.Series(rows, dtype="float64")
        agg_key = (agg or "median").lower()
        if agg_key == "mean":
            return float(series.mean())
        if agg_key == "min":
            return float(series.min())
        if agg_key == "max":
            return float(series.max())
        return float(series.median())

    def get_team_defense(self, team_abbrev, season=None):
        """
        Fetch latest defensive rating for a team.

        Args:
            team_abbrev: Team abbreviation (e.g., "LAL")
            season: Optional season filter (e.g., "2024-25")

        Returns:
            float | None: defensive rating if available
        """
        if not team_abbrev:
            return None

        if season:
            query = """
                SELECT def_rating
                FROM team_defense
                WHERE team_abbrev = ?
                  AND season = ?
                ORDER BY last_updated DESC
                LIMIT 1
            """
            params = (team_abbrev, season)
        else:
            query = """
                SELECT def_rating
                FROM team_defense
                WHERE team_abbrev = ?
                ORDER BY season DESC, last_updated DESC
                LIMIT 1
            """
            params = (team_abbrev,)

        row = self.conn.execute(query, params).fetchone()
        if not row:
            return None
        try:
            return float(row[0]) if row[0] is not None else None
        except (TypeError, ValueError):
            return None

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
