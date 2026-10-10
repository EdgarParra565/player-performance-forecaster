"""SQLite database manager for NBA data, betting lines, and predictions."""
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# team_priors rows carry no game date; the hourly runner recomputes a live
# matchup's prior every tick, so anything older than a day is a past game's
# prior and must not blend into today's projection.
TEAM_PRIOR_MAX_AGE_HOURS = 24.0


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


# Map a canonical ``stat_type`` → the SQL expression that recovers the realized
# value from a ``game_logs`` row. PRA / RA are derived from raw columns. Shared
# by ``backfill_predictions_outcomes`` (predictions) and ``settle_bet_log``
# (bet_log) so both grade against the exact same stat definitions.
_GAMELOG_STAT_EXPR = {
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


def _grade_bet_side(side, actual, line):
    """Grade a paper bet given the realized value and the line it was staked at.

    Returns 'won' / 'lost' / 'push'. A push (actual == line) is a push for
    either side; otherwise an over wins when actual > line and an under wins
    when actual < line.
    """
    a = _safe_float(actual)
    line_v = _safe_float(line)
    if a is None or line_v is None:
        return None
    if a == line_v:
        return "push"
    over_hit = a > line_v
    if str(side or "").strip().lower() == "over":
        return "won" if over_hit else "lost"
    return "lost" if over_hit else "won"


class DatabaseNotReadyError(RuntimeError):
    """A ``read_only=True`` open found no usable database at the path (file
    missing, empty, or missing tables/migrated columns the readers need)."""


class DatabaseManager:
    """Manages all database operations for NBA data.

    ``DatabaseManager(path)`` — the writer open: creates the file/parent dir,
    runs schema.sql + every idempotent migration. Used by all ETL/model code.

    ``DatabaseManager(path, read_only=True)`` — for read-only services (the
    flagship API): opens the existing file with SQLite ``mode=ro``, runs NO
    DDL or migrations, never creates the file, and raises
    ``DatabaseNotReadyError`` when the file is missing/empty or lacks a table
    or migrated column from ``REQUIRED_SCHEMA``. Any write through it fails
    with ``sqlite3.OperationalError`` ("attempt to write a readonly database").
    """

    def __init__(self, db_path='data/database/nba_data.db', read_only: bool = False):
        self.db_path = Path(db_path)
        self.read_only = bool(read_only)
        self.conn = None
        if self.read_only:
            self._open_read_only()
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    @contextmanager
    def _atomic(self, name: str = "writer"):
        """All-or-nothing scope for one writer's statements (SAVEPOINT).

        ``executemany`` runs one statement per row: when row k fails, rows
        0..k-1 stay in the open transaction and the NEXT unrelated
        ``commit()`` persists them — a silent partial write. Inside this
        scope a failure rolls back exactly this writer's rows (work a caller
        left pending outside the savepoint is untouched) and re-raises; on
        success the savepoint is released, which commits when it is the
        outermost transaction.
        """
        sp = f"sp_{name}"
        self.conn.execute(f"SAVEPOINT {sp}")
        try:
            yield
        except BaseException:
            self.conn.execute(f"ROLLBACK TO {sp}")
            self.conn.execute(f"RELEASE {sp}")
            raise
        self.conn.execute(f"RELEASE {sp}")

    @classmethod
    def required_schema(cls) -> dict:
        """``{table: {columns}}`` a read-only open requires: every table in
        schema.sql plus the columns added by migrations (an un-migrated file
        would make the newer readers fail mid-request instead of at open)."""
        import re

        schema = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
        required: dict = {
            name: set() for name in re.findall(
                r"CREATE TABLE IF NOT EXISTS\s+(\w+)", schema)
        }
        for table in cls._SPORT_COLUMN_TABLES:
            required.setdefault(table, set()).add("sport")
        for table in cls._LAST_SEEN_TABLES:
            required.setdefault(table, set()).add("last_seen_at_utc")
        for table, cols in cls._MIGRATED_COLUMNS.items():
            required.setdefault(table, set()).update(cols)
        return required

    def _open_read_only(self):
        path = self.db_path
        if not path.is_file():
            raise DatabaseNotReadyError(f"database file not found: {path}")
        if path.stat().st_size == 0:
            raise DatabaseNotReadyError(f"database file is empty: {path}")
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        try:
            self.conn = sqlite3.connect(uri, uri=True)
            have: dict = {}
            for (table,) in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall():
                have[table] = {
                    r[1] for r in self.conn.execute(
                        f"PRAGMA table_info({table})").fetchall()
                }
        except sqlite3.DatabaseError as exc:
            self.close()
            raise DatabaseNotReadyError(f"not a readable SQLite database: {path}: {exc}") from exc
        missing_tables = sorted(t for t in self.required_schema() if t not in have)
        missing_cols = sorted(
            f"{t}.{c}"
            for t, cols in self.required_schema().items() if t in have
            for c in cols if c not in have[t]
        )
        if missing_tables or missing_cols:
            self.close()
            detail = []
            if missing_tables:
                detail.append("missing tables: " + ", ".join(missing_tables))
            if missing_cols:
                detail.append("missing columns: " + ", ".join(missing_cols))
            raise DatabaseNotReadyError(
                f"database at {path} is incomplete ({'; '.join(detail)}) — "
                "open it once with a writer DatabaseManager(path) to migrate"
            )
        logger.info("Database opened read-only at %s", path)

    # Tables that gain a ``sport`` discriminator for the multi-sport rollout
    # (NFL first). Added via idempotent migration rather than editing every
    # CREATE TABLE so existing DBs upgrade in place. Default 'nba' keeps all
    # current rows + inserts that omit the column working unchanged.
    _SPORT_COLUMN_TABLES = (
        "game_logs", "betting_lines", "betting_line_snapshots",
        "players", "web_prop_cards", "predictions", "web_team_lines",
    )

    def _initialize_database(self):
        """Create database and tables if they don't exist."""
        self.conn = sqlite3.connect(self.db_path)

        # Read and execute schema
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, "r", encoding="utf-8") as f:
            self.conn.executescript(f.read())

        self._ensure_sport_columns()
        self._ensure_betting_lines_source_column()
        self._ensure_last_seen_columns()
        self._ensure_betting_lines_main_line_column()
        self._ensure_game_date_columns()
        self._ensure_bet_log_unique_index()
        self._migrate_team_priors_to_abbrev_keys()

        logger.info("Database initialized at %s", self.db_path)

    # Change-only boards: a row is written only when the line moves, so
    # observed_at_utc means "last CHANGED". last_seen_at_utc is bumped on every
    # unchanged re-scrape so freshness windows (since_hours) keep stable lines.
    # table -> the column the last_seen backfill starts from ("last changed").
    _LAST_SEEN_TABLES = {
        "web_prop_cards": "observed_at_utc",
        "web_team_lines": "observed_at_utc",
        "betting_lines": "scraped_at",
    }

    @staticmethod
    def seen_at_sql(alias: str = "", base_col: str = "observed_at_utc") -> str:
        """SQL expression for "when was this line last observed" (NULL-safe
        for rows written before the column existed). ``base_col`` is the
        table's last-changed timestamp (``scraped_at`` for betting_lines)."""
        p = f"{alias}." if alias else ""
        return f"COALESCE({p}last_seen_at_utc, {p}{base_col})"

    def _ensure_last_seen_columns(self):
        """Add nullable ``last_seen_at_utc`` to the change-only tables and
        backfill it from their last-changed column once. Idempotent."""
        for table, base_col in self._LAST_SEEN_TABLES.items():
            try:
                cols = {
                    r[1] for r in self.conn.execute(
                        f"PRAGMA table_info({table})").fetchall()
                }
            except sqlite3.OperationalError:
                continue
            if not cols or "last_seen_at_utc" in cols:
                continue
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN last_seen_at_utc TEXT")
            self.conn.execute(
                f"UPDATE {table} SET last_seen_at_utc = {base_col} "
                "WHERE last_seen_at_utc IS NULL"
            )
            self.conn.commit()

    def _bump_last_seen(self, table, id_col, key_sql, seen):
        """Move ``last_seen_at_utc`` forward on the CURRENT row of each key.

        ``seen`` maps key tuple → latest observed_at of an unchanged re-scrape.
        Only ever moves forward (re-parsing an old snapshot is a no-op)."""
        if not seen:
            return
        sql = f"""
            UPDATE {table} SET last_seen_at_utc = ?
            WHERE {id_col} = (
                SELECT {id_col} FROM {table} WHERE {key_sql}
                ORDER BY observed_at_utc DESC, {id_col} DESC LIMIT 1
            )
            AND (last_seen_at_utc IS NULL
                 OR datetime(last_seen_at_utc) < datetime(?))
        """
        self.conn.executemany(
            sql, [(ts, *key, ts) for key, ts in seen.items()]
        )

    # Columns added by ALTER-style migrations (beyond sport / last_seen), so a
    # read-only open can tell an un-migrated file apart from a current one.
    _MIGRATED_COLUMNS = {
        "betting_lines": ("source", "is_main_line"),
        "web_team_lines": ("game_date",),
        "team_priors": ("game_date",),
    }

    def _ensure_game_date_columns(self):
        """Game dates for team lines + priors (a team with two games in the
        lines window must not share one prior). Idempotent.

        * ``web_team_lines.game_date`` — nullable ALTER (NULL = undated).
        * ``team_priors`` — rebuilt once with ``game_date TEXT NOT NULL
          DEFAULT ''`` in the primary key (SQLite can't alter a PK); legacy
          rows keep ``''`` = date unknown.
        """
        try:
            wtl = {r[1] for r in self.conn.execute("PRAGMA table_info(web_team_lines)")}
            tp = {r[1] for r in self.conn.execute("PRAGMA table_info(team_priors)")}
        except sqlite3.OperationalError:
            return
        if wtl and "game_date" not in wtl:
            self.conn.execute("ALTER TABLE web_team_lines ADD COLUMN game_date TEXT")
        if tp and "game_date" not in tp:
            cols = [
                "away_team", "home_team", "computed_at_utc", "consensus_total",
                "home_spread", "away_spread", "home_team_total", "away_team_total",
                "home_win_prob_devig", "away_win_prob_devig", "pace_factor",
                "n_books", "latest_observed_at",
            ]
            col_sql = ", ".join(cols)
            self.conn.executescript(f"""
                ALTER TABLE team_priors RENAME TO team_priors_pre_game_date;
                CREATE TABLE team_priors (
                    away_team             TEXT NOT NULL,
                    home_team             TEXT NOT NULL,
                    game_date             TEXT NOT NULL DEFAULT '',
                    computed_at_utc       TIMESTAMP NOT NULL,
                    consensus_total       REAL,
                    home_spread           REAL,
                    away_spread           REAL,
                    home_team_total       REAL,
                    away_team_total       REAL,
                    home_win_prob_devig   REAL,
                    away_win_prob_devig   REAL,
                    pace_factor           REAL,
                    n_books               INTEGER,
                    latest_observed_at    TIMESTAMP,
                    PRIMARY KEY (away_team, home_team, game_date)
                );
                INSERT INTO team_priors ({col_sql}, game_date)
                    SELECT {col_sql}, '' FROM team_priors_pre_game_date;
                DROP TABLE team_priors_pre_game_date;
            """)
        self.conn.commit()

    # SQL predicate: row is a main line (legacy/untagged NULL rows count as main).
    MAIN_LINE_SQL = "COALESCE({p}is_main_line, 1) = 1"

    @classmethod
    def main_line_sql(cls, alias: str = "") -> str:
        return cls.MAIN_LINE_SQL.format(p=f"{alias}." if alias else "")

    def _ensure_betting_lines_main_line_column(self):
        """Add ``betting_lines.is_main_line`` (1 main / 0 alt rung / NULL
        untagged) and tag existing ladders once. Idempotent.

        Backfill groups legacy rows by (player, date, book, stat, scraped_at):
        one writer call stamps one ``scraped_at``, so a group is one scrape's
        ladder; its main rung is picked by ``odds.main_line_index``."""
        try:
            cols = {
                r[1] for r in self.conn.execute(
                    "PRAGMA table_info(betting_lines)").fetchall()
            }
        except sqlite3.OperationalError:
            return
        if not cols or "is_main_line" in cols:
            return
        from nba_model.model.odds import main_line_index

        self.conn.execute("ALTER TABLE betting_lines ADD COLUMN is_main_line INTEGER")
        rows = self.conn.execute(
            """
            SELECT line_id, player_id, game_date, lower(book), lower(stat_type),
                   scraped_at, line_value, over_odds, under_odds
            FROM betting_lines ORDER BY line_id
            """
        ).fetchall()
        groups: dict = {}
        for r in rows:
            groups.setdefault(r[1:6], []).append(
                {"line_id": r[0], "line_value": r[6],
                 "over_odds": r[7], "under_odds": r[8]}
            )
        updates = []
        for ladder in groups.values():
            main = main_line_index(ladder)
            updates.extend(
                (1 if i == main else 0, c["line_id"]) for i, c in enumerate(ladder)
            )
        self.conn.executemany(
            "UPDATE betting_lines SET is_main_line = ? WHERE line_id = ?", updates)
        self.conn.commit()

    def _migrate_team_priors_to_abbrev_keys(self):
        """Re-key legacy ``team_priors`` rows from nicknames ("76ers") to the
        NBA codes ("PHI") every consumer looks up by. Idempotent: rows already
        keyed by codes (or unknown names) are left alone; when both a legacy
        and a code-keyed row exist for the same matchup, the newer
        ``computed_at_utc`` wins."""
        from nba_model.scrapers.team_names import NBA_TEAM_ABBREVS, team_abbrev

        try:
            cols = {r[1] for r in self.conn.execute("PRAGMA table_info(team_priors)")}
            gd_sql = "game_date" if "game_date" in cols else "''"
            rows = self.conn.execute(
                f"SELECT away_team, home_team, computed_at_utc, {gd_sql} FROM team_priors"
            ).fetchall()
        except sqlite3.OperationalError:
            return
        changed = False
        for away, home, computed, gd in rows:
            if away in NBA_TEAM_ABBREVS and home in NBA_TEAM_ABBREVS:
                continue
            new_away = team_abbrev(away) or away
            new_home = team_abbrev(home) or home
            if (new_away, new_home) == (away, home):
                continue
            gd_where = " AND game_date = ?" if gd_sql == "game_date" else ""
            gd_param = (gd,) if gd_sql == "game_date" else ()
            existing = self.conn.execute(
                "SELECT computed_at_utc FROM team_priors "
                "WHERE away_team = ? AND home_team = ?" + gd_where,
                (new_away, new_home, *gd_param),
            ).fetchone()
            if existing is not None and str(existing[0] or "") >= str(computed or ""):
                self.conn.execute(
                    "DELETE FROM team_priors WHERE away_team = ? AND home_team = ?" + gd_where,
                    (away, home, *gd_param),
                )
            else:
                self.conn.execute(
                    "UPDATE OR REPLACE team_priors SET away_team = ?, home_team = ? "
                    "WHERE away_team = ? AND home_team = ?" + gd_where,
                    (new_away, new_home, away, home, *gd_param),
                )
            changed = True
        if changed:
            self.conn.commit()

    def _ensure_betting_lines_source_column(self):
        """Add a nullable ``source`` provenance column to ``betting_lines``.

        Real odds harvested from a public aggregator (e.g. VegasInsider, which
        republishes ~11 books' lines) are attributed to the UNDERLYING book in
        ``book`` but tagged here with the aggregator name so a row's origin
        stays auditable — a direct FanDuel scrape (``source`` NULL) is
        distinguishable from a FanDuel line lifted off an aggregator
        (``source='vegasinsider'``). Idempotent; additive (default NULL) so
        every existing row and insert path is unaffected."""
        try:
            cols = {
                r[1] for r in self.conn.execute(
                    "PRAGMA table_info(betting_lines)").fetchall()
            }
        except sqlite3.OperationalError:
            return
        if cols and "source" not in cols:
            self.conn.execute("ALTER TABLE betting_lines ADD COLUMN source TEXT")
            self.conn.commit()

    def _ensure_sport_columns(self):
        """Add ``sport TEXT NOT NULL DEFAULT 'nba'`` to the multi-sport tables
        when missing. Idempotent; table names come from a hardcoded tuple
        (no injection)."""
        for table in self._SPORT_COLUMN_TABLES:
            try:
                cols = {
                    r[1] for r in self.conn.execute(
                        f"PRAGMA table_info({table})").fetchall()
                }
            except sqlite3.OperationalError:
                continue
            if not cols or "sport" in cols:
                continue
            self.conn.execute(
                f"ALTER TABLE {table} "
                "ADD COLUMN sport TEXT NOT NULL DEFAULT 'nba'"
            )
        self.conn.commit()

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

    def get_team_recent_avg_points(
        self,
        team_abbrev: str,
        n_games: int = 20,
    ) -> Optional[float]:
        """Average points the team itself scored over its last N games.

        This is the *per-team* baseline that matches ``team_priors``'
        per-team ``implied_team_total`` (≈ ``(consensus_total ± spread) / 2``),
        so ``blend_team_prior`` compares like-with-like. (Contrast
        ``get_team_recent_avg_total`` which returns the full game total
        ``pts + opp_pts``.) Returns ``None`` when the team has no recent games.
        """
        if not team_abbrev:
            return None
        rows = self.conn.execute(
            """
            SELECT pts FROM games
            WHERE upper(team_abbrev) = upper(?)
              AND pts IS NOT NULL
            ORDER BY game_date DESC
            LIMIT ?
            """,
            (team_abbrev, int(max(1, n_games))),
        ).fetchall()
        pts = [float(r[0]) for r in rows if r and r[0] is not None]
        if not pts:
            return None
        return sum(pts) / len(pts)

    _TEAM_PRIOR_COLS = (
        "away_team", "home_team", "game_date", "computed_at_utc",
        "consensus_total", "home_spread", "away_spread",
        "home_team_total", "away_team_total",
        "home_win_prob_devig", "away_win_prob_devig",
        "pace_factor", "n_books", "latest_observed_at",
    )

    def upsert_team_priors(self, records):
        """Upsert reverse-engineered priors (one row per matchup + game date).

        Team keys are normalized to NBA codes ("76ers" → "PHI") — the key
        space players.team / games.team_abbrev use — so consumer lookups
        actually hit. Unknown names are stored as given. ``game_date``
        (YYYY-MM-DD) keys two games of the same matchup apart; missing → ''
        (date unknown).
        """
        if not records:
            return {"upserted": 0, "attempted": 0}
        from nba_model.scrapers.team_names import team_abbrev
        cols = ", ".join(self._TEAM_PRIOR_COLS)
        query = f"""
            INSERT OR REPLACE INTO team_priors ({cols})
            VALUES ({", ".join("?" * len(self._TEAM_PRIOR_COLS))})
        """
        payload = []
        for r in records:
            payload.append((
                team_abbrev(r.get("away_team")) or r.get("away_team"),
                team_abbrev(r.get("home_team")) or r.get("home_team"),
                str(r.get("game_date") or "").strip()[:10],
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
        with self._atomic():
            self.conn.executemany(query, payload)
        self.conn.commit()
        upserted = self.conn.total_changes - before
        return {"upserted": int(upserted), "attempted": int(len(payload))}

    @staticmethod
    def _team_prior_fresh_clause(max_age_hours):
        """``(sql, params)`` restricting team_priors to recent computations
        (a prior is recomputed every tick while its game's lines are up, so an
        old ``computed_at_utc`` means a past game)."""
        if max_age_hours is None:
            return "1=1", ()
        return (
            "datetime(computed_at_utc) >= datetime('now', ?)",
            (f"-{float(max_age_hours)} hours",),
        )

    @staticmethod
    def _slate_today() -> str:
        """Today's NBA slate date (America/New_York)."""
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("America/New_York")).date().isoformat()

    @classmethod
    def _pick_prior(cls, candidates: list, game_date: Optional[str]):
        """Choose ONE prior among a team's/matchup's candidate rows.

        ``candidates``: dicts with ``game_date`` ('' = unknown) and
        ``computed_at_utc``. With ``game_date``: that date's prior, else an
        undated one — NEVER another date's (two games of one team must not
        share a prior). Without: the earliest dated game on/after today's
        slate, else an undated one; past-dated priors are ignored. Ties →
        newest ``computed_at_utc``.
        """
        def newest(rows):
            return max(rows, key=lambda r: str(r.get("computed_at_utc") or "")) if rows else None

        undated = [c for c in candidates if not c.get("game_date")]
        dated = [c for c in candidates if c.get("game_date")]
        if game_date:
            want = str(game_date)[:10]
            return newest([c for c in dated if c["game_date"] == want]) or newest(undated)
        today = cls._slate_today()
        upcoming = [c for c in dated if c["game_date"] >= today]
        if upcoming:
            first = min(c["game_date"] for c in upcoming)
            return newest([c for c in upcoming if c["game_date"] == first])
        return newest(undated)

    def _fresh_prior_rows(self, max_age_hours, where_sql: str = "1=1", params=()) -> list:
        fresh_sql, fresh_params = self._team_prior_fresh_clause(max_age_hours)
        rows = self.conn.execute(
            f"SELECT {', '.join(self._TEAM_PRIOR_COLS)} FROM team_priors "
            f"WHERE {where_sql} AND {fresh_sql}",
            (*params, *fresh_params),
        ).fetchall()
        return [dict(zip(self._TEAM_PRIOR_COLS, r)) for r in rows]

    def get_team_prior(
        self,
        away_team: str,
        home_team: str,
        max_age_hours: Optional[float] = TEAM_PRIOR_MAX_AGE_HOURS,
        game_date: Optional[str] = None,
    ):
        """The prior for a matchup (see ``_pick_prior`` for how ``game_date``
        selects among several games); None if absent.

        Accepts any team form (nickname / full name / code); only priors
        computed within ``max_age_hours`` count (``None`` = no age cap).
        """
        from nba_model.scrapers.team_names import team_abbrev

        away_key = team_abbrev(away_team) or away_team
        home_key = team_abbrev(home_team) or home_team
        rows = self._fresh_prior_rows(
            max_age_hours,
            "lower(away_team) = lower(?) AND lower(home_team) = lower(?)",
            (away_key, home_key),
        )
        return self._pick_prior(rows, game_date)

    def get_team_prior_inputs(
        self,
        player_team: str,
        opponent_team: str,
        max_age_hours: Optional[float] = TEAM_PRIOR_MAX_AGE_HOURS,
        game_date: Optional[str] = None,
    ) -> dict:
        """Resolve the team-prior signals for a player's team in a matchup.

        Both orientations (player home / away) are candidates; ``game_date``
        (the projection's game date) picks the right game when the two teams
        meet more than once in the window. Returns a dict ready to splat into
        ``simulation.blend_team_prior`` (``pace_factor`` /
        ``implied_team_total`` / ``team_recent_avg_total``), or ``{}``.
        """
        if not player_team or not opponent_team:
            return {}
        from nba_model.scrapers.team_names import team_abbrev

        me = team_abbrev(player_team) or player_team
        opp = team_abbrev(opponent_team) or opponent_team
        rows = self._fresh_prior_rows(
            max_age_hours,
            "((lower(away_team) = lower(?) AND lower(home_team) = lower(?)) OR "
            " (lower(away_team) = lower(?) AND lower(home_team) = lower(?)))",
            (opp, me, me, opp),
        )
        prior = self._pick_prior(rows, game_date)
        if prior is None:
            return {}
        player_is_home = str(prior["home_team"]).lower() == str(me).lower()
        implied = (prior.get("home_team_total") if player_is_home
                   else prior.get("away_team_total"))
        own_key = prior.get("home_team") if player_is_home else prior.get("away_team")
        return {
            "pace_factor": prior.get("pace_factor"),
            "implied_team_total": implied,
            "team_recent_avg_total": self.get_team_recent_avg_points(own_key),
        }

    def get_team_prior_inputs_map(
        self,
        max_age_hours: Optional[float] = TEAM_PRIOR_MAX_AGE_HOURS,
        game_date: Optional[str] = None,
    ) -> dict:
        """Map every team (NBA code) with a FRESH prior to its
        ``blend_team_prior`` inputs, one game per team.

        ``game_date`` = the slate being projected: each team gets THAT game's
        prior (or an undated one), never another game's. Without it each team
        gets its next upcoming game (``_pick_prior``). Used by the prop-board
        / hourly recompute / scanner full-mode paths.
        """
        from nba_model.scrapers.team_names import team_abbrev

        per_team: dict = {}
        for row in self._fresh_prior_rows(max_age_hours):
            for team, team_total in ((row["home_team"], row["home_team_total"]),
                                     (row["away_team"], row["away_team_total"])):
                if not team:
                    continue
                key = team_abbrev(team) or str(team).upper()
                per_team.setdefault(key, []).append({
                    "game_date": row["game_date"],
                    "computed_at_utc": row["computed_at_utc"],
                    "pace_factor": row["pace_factor"],
                    "implied_team_total": team_total,
                })
        out: dict = {}
        for key, cands in per_team.items():
            pick = self._pick_prior(cands, game_date)
            if pick is None:
                continue
            out[key] = {
                "pace_factor": pick["pace_factor"],
                "implied_team_total": pick["implied_team_total"],
                "team_recent_avg_total": self.get_team_recent_avg_points(key),
            }
        return out

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
        # Map stat_type → SQL expression against game_logs columns (shared with
        # settle_bet_log). PRA / RA are computed from raw columns, others direct.
        stat_expr = _GAMELOG_STAT_EXPR

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

    # ------------------------------------------------------------------
    # Paper-trading bet log (WS10 — measurement layer, NOT execution)
    # ------------------------------------------------------------------

    # One paper-trade pick = (slate, player, stat, side, line, book, mode,
    # sport); re-running bet_slip for the same slate must not double-count.
    _BET_LOG_PICK_KEY_SQL = (
        "game_date, lower(player_name), lower(stat_type), side, line, "
        "lower(COALESCE(book, '')), COALESCE(model_mode, ''), sport"
    )

    def _ensure_bet_log_unique_index(self):
        """UNIQUE index on the bet_log pick key. Skipped (warning) when the
        table already holds duplicate picks — deleting paper-trade history is
        the owner's call; ``insert_bet_log_rows`` still refuses new dupes."""
        try:
            exists = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='ux_bet_log_pick'"
            ).fetchone()
            if exists:
                return
            dupes = self.conn.execute(
                f"SELECT COUNT(*) FROM (SELECT 1 FROM bet_log "
                f"GROUP BY {self._BET_LOG_PICK_KEY_SQL} HAVING COUNT(*) > 1)"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            return
        if dupes:
            logger.warning(
                "bet_log has %s duplicate pick groups; unique index not created "
                "(new duplicates are still skipped at insert)", dupes)
            return
        self.conn.execute(
            f"CREATE UNIQUE INDEX ux_bet_log_pick ON bet_log ({self._BET_LOG_PICK_KEY_SQL})")
        self.conn.commit()

    def _bet_log_pick_exists(self, row) -> bool:
        return self.conn.execute(
            """
            SELECT 1 FROM bet_log
            WHERE game_date = ? AND lower(player_name) = lower(?)
              AND lower(stat_type) = lower(?) AND side = ? AND line = ?
              AND lower(COALESCE(book, '')) = lower(COALESCE(?, ''))
              AND COALESCE(model_mode, '') = COALESCE(?, '') AND sport = ?
            LIMIT 1
            """,
            (row[1], row[3], row[4], row[7], row[6], row[5], row[11], row[16]),
        ).fetchone() is not None

    def insert_bet_log_rows(self, rows):
        """Insert paper-trade ``bet_log`` rows.

        Returns ``{inserted, attempted, duplicates_ignored}``. Each row is a
        dict of the bet-slip fields. Missing optional keys default to NULL;
        ``created_at_utc`` defaults to now, ``status`` to 'pending', ``sport``
        to 'nba'. Rows missing a required field (game_date, player_name,
        stat_type, line) or with an invalid side are skipped. A pick already
        logged under the same key (slate, player, stat, side, line, book,
        model_mode, sport) is ignored, so re-running bet_slip for a slate
        doesn't double-count stakes / P&L."""
        rows = list(rows or [])
        if not rows:
            return {"inserted": 0, "attempted": 0}

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        query = """
            INSERT OR IGNORE INTO bet_log (
                created_at_utc, game_date, player_id, player_name, stat_type,
                book, line, side, model_prob, implied_prob, edge, model_mode,
                distribution, kelly_fraction, stake_units, status, sport
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        payload = []
        duplicates = 0
        seen_keys: set = set()
        for r in rows:
            game_date = r.get("game_date")
            player_name = str(r.get("player_name", "")).strip()
            stat_type = str(r.get("stat_type", "")).strip().lower()
            line = _safe_float(r.get("line"))
            side = str(r.get("side", "")).strip().lower()
            if (not game_date or not player_name or not stat_type
                    or line is None or side not in ("over", "under")):
                continue
            payload.append((
                str(r.get("created_at_utc") or now),
                str(game_date)[:10],
                _safe_int(r.get("player_id")),
                player_name,
                stat_type,
                (str(r.get("book")).strip() if r.get("book") else None),
                line,
                side,
                _safe_float(r.get("model_prob")),
                _safe_float(r.get("implied_prob")),
                _safe_float(r.get("edge")),
                (str(r.get("model_mode")) if r.get("model_mode") else None),
                (str(r.get("distribution")) if r.get("distribution") else None),
                _safe_float(r.get("kelly_fraction")),
                _safe_float(r.get("stake_units")),
                str(r.get("status") or "pending"),
                str(r.get("sport") or "nba"),
            ))
            new = payload[-1]
            key = (new[1], new[3].lower(), new[4], new[7], new[6],
                   (new[5] or "").lower(), new[11] or "", new[16])
            if key in seen_keys or self._bet_log_pick_exists(new):
                payload.pop()
                duplicates += 1
                continue
            seen_keys.add(key)
        if not payload:
            return {"inserted": 0, "attempted": 0, "duplicates_ignored": duplicates}

        before = self.conn.total_changes
        with self._atomic():
            self.conn.executemany(query, payload)
        self.conn.commit()
        inserted = self.conn.total_changes - before
        duplicates += len(payload) - inserted
        logger.info("Inserted %s bet_log rows (%s duplicates ignored)", inserted, duplicates)
        return {"inserted": int(inserted), "attempted": int(len(payload)),
                "duplicates_ignored": int(duplicates)}

    def _closing_clv_delta(self, player_id, game_date, stat_type, side,
                           entry_implied, line=None, book=None):
        """CLV delta for one pick from the latest line snapshot, or ``None``.

        Reuses ``clv_proxy`` odds→implied conversion. Returns
        ``closing_side_implied − entry_implied`` (positive = the market closed
        toward the bet's side vs the entry price). ``None`` when there's no
        snapshot AT THE PICK'S LINE, no odds on that side, or no entry implied
        prob. A price at a different line is not the same bet, so it is never
        used; among same-line snapshots the pick's own book wins, then the
        latest snapshot (``snapshot_id`` breaks ties within one poll)."""
        entry = _safe_float(entry_implied)
        line_value = _safe_float(line)
        if entry is None or player_id is None or line_value is None:
            return None
        row = self.conn.execute(
            """
            SELECT over_odds, under_odds
            FROM betting_line_snapshots
            WHERE player_id = ?
              AND DATE(game_date) = DATE(?)
              AND lower(stat_type) = lower(?)
              AND ABS(line_value - ?) < 1e-6
            ORDER BY (lower(book) = lower(?)) DESC,
                     snapshot_ts_utc DESC, snapshot_id DESC
            LIMIT 1
            """,
            (int(player_id), str(game_date), str(stat_type), line_value,
             str(book or "")),
        ).fetchone()
        if not row:
            return None
        from nba_model.evaluation.clv_proxy import _safe_implied_prob
        close_odds = row[0] if str(side).strip().lower() == "over" else row[1]
        close_implied = _safe_implied_prob(close_odds)
        if close_implied is None:
            return None
        return float(close_implied - entry)

    def settle_bet_log(self, fill_clv=True):
        """Grade every pending ``bet_log`` row whose game now has logs.

        Mirrors ``backfill_predictions_outcomes``: look up the pick's realized
        stat in ``game_logs`` (via the shared ``_GAMELOG_STAT_EXPR``), set
        ``status`` ∈ {won, lost, push} for the staked side, and stamp
        ``settled_at_utc`` + ``actual_value``. When ``fill_clv`` and a closing
        ``betting_line_snapshots`` row exists, fill ``clv_delta`` too.

        Idempotent — only touches ``status = 'pending'`` rows, so re-running
        just settles games that have since landed. Returns settle counts."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        pending = self.conn.execute(
            """
            SELECT log_id, player_id, game_date, stat_type, line, side,
                   implied_prob, book
            FROM bet_log
            WHERE status = 'pending'
            """
        ).fetchall()

        update_stmt = """
            UPDATE bet_log
               SET status = ?,
                   settled_at_utc = ?,
                   actual_value = ?,
                   clv_delta = COALESCE(?, clv_delta)
             WHERE log_id = ?
        """

        settled = 0
        scanned = 0
        clv_filled = 0
        unsupported_stats: set[str] = set()

        for (log_id, player_id, game_date, stat_type, line, side,
             implied_prob, book) in pending:
            scanned += 1
            stat = (stat_type or "").strip().lower()
            expr = _GAMELOG_STAT_EXPR.get(stat)
            if expr is None or player_id is None:
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
            status = _grade_bet_side(side, actual, line)
            if status is None:
                continue
            clv = self._closing_clv_delta(
                player_id, game_date, stat, side, implied_prob,
                line=line, book=book,
            ) if fill_clv else None
            if clv is not None:
                clv_filled += 1
            self.conn.execute(
                update_stmt, (status, now, actual, clv, int(log_id)))
            settled += 1

        self.conn.commit()
        result = {
            "scanned": int(scanned),
            "settled": int(settled),
            "remaining_pending": int(scanned - settled),
            "clv_filled": int(clv_filled),
        }
        if unsupported_stats:
            result["unsupported_stats"] = sorted(unsupported_stats)
        logger.info(
            "Settled bet_log: %s settled, %s pending, %s clv filled (scanned %s)",
            settled, result["remaining_pending"], clv_filled, scanned,
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
        with self._atomic("players_sync"):
            self.conn.executemany(query, rows)
        self.conn.commit()
        upserted = self.conn.total_changes - before

        # Second pass: any player_id that's in ``players`` but not in
        # ``nba_active_players_ref`` (recently-traded veterans dropped from
        # the active roster API, draft picks added before the next ref sync,
        # etc.) still gets a team if we can derive one from game_logs. The
        # audit_db data-quality flag tracks players with NULL team — closing
        # this loop keeps that count at zero whenever there's a matchup to
        # read from.
        team_from_logs = self.conn.execute(
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
            )
            SELECT player_id, team
            FROM ranked
            WHERE rn = 1
              AND player_id IN (
                  SELECT player_id FROM players
                  WHERE team IS NULL OR team = ''
              )
            """
        ).fetchall()
        if team_from_logs:
            with self._atomic("players_team_patch"):
                self.conn.executemany(
                    "UPDATE players SET team = ?, last_updated = CURRENT_TIMESTAMP "
                    "WHERE player_id = ? AND (team IS NULL OR team = '')",
                    [(team, pid) for pid, team in team_from_logs],
                )
            self.conn.commit()

        logger.info(
            "sync_players_table: upserted %s rows (%s with team derived, "
            "%s additionally patched from game_logs)",
            upserted, sum(1 for r in rows if r[2]), len(team_from_logs),
        )
        return {
            "upserted": int(upserted),
            "attempted": int(len(rows)),
            "patched_from_game_logs": int(len(team_from_logs)),
        }

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
        with self._atomic():
            self.conn.executemany(query, payload)
        self.conn.commit()
        logger.info("Upserted %s team_defense rows", len(payload))

    def insert_betting_lines_records(self, records):
        """
        Insert betting lines change-only, tagging alt-line ladders.

        One call is treated as one scrape. Per (player, date, book, stat) the
        batch's rungs are tagged ``is_main_line`` (1 = the main line picked by
        ``odds.main_line_index``, 0 = alt rung). A row is skipped only when it
        equals the LATEST stored row of its kind — the latest main line for
        the key, or the latest alt row at the same line value — so an
        A→B→A move is recorded (the old check skipped anything that matched
        ANY historical row, leaving latest-by-scraped_at stale). A skipped
        unchanged row bumps the stored current row's ``last_seen_at_utc`` so
        ``scraped_at`` since-windows keep a line that simply didn't move
        (readers use ``COALESCE(last_seen_at_utc, scraped_at)``). Book keys
        compare case-insensitively ('FanDuel' == 'fanduel'); the stored name
        is kept as given.

        Args:
            records: Iterable of dicts with keys:
                player_id, game_date, book, stat_type, line_value, over_odds,
                under_odds (+ optional provenance ``source``)

        Returns:
            dict: inserted, duplicates_ignored, rejected_implausible, attempted,
            alt_lines_tagged.
        """
        if not records:
            return {
                "inserted": 0,
                "duplicates_ignored": 0,
                "attempted": 0,
            }

        # SECURITY (data poisoning defense): drop scraper-poisoned rows that
        # claim a structurally implausible line/odds. The scrapers are the
        # only third-party-influenced surface; even one bad row pollutes the
        # consensus mean + EV math for every user. The validator's range is
        # deliberately generous - real outliers pass, deliberate bad data
        # (negative lines, 9999.5 points, nan, inf, odds=0) gets dropped.
        from nba_model.model.odds import main_line_index
        from nba_model.web.input_validation import is_plausible_betting_line

        def _odds(v):
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        valid = []
        rejected_implausible = 0
        seen_in_batch = set()
        batch_dupes = 0
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
            group = (row[0], str(row[1]), str(row[2]).strip().lower(),
                     str(row[3]).strip().lower())
            ident = group + (self._norm_line(row[4]), _odds(row[5]), _odds(row[6]))
            if ident in seen_in_batch:
                batch_dupes += 1
                continue
            seen_in_batch.add(ident)
            valid.append({"row": row, "group": group, "source": rec.get("source"),
                          "line_value": row[4], "over_odds": row[5],
                          "under_odds": row[6]})

        if not valid:
            return {
                "inserted": 0,
                "duplicates_ignored": int(batch_dupes),
                "rejected_implausible": int(rejected_implausible),
                "attempted": 0,
            }

        # Tag each scrape-ladder's main rung.
        ladders: dict = {}
        for item in valid:
            ladders.setdefault(item["group"], []).append(item)
        alt_tagged = 0
        for ladder in ladders.values():
            main = main_line_index(ladder)
            for i, item in enumerate(ladder):
                item["is_main"] = 1 if i == main else 0
                alt_tagged += 1 - item["is_main"]

        main_sql = self.main_line_sql()
        latest_main_q = f"""
            SELECT line_value, over_odds, under_odds, line_id FROM betting_lines
            WHERE player_id = ? AND game_date = ? AND lower(book) = ?
              AND lower(stat_type) = ? AND {main_sql}
            ORDER BY scraped_at DESC, line_id DESC LIMIT 1
        """
        latest_alt_q = """
            SELECT over_odds, under_odds, line_id FROM betting_lines
            WHERE player_id = ? AND game_date = ? AND lower(book) = ?
              AND lower(stat_type) = ? AND is_main_line = 0
              AND round(line_value, 3) = ?
            ORDER BY scraped_at DESC, line_id DESC LIMIT 1
        """
        state: dict = {}
        current_row_id: dict = {}  # skey -> line_id of the stored current row
        payload = []
        bump_ids: set = set()
        for item in valid:
            row, group = item["row"], item["group"]
            line = self._norm_line(row[4])
            prices = (_odds(row[5]), _odds(row[6]))
            if item["is_main"]:
                skey = ("main",) + group
                if skey not in state:
                    got = self.conn.execute(latest_main_q, group).fetchone()
                    state[skey] = (None if got is None else
                                   (self._norm_line(got[0]), _odds(got[1]), _odds(got[2])))
                    current_row_id[skey] = None if got is None else got[3]
                current = (line,) + prices
            else:
                skey = ("alt",) + group + (line,)
                if skey not in state:
                    got = self.conn.execute(latest_alt_q, group + (line,)).fetchone()
                    state[skey] = (None if got is None else (_odds(got[0]), _odds(got[1])))
                    current_row_id[skey] = None if got is None else got[2]
                current = prices
            if state[skey] is not None and state[skey] == current:
                # Unchanged re-scrape: no new row, but the line was SEEN now.
                if current_row_id.get(skey) is not None:
                    bump_ids.add(current_row_id[skey])
                continue
            state[skey] = current
            current_row_id[skey] = None  # row being inserted this batch
            payload.append(row + (item["source"], item["is_main"]))

        query = """
            INSERT INTO betting_lines
                (player_id, game_date, book, stat_type, line_value,
                 over_odds, under_odds, source, is_main_line, last_seen_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """
        before_changes = self.conn.total_changes
        with self._atomic("betting_lines"):
            if payload:
                self.conn.executemany(query, payload)
            inserted = self.conn.total_changes - before_changes
            if bump_ids:
                # Same 'YYYY-MM-DD HH:MM:SS' UTC format as scraped_at's default.
                self.conn.executemany(
                    "UPDATE betting_lines SET last_seen_at_utc = datetime('now') "
                    "WHERE line_id = ?",
                    [(i,) for i in sorted(bump_ids)],
                )
        if payload or bump_ids:
            self.conn.commit()
        ignored = batch_dupes + (len(valid) - inserted)
        if rejected_implausible:
            logger.warning(
                "Dropped %s implausible betting_lines rows (likely scraper "
                "poisoning); see input_validation.is_plausible_betting_line",
                rejected_implausible,
            )
        logger.info(
            "Inserted %s betting_lines rows (%s duplicates ignored, "
            "%s implausible, %s alt rungs tagged)",
            inserted, ignored, rejected_implausible, alt_tagged,
        )
        return {
            "inserted": int(inserted),
            "duplicates_ignored": int(ignored),
            "rejected_implausible": int(rejected_implausible),
            "attempted": int(len(valid) + batch_dupes),
            "alt_lines_tagged": int(alt_tagged),
            "last_seen_bumped": int(len(bump_ids)),
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
        with self._atomic():
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
        with self._atomic():
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
            limit_total: hard cap for total snapshots returned. Applied
                newest-first, so the cap drops the OLDEST snapshots (it used
                to sort by URL and silently drop every alphabetically-late
                URL — sportsbook.*, www.bovada, www.vegasinsider — once
                stale URLs pushed the count past the cap).
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
                ORDER BY datetime(fetched_at_utc) DESC, snapshot_id DESC
            """
            rows = self.conn.execute(query, tuple(urls)).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT snapshot_id, source_url, fetched_at_utc, text_content, text_length, content_sha256
                FROM web_text_snapshots
                ORDER BY datetime(fetched_at_utc) DESC, snapshot_id DESC
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
        with self._atomic():
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

    @staticmethod
    def _norm_line(value):
        """Round a line to 3 decimals for change comparison; pass None through."""
        if value is None:
            return None
        try:
            return round(float(value), 3)
        except (TypeError, ValueError):
            return None

    def _latest_web_prop_card_line(self, book, player_name, stat_type, side):
        """Most-recently-observed line for a (book, player, stat, side), or None.

        Used to skip re-inserting a prop card whose line hasn't moved since the
        last scrape — a new row is only written when the book actually changes
        the line (or no prior row exists).
        """
        cur = self.conn.execute(
            """
            SELECT line_value
            FROM web_prop_cards
            WHERE book = ? AND player_name = ? AND stat_type = ? AND side = ?
            ORDER BY observed_at_utc DESC, card_id DESC
            LIMIT 1
            """,
            (book, player_name, stat_type, side),
        )
        row = cur.fetchone()
        return None if row is None else self._norm_line(row[0])

    def _latest_web_team_line_state(
        self, book, away_team, home_team, market_type, side, team, game_date=None
    ):
        """Most-recent (line_value, odds) for a team-line key, or None.

        ``team`` / ``game_date`` may be NULL; ``IS`` matches NULL and values
        alike, so two games of the same matchup are separate keys.
        """
        cur = self.conn.execute(
            """
            SELECT line_value, odds_american
            FROM web_team_lines
            WHERE book = ? AND away_team = ? AND home_team = ?
              AND market_type = ? AND side = ? AND team IS ? AND game_date IS ?
            ORDER BY observed_at_utc DESC, line_id DESC
            LIMIT 1
            """,
            (book, away_team, home_team, market_type, side, team, game_date),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return (self._norm_line(row[0]), row[1])

    def insert_web_prop_cards(self, records):
        """
        Insert parsed web prop cards with dedupe via record_sha256.

        Beyond the per-snapshot ``record_sha256`` uniqueness, this also skips
        any card whose line equals the most-recently-stored line for the same
        (book, player, stat, side): re-scraping an unchanged game adds no row,
        and a new row lands only when the book moves the line. A skipped
        re-scrape still bumps the current row's ``last_seen_at_utc`` so
        freshness windows keep lines that are stable across days.

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
                record_sha256,
                last_seen_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        payload = []
        skipped_unchanged = 0
        # key -> newest observed_at among unchanged re-scrapes (last_seen bump).
        seen_unchanged: dict = {}
        # Cache of the line we consider "current" per (book, player, stat, side),
        # seeded from the DB and updated as we accept rows so two unchanged
        # cards in the same batch don't both insert.
        latest_line = {}
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

            key = (row[2], row[4], row[6], row[8])  # book, player, stat, side
            if key not in latest_line:
                latest_line[key] = self._latest_web_prop_card_line(*key)
            new_line = self._norm_line(line_value)
            if latest_line[key] is not None and latest_line[key] == new_line:
                skipped_unchanged += 1
                if row[3] > seen_unchanged.get(key, ""):
                    seen_unchanged[key] = row[3]
                continue
            latest_line[key] = new_line
            payload.append(row + (row[3],))  # last_seen starts at observed

        prop_key_sql = "book = ? AND player_name = ? AND stat_type = ? AND side = ?"
        if not payload:
            with self._atomic("prop_cards"):
                self._bump_last_seen("web_prop_cards", "card_id", prop_key_sql, seen_unchanged)
            self.conn.commit()
            return {
                "inserted": 0,
                "attempted": 0,
                "skipped_unchanged": skipped_unchanged,
            }

        before_changes = self.conn.total_changes
        with self._atomic("prop_cards"):
            self.conn.executemany(query, payload)
            inserted_changes = self.conn.total_changes - before_changes
            # After the inserts, so an in-batch duplicate bumps the row this batch wrote.
            self._bump_last_seen("web_prop_cards", "card_id", prop_key_sql, seen_unchanged)
        self.conn.commit()
        inserted = inserted_changes
        logger.info(
            "Inserted %s web_prop_cards rows (%s skipped: line unchanged)",
            inserted,
            skipped_unchanged,
        )
        return {
            "inserted": int(inserted),
            "attempted": int(len(payload)),
            "skipped_unchanged": int(skipped_unchanged),
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
                # Last SEEN, not last changed: stable lines stay in-window.
                clauses.append(
                    f"datetime({self.seen_at_sql()}) >= datetime('now', ?)"
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
                    {self.seen_at_sql()} AS seen_at_utc,
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
                MAX(seen_at_utc)              AS latest_observed_at
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
        """Insert game-level team lines with dedupe via record_sha256.

        Also skips any row whose (line_value, odds) equals the most-recent
        stored values for the same (book, away, home, market, side, team):
        re-scraping the same game adds nothing, and a new row is written only
        when the book moves the line or odds. A skipped re-scrape bumps the
        current row's ``last_seen_at_utc`` (see ``insert_web_prop_cards``).
        """
        if not records:
            return {"inserted": 0, "attempted": 0, "skipped_unchanged": 0}

        query = """
            INSERT OR IGNORE INTO web_team_lines (
                snapshot_id, source_url, book, observed_at_utc,
                away_team, home_team, market_type, side, team,
                line_value, odds_american,
                parse_confidence, raw_text, parser_version, record_sha256,
                sport, game_date, last_seen_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        payload = []
        skipped_unchanged = 0
        seen_unchanged: dict = {}
        # (book, away, home, market, side, team) -> last (line, odds) we kept.
        latest_state = {}
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
                (str(rec.get("sport", "nba")).strip().lower() or "nba"),
                (str(rec.get("game_date") or "").strip()[:10] or None),
            )
            # Required: source_url, book, observed_at_utc, both teams,
            # market_type, side, parser_version, record_sha256.
            if not all([row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[13], row[14]]):
                continue

            # book, away, home, market, side, team, game_date
            key = (row[2], row[4], row[5], row[6], row[7], row[8], row[16])
            if key not in latest_state:
                latest_state[key] = self._latest_web_team_line_state(*key)
            new_state = (self._norm_line(line_value), odds)
            if latest_state[key] is not None and latest_state[key] == new_state:
                skipped_unchanged += 1
                if row[3] > seen_unchanged.get(key, ""):
                    seen_unchanged[key] = row[3]
                continue
            latest_state[key] = new_state
            payload.append(row + (row[3],))  # last_seen starts at observed

        team_key_sql = (
            "book = ? AND away_team = ? AND home_team = ? "
            "AND market_type = ? AND side = ? AND team IS ? AND game_date IS ?"
        )
        if not payload:
            with self._atomic("team_lines"):
                self._bump_last_seen("web_team_lines", "line_id", team_key_sql, seen_unchanged)
            self.conn.commit()
            return {"inserted": 0, "attempted": 0, "skipped_unchanged": skipped_unchanged}

        before = self.conn.total_changes
        with self._atomic("team_lines"):
            self.conn.executemany(query, payload)
            inserted = self.conn.total_changes - before
            self._bump_last_seen("web_team_lines", "line_id", team_key_sql, seen_unchanged)
        self.conn.commit()
        logger.info(
            "Inserted %s web_team_lines rows (%s skipped: line unchanged)",
            inserted,
            skipped_unchanged,
        )
        return {
            "inserted": int(inserted),
            "attempted": int(len(payload)),
            "skipped_unchanged": int(skipped_unchanged),
        }

    def get_consensus_team_lines(
        self,
        away_team=None,
        home_team=None,
        market_type=None,
        side=None,
        since_hours=None,
        min_books=1,
        sport="nba",
    ):
        """Return cross-book consensus for game-level markets.

        For each (away_team, home_team, game_date, market_type, side) the
        latest line from each book is selected, then averaged across books.
        Returns one row per (game, market, side) with mean line, mean odds,
        the list of contributing books and ``game_date`` (None = undated).

        A game is (away, home, game_date): two meetings of the same teams in
        the window stay separate. A row from a book that prints no date joins
        the matchup's dated game only when exactly ONE dated game exists for
        that matchup in the window; otherwise it stays undated.

        ``sport`` defaults to 'nba' so NBA consensus never sees MLB (or other
        non-NBA) rows; pass ``sport=None`` to span all sports or 'mlb' for MLB.
        """
        clauses: list[str] = []
        params: list = []
        if sport is not None:
            clauses.append("lower(sport) = lower(?)")
            params.append(str(sport).strip())
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
                # Last SEEN, not last changed: stable lines stay in-window.
                clauses.append(
                    f"datetime({self.seen_at_sql()}) >= datetime('now', ?)"
                )
                params.append(f"-{hours} hours")
        where_sql = (" AND ".join(clauses)) if clauses else "1=1"

        # Pull the latest line per (game, market, side, book) and aggregate
        # in Python so we can convert odds → implied probability → mean →
        # American.  Averaging raw American odds is mathematically wrong
        # whenever the sample contains both + and - values (the signs flip
        # the magnitude, distorting the mean).
        query = f"""
            WITH base AS (
                SELECT line_id, away_team, home_team, market_type, side, book,
                       line_value, odds_american, observed_at_utc, game_date,
                       {self.seen_at_sql()} AS seen_at_utc
                FROM web_team_lines
                WHERE {where_sql}
            ),
            dated AS (
                SELECT lower(away_team) AS a, lower(home_team) AS h,
                       MIN(game_date) AS only_date,
                       COUNT(DISTINCT game_date) AS n_dates
                FROM base WHERE game_date IS NOT NULL
                GROUP BY lower(away_team), lower(home_team)
            ),
            resolved AS (
                SELECT b.*,
                       COALESCE(b.game_date,
                                CASE WHEN d.n_dates = 1 THEN d.only_date END) AS gd
                FROM base b
                LEFT JOIN dated d
                  ON d.a = lower(b.away_team) AND d.h = lower(b.home_team)
            ),
            latest_per_book AS (
                SELECT
                    away_team, home_team, gd, market_type, side,
                    book, line_value, odds_american, seen_at_utc,
                    ROW_NUMBER() OVER (
                        PARTITION BY lower(away_team), lower(home_team), gd,
                                     lower(market_type), lower(side), lower(book)
                        ORDER BY observed_at_utc DESC, line_id DESC
                    ) AS rn
                FROM resolved
            )
            SELECT
                away_team, home_team, market_type, side,
                book, line_value, odds_american, seen_at_utc, gd
            FROM latest_per_book
            WHERE rn = 1
            ORDER BY away_team, home_team, gd, market_type, side, book
        """
        rows = self.conn.execute(query, tuple(params)).fetchall()

        # Group rows by (away, home, game_date, market, side) and aggregate.
        groups: dict[tuple, dict] = {}
        for away, home, market, side, book, line, odds, obs_at, gd in rows:
            key = (away.lower(), home.lower(), gd, market.lower(), side.lower())
            slot = groups.setdefault(key, {
                "away_team": away, "home_team": home, "game_date": gd,
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
                "game_date": slot["game_date"],
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
            r["home_team"], r["away_team"], r["game_date"] or "",
            r["market_type"], r["side"],
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
        with self._atomic():
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
            with self._atomic("game_logs"):
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
            # _atomic already rolled this writer's rows back (savepoint), so a
            # caller's unrelated pending work isn't discarded here.
            logger.error("Error inserting game logs: %s", e)
            raise

    def insert_mlb_game_logs(self, records):
        """Insert MLB long-format game-log rows, skipping exact duplicates.

        Each record: player_id, player_name, team, opponent, game_pk,
        game_date, season, player_group ('hitting'|'pitching'), stat_type,
        value. Dedup is by ``UNIQUE(player_id, game_pk, stat_type)`` via
        ``INSERT OR IGNORE`` — game-log results are immutable, so re-running
        the ingest never double-counts.
        """
        if not records:
            return {"inserted": 0, "duplicates_ignored": 0, "attempted": 0}

        query = """
            INSERT OR IGNORE INTO mlb_game_logs
                (sport, player_id, player_name, team, opponent, game_pk,
                 game_date, season, player_group, stat_type, value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        payload = []
        for rec in records:
            try:
                player_id = int(rec.get("player_id"))
                game_pk = int(rec.get("game_pk"))
                value = float(rec.get("value"))
            except (TypeError, ValueError):
                continue
            stat_type = str(rec.get("stat_type", "")).strip().lower()
            game_date = str(rec.get("game_date", "")).strip()
            group = str(rec.get("player_group", "")).strip().lower()
            if not (stat_type and game_date and group):
                continue
            season = rec.get("season")
            payload.append((
                str(rec.get("sport", "mlb")).strip().lower() or "mlb",
                player_id,
                (str(rec.get("player_name")).strip() or None) if rec.get("player_name") else None,
                (str(rec.get("team")).strip() or None) if rec.get("team") else None,
                (str(rec.get("opponent")).strip() or None) if rec.get("opponent") else None,
                game_pk,
                game_date,
                int(season) if season is not None else None,
                group,
                stat_type,
                value,
            ))

        if not payload:
            return {"inserted": 0, "duplicates_ignored": 0, "attempted": 0}

        before = self.conn.total_changes
        with self._atomic():
            self.conn.executemany(query, payload)
        self.conn.commit()
        inserted = self.conn.total_changes - before
        logger.info(
            "Inserted %s mlb_game_logs rows (%s duplicates ignored)",
            inserted, len(payload) - inserted,
        )
        return {
            "inserted": int(inserted),
            "duplicates_ignored": int(len(payload) - inserted),
            "attempted": int(len(payload)),
        }

    def insert_mlb_prop_lines(self, records):
        """Insert MLB prop-line observations with change-only semantics.

        Each record: snapshot_id, source_url, source, book, observed_at_utc,
        game_date, player_name, stat_type, stat_group, market_shape,
        line_value, side, over_odds, under_odds, parser_version,
        record_sha256. Beyond ``record_sha256`` uniqueness, a row is skipped
        when its (line, over_odds, under_odds) equals the most-recently-stored
        state for the same (book, player, stat, side, game_date) — re-scraping
        an unchanged board adds nothing; a line OR price move lands a new row,
        and the next slate's first observation always lands (game_date is part
        of the key, so an identical price on a new day is not "unchanged").
        """
        if not records:
            return {"inserted": 0, "attempted": 0, "skipped_unchanged": 0}

        query = """
            INSERT OR IGNORE INTO mlb_prop_lines (
                sport, snapshot_id, source_url, source, book, observed_at_utc,
                game_date, player_name, stat_type, stat_group, market_shape,
                line_value, side, over_odds, under_odds, parser_version,
                record_sha256
            )
            VALUES ('mlb', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        def _odds(v):
            return None if v is None else int(v)

        payload = []
        skipped_unchanged = 0
        latest = {}
        for rec in records:
            try:
                line_value = float(rec.get("line_value"))
                over_odds = _odds(rec.get("over_odds"))
                under_odds = _odds(rec.get("under_odds"))
            except (TypeError, ValueError):
                continue
            snapshot_id = rec.get("snapshot_id")
            row = (
                int(snapshot_id) if snapshot_id is not None else None,
                str(rec.get("source_url") or "").strip() or None,
                str(rec.get("source", "")).strip(),
                str(rec.get("book", "")).strip(),
                str(rec.get("observed_at_utc", "")).strip(),
                str(rec.get("game_date") or "").strip() or None,
                str(rec.get("player_name", "")).strip(),
                str(rec.get("stat_type", "")).strip().lower(),
                str(rec.get("stat_group", "")).strip().lower(),
                str(rec.get("market_shape", "")).strip().lower(),
                line_value,
                str(rec.get("side", "")).strip().lower(),
                over_odds,
                under_odds,
                str(rec.get("parser_version", "")).strip(),
                str(rec.get("record_sha256", "")).strip(),
            )
            if not all(row[i] for i in (2, 3, 4, 6, 7, 8, 9, 11, 14, 15)):
                continue

            # book, player, stat, side, game_date
            key = (row[3], row[6], row[7], row[11], row[5])
            if key not in latest:
                prev = self.conn.execute(
                    """
                    SELECT line_value, over_odds, under_odds
                    FROM mlb_prop_lines
                    WHERE book = ? AND player_name = ? AND stat_type = ? AND side = ?
                      AND game_date IS ?
                    ORDER BY observed_at_utc DESC, mlb_prop_line_id DESC
                    LIMIT 1
                    """,
                    key,
                ).fetchone()
                latest[key] = (
                    None if prev is None
                    else (self._norm_line(prev[0]), prev[1], prev[2])
                )
            state = (self._norm_line(line_value), over_odds, under_odds)
            if latest[key] is not None and latest[key] == state:
                skipped_unchanged += 1
                continue
            latest[key] = state
            payload.append(row)

        if not payload:
            return {"inserted": 0, "attempted": 0,
                    "skipped_unchanged": int(skipped_unchanged)}

        before = self.conn.total_changes
        with self._atomic():
            self.conn.executemany(query, payload)
        self.conn.commit()
        inserted = self.conn.total_changes - before
        logger.info(
            "Inserted %s mlb_prop_lines rows (%s skipped: unchanged)",
            inserted, skipped_unchanged,
        )
        return {
            "inserted": int(inserted),
            "attempted": int(len(payload)),
            "skipped_unchanged": int(skipped_unchanged),
        }

    def get_mlb_player_game_logs(self, player_id, stat_type, n_games=50):
        """Most-recent N MLB game-log values for a player+stat (sport='mlb')."""
        cur = self.conn.execute(
            """
            SELECT game_date, game_pk, team, opponent, value
            FROM mlb_game_logs
            WHERE player_id = ? AND lower(stat_type) = lower(?) AND sport = 'mlb'
            ORDER BY game_date DESC, game_pk DESC
            LIMIT ?
            """,
            (int(player_id), str(stat_type), int(n_games)),
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

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

    def delete_nonfinite_predictions(self):
        """Delete predictions with a non-finite projected moment (one-off cleanup).

        ``predictions`` rows persisted before the NULL-in-window μ/σ fix could
        carry a NaN ``predicted_mean`` / ``predicted_std`` / ``prob_over``.
        SQLite stores a Python ``float('nan')`` REAL as NULL, so those broken
        rows are exactly the ones with a NULL in any of those three columns
        (``insert_prediction`` always binds a real float, so a NULL there is the
        NaN signature, not a legitimately-missing value). Returns the row count
        removed. Idempotent."""
        cursor = self.conn.execute(
            """
            DELETE FROM predictions
            WHERE predicted_mean IS NULL
               OR predicted_std IS NULL
               OR prob_over IS NULL
            """
        )
        self.conn.commit()
        return int(cursor.rowcount)

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
                AND DATE(p.game_date) = DATE(gl.game_date)
            WHERE p.game_date BETWEEN ? AND ?
              AND p.actual_result IS NOT NULL
            ORDER BY p.game_date DESC
        """
        return pd.read_sql_query(query, self.conn, params=(start_date, end_date))

    def _latest_main_line_values(self, where_sql: str, params) -> list:
        """Each book's CURRENT main line for the filter: latest main-line row
        per lower(book) (newest scraped_at, then line_id). Aggregating these —
        not every historical row / alt rung — is what "the market line" means.
        """
        query = f"""
            WITH ranked AS (
                SELECT line_value,
                       ROW_NUMBER() OVER (
                           PARTITION BY lower(book)
                           ORDER BY scraped_at DESC, line_id DESC
                       ) AS rn
                FROM betting_lines
                WHERE {where_sql} AND {self.main_line_sql()}
            )
            SELECT line_value FROM ranked WHERE rn = 1
        """
        return [r[0] for r in self.conn.execute(query, tuple(params)).fetchall()
                if r and r[0] is not None]

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

        clauses = ["player_id = ?", "game_date = ?", "stat_type = ?"]
        params = [player_id, game_date, stat_type]
        if book:
            clauses.append("lower(book) = lower(?)")
            params.append(book)
        rows = self._latest_main_line_values(" AND ".join(clauses), params)
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
        clauses = ["player_id = ?", "game_date = ?",
                   f"lower(stat_type) IN ({placeholders})"]
        params = [player_id, game_date, *spread_aliases]
        if book:
            clauses.append("lower(book) = lower(?)")
            params.append(book)
        rows = self._latest_main_line_values(" AND ".join(clauses), params)
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
            self.conn = None
            logger.info("Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
