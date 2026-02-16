import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


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
        schema_path = Path(__file__).parent / 'schema.sql'
        with open(schema_path, 'r') as f:
            self.conn.executescript(f.read())

        logger.info(f"Database initialized at {self.db_path}")

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
        logger.info(f"✓ Upserted {len(payload)} team_defense rows")

    def insert_betting_lines_records(self, records):
        """
        Insert betting lines while skipping exact duplicates.

        Args:
            records: Iterable of dicts with keys:
                player_id, game_date, book, stat_type, line_value, over_odds, under_odds
        """
        if not records:
            return

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

        payload = []
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
            payload.append(row + row)

        if not payload:
            return

        before_changes = self.conn.total_changes
        self.conn.executemany(query, payload)
        self.conn.commit()
        inserted = self.conn.total_changes - before_changes
        ignored = len(payload) - inserted
        logger.info(f"✓ Inserted {inserted} betting_lines rows ({ignored} duplicates ignored)")

    def insert_game_logs(self, game_logs_df):
        """
        Bulk insert game logs from DataFrame.

        Args:
            game_logs_df: DataFrame with columns matching game_logs table
        """
        try:
            if game_logs_df is None or game_logs_df.empty:
                return

            # Remove duplicates before inserting
            game_logs_df = game_logs_df.drop_duplicates(subset=['player_id', 'game_id'])

            # Keep only actual table columns and rely on INSERT OR IGNORE for
            # existing rows already present in SQLite.
            table_columns = {
                row[1]
                for row in self.conn.execute("PRAGMA table_info(game_logs)").fetchall()
            }
            blocked_columns = {'game_log_id', 'created_at'}
            columns = [col for col in game_logs_df.columns if col in table_columns and col not in blocked_columns]
            if not columns:
                logger.warning("No valid game_logs columns to insert")
                return

            payload = game_logs_df[columns].where(pd.notna(game_logs_df[columns]), None)
            query = f"""
                INSERT OR IGNORE INTO game_logs ({", ".join(columns)})
                VALUES ({", ".join(["?"] * len(columns))})
            """

            before_changes = self.conn.total_changes
            self.conn.executemany(query, payload.itertuples(index=False, name=None))
            self.conn.commit()
            inserted = self.conn.total_changes - before_changes
            ignored = len(payload) - inserted
            logger.info(f"✓ Inserted {inserted} game logs ({ignored} duplicates ignored)")
        except Exception as e:
            logger.error(f"Error inserting game logs: {e}")
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

        rows = [r[0] for r in self.conn.execute(query, params).fetchall() if r and r[0] is not None]
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
        spread_aliases = sorted({str(alias).strip().lower() for alias in spread_aliases if str(alias).strip()})
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

        rows = [r[0] for r in self.conn.execute(query, params).fetchall() if r and r[0] is not None]
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