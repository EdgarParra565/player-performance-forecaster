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

    def insert_game_logs(self, game_logs_df):
        """
        Bulk insert game logs from DataFrame.

        Args:
            game_logs_df: DataFrame with columns matching game_logs table
        """
        try:
            # Remove duplicates before inserting
            game_logs_df = game_logs_df.drop_duplicates(subset=['player_id', 'game_id'])

            game_logs_df.to_sql(
                'game_logs',
                self.conn,
                if_exists='append',
                index=False
            )
            self.conn.commit()
            logger.info(f"✓ Inserted {len(game_logs_df)} game logs")
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
                predicted_std, prob_over, line_value, expected_value
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
        self.conn.execute(query, values)
        self.conn.commit()

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

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()