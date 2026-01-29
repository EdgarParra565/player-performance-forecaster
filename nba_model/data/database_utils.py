import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

load_dotenv()


class Database:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=os.getenv('SUPABASE_HOST'),
            database=os.getenv('SUPABASE_DB'),
            user=os.getenv('SUPABASE_USER'),
            password=os.getenv('SUPABASE_PASSWORD')
        )

    def insert_game_log(self, game_data):
        """Insert game log into database."""
        with self.conn.cursor() as cur:
            query = """
                    INSERT INTO game_logs
                    (game_id, player_id, game_date, opponent, minutes, points, rebounds, assists)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (game_id) DO NOTHING \
                    """
            cur.execute(query, game_data)
        self.conn.commit()

    def get_recent_games(self, player_id, n_games=50):
        """Fetch recent games from database."""
        query = """
                SELECT * \
                FROM game_logs
                WHERE player_id = %s
                ORDER BY game_date DESC
                    LIMIT %s \
                """
        return pd.read_sql(query, self.conn, params=(player_id, n_games))

