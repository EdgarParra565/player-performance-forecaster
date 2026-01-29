from nba_api.stats.endpoints import playergamelogs
from nba_api.stats.static import players
import pandas as pd
from datetime import datetime, timedelta
import time


class NBADataFetcher:
    """Fetches and caches NBA player game logs."""

    def __init__(self, cache_dir='data/cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_player_id(self, player_name):
        """Get NBA API player ID from name."""
        player_dict = players.find_players_by_full_name(player_name)
        if not player_dict:
            raise ValueError(f"Player {player_name} not found")
        return player_dict[0]['id']

    def fetch_recent_games(self, player_name, n_games=50):
        """
        Fetch most recent N games for a player.
        Implements rate limiting and caching.
        """
        player_id = self.get_player_id(player_name)

        # Check cache first
        cache_file = f"{self.cache_dir}/{player_id}_{datetime.now().strftime('%Y%m%d')}.csv"
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file)

        # Fetch from API with rate limiting
        time.sleep(0.6)  # NBA API rate limit: ~1 req/sec

        gamelog = playergamelogs.PlayerGameLogs(
            season_nullable='2024-25',
            player_id_nullable=player_id
        )
        df = gamelog.get_data_frames()[0]

        # Cache results
        df.to_csv(cache_file, index=False)

        return df.head(n_games)