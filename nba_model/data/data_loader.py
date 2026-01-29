from nba_api.stats.endpoints import playergamelogs
from nba_api.stats.static import players
import pandas as pd
import time
from pathlib import Path
import json
import os
from nba_model.data.database.db_manager import DatabaseManager

class DataLoader:
    """Load NBA data with local caching."""

    def __init__(self, cache_dir='data/raw', db_path='data/database/nba_data.db'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db = DatabaseManager(db_path=db_path)  # ← Pass the path

    def get_player_id(self, player_name):
        """Get player ID from name."""
        player_dict = players.find_players_by_full_name(player_name)
        if not player_dict:
            raise ValueError(f"Player '{player_name}' not found")
        return player_dict[0]['id']

    def load_player_data(self, player_name, n_games=50, force_refresh=False):
        """
        Load player game logs with multi-tier caching:
        1. Check database
        2. Check file cache
        3. Fetch from API

        Args:
            player_name: Full player name
            n_games: Number of recent games
            force_refresh: Skip cache and fetch fresh data

        Returns:
            pd.DataFrame: Game logs
        """
        player_id = self.get_player_id(player_name)

        # Check database first (unless force refresh)
        if not force_refresh:
            df = self.db.get_player_games(player_id, n_games)
            if not df.empty and len(df) >= n_games:
                print(f"✓ Loaded {len(df)} games from database")
                return df

        # Check file cache
        cache_file = self.cache_dir / f"{player_id}_gamelogs.json"
        if cache_file.exists() and not force_refresh:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                if len(df) >= n_games:
                    print(f"✓ Loaded {len(df)} games from cache")
                    return df.head(n_games)

        # Fetch from API
        print(f"→ Fetching fresh data from NBA API for {player_name}...")
        time.sleep(0.6)  # Rate limiting

        gamelog = playergamelogs.PlayerGameLogs(
            season_nullable='2024-25',
            player_id_nullable=player_id
        )
        df = gamelog.get_data_frames()[0]

        # Clean and standardize column names
        df = self._clean_game_logs(df, player_id)

        # Save to file cache
        df.to_json(cache_file, orient='records', date_format='iso')

        # Save to database
        self.db.insert_game_logs(df)

        print(f"✓ Fetched and cached {len(df)} games")
        return df.head(n_games)

    def _clean_game_logs(self, df, player_id):
        """Standardize game log format for database."""
        # Map API columns to database columns
        column_mapping = {
            'PLAYER_ID': 'player_id',
            'GAME_ID': 'game_id',
            'GAME_DATE': 'game_date',
            'SEASON_YEAR': 'season',
            'MATCHUP': 'matchup',
            'MIN': 'minutes',
            'PTS': 'points',
            'FGM': 'fgm',
            'FGA': 'fga',
            'FG_PCT': 'fg_pct',
            'FG3M': 'fg3m',
            'FG3A': 'fg3a',
            'FG3_PCT': 'fg3_pct',
            'FTM': 'ftm',
            'FTA': 'fta',
            'FT_PCT': 'ft_pct',
            'OREB': 'oreb',
            'DREB': 'dreb',
            'REB': 'rebounds',
            'AST': 'assists',
            'STL': 'steals',
            'BLK': 'blocks',
            'TOV': 'turnovers',
            'PLUS_MINUS': 'plus_minus',
            'WL': 'result'
        }

        # Only rename columns that exist in the dataframe
        existing_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_cols)

        # Ensure player_id is set
        df['player_id'] = player_id

        # Extract home/away
        if 'matchup' in df.columns:
            df['home_away'] = df['matchup'].apply(
                lambda x: 'home' if 'vs.' in str(x) else 'away'
            )
        else:
            df['home_away'] = 'unknown'

        # Convert minutes from "MM:SS" to decimal
        if 'minutes' in df.columns and df['minutes'].dtype == 'object':
            df['minutes'] = df['minutes'].apply(self._convert_minutes)

        # Return only columns that exist (to avoid KeyError)
        return_cols = [v for v in column_mapping.values() if v in df.columns] + ['home_away']
        return df[return_cols]

        df = df.rename(columns=column_mapping)
        df['player_id'] = player_id

        # Extract home/away
        df['home_away'] = df['matchup'].apply(
            lambda x: 'home' if 'vs.' in x else 'away'
        )

        # Convert minutes from "MM:SS" to decimal
        if df['minutes'].dtype == 'object':
            df['minutes'] = df['minutes'].apply(self._convert_minutes)

        return df[list(column_mapping.values()) + ['home_away']]

    def _convert_minutes(self, time_str):
        """Convert 'MM:SS' to decimal minutes."""
        if pd.isna(time_str) or time_str == '':
            return 0.0
        try:
            parts = str(time_str).split(':')
            return float(parts[0]) + float(parts[1]) / 60
        except:
            return 0.0