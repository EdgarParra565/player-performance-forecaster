from nba_model.data.database.db_manager import DatabaseManager
import pandas as pd

# Initialize database
db = DatabaseManager()

# Test insert player
db.insert_player(2544, "LeBron James", "LAL", "F")
print("✓ Player inserted")

# Test insert game logs
sample_data = pd.DataFrame({
    'player_id': [2544],
    'game_id': ['0022400001'],
    'game_date': ['2024-10-22'],
    'season': ['2024-25'],
    'matchup': ['LAL vs. DEN'],
    'home_away': ['home'],
    'result': ['W'],
    'minutes': [35.5],
    'points': [28],
    'fgm': [10],
    'fga': [20],
    'fg_pct': [0.5],
    'fg3m': [2],
    'fg3a': [6],
    'fg3_pct': [0.333],
    'ftm': [6],
    'fta': [8],
    'ft_pct': [0.75],
    'oreb': [1],
    'dreb': [7],
    'rebounds': [8],
    'assists': [10],
    'steals': [2],
    'blocks': [1],
    'turnovers': [3],
    'plus_minus': [12]
})

db.insert_game_logs(sample_data)
print("✓ Game log inserted")

# Test query
games = db.get_player_games(2544, n_games=5)
print(f"✓ Retrieved {len(games)} games")
print(games[['game_date', 'points', 'assists', 'rebounds']])

db.close()