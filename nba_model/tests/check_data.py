from nba_model.data.data_loader import DataLoader
import pandas as pd

loader = DataLoader()

# Load LeBron's recent games
df = loader.load_player_data("LeBron James", n_games=30)
df['game_date'] = pd.to_datetime(df['game_date'])

print(f"✓ Loaded {len(df)} games for LeBron James")
print(f"\nDate range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
print(f"\nMost recent games:")
print(df[['game_date', 'matchup', 'points', 'assists', 'rebounds']].head(10))

# Suggest good backtest dates
latest_game = df['game_date'].max()
earliest_game = df['game_date'].min()

print(f"\n📅 Suggested backtest period:")
print(f"   start_date = '{earliest_game.strftime('%Y-%m-%d')}'")
print(f"   end_date = '{latest_game.strftime('%Y-%m-%d')}'")