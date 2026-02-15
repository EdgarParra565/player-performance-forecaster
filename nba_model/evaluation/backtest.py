import pandas as pd
import numpy as np
from nba_model.data.data_loader import DataLoader
from nba_model.model.feature_engineering import add_rolling_stats
from nba_model.model.probability import prob_over
from nba_model.data.database.db_manager import DatabaseManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """
    Evaluate model performance on historical data.

    Simulates making predictions on past games and compares to actual outcomes.
    """

    def __init__(self, start_date, end_date, line_value=None, stat_type='points'):
        """
        Args:
            start_date: Start of backtest period (YYYY-MM-DD)
            end_date: End of backtest period (YYYY-MM-DD)
            line_value: Fixed betting line (e.g., 27.5 points). If None, uses rolling average.
            stat_type: 'points', 'assists', or 'rebounds'
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.line_value = line_value
        self.stat_type = stat_type.lower()
        self.results = []
        self.loader = DataLoader()
        self.db = DatabaseManager()

        # Validate dates
        today = pd.Timestamp.now()
        if self.end_date > today:
            logger.warning(f"End date {self.end_date.date()} is in the future. Using today instead.")
            self.end_date = today

        if self.start_date > today:
            raise ValueError(f"Start date {self.start_date.date()} cannot be in the future!")

        if self.start_date >= self.end_date:
            raise ValueError(f"Start date must be before end date!")

        valid_stats = {'points', 'assists', 'rebounds'}
        if self.stat_type not in valid_stats:
            raise ValueError(f"stat_type must be one of {sorted(valid_stats)}")

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize historical input data to a stable lowercase schema."""
        rename_map = {
            'GAME_DATE': 'game_date',
            'PTS': 'points',
            'AST': 'assists',
            'REB': 'rebounds',
            'MIN': 'minutes',
        }
        existing = {src: dst for src, dst in rename_map.items() if src in df.columns and dst not in df.columns}
        if existing:
            df = df.rename(columns=existing)
        return df


    def run_backtest(self, player_name, window=10):
        """
        Run backtest for a single player over the date range.

        Args:
            player_name: Full player name (e.g., "LeBron James")
            window: Rolling window size for feature engineering

        Returns:
            dict: Performance metrics
        """
        logger.info(f"Running backtest for {player_name} from {self.start_date.date()} to {self.end_date.date()}")
        self.results = []

        player_id = self.loader.get_player_id(player_name)

        # Load ALL historical data (need enough for rolling window)
        all_games = self.loader.load_player_data(player_name, n_games=200)
        all_games = self._normalize_columns(all_games)
        if self.stat_type not in all_games.columns:
            raise KeyError(f"Missing stat column '{self.stat_type}' in loaded game logs")
        all_games['game_date'] = pd.to_datetime(all_games['game_date'])
        all_games = all_games.sort_values('game_date')

        # Filter to backtest period
        test_games = all_games[
            (all_games['game_date'] >= self.start_date) &
            (all_games['game_date'] <= self.end_date)
            ]

        logger.info(f"Found {len(test_games)} games in backtest period")

        # For each game in test period, make prediction using only prior data
        for idx, game in test_games.iterrows():
            game_date = game['game_date']

            # Get only data BEFORE this game
            train_data = all_games[all_games['game_date'] < game_date].tail(50)

            if len(train_data) < window:
                logger.warning(f"Skipping {game_date.date()} - insufficient history")
                continue

            # Make prediction
            prediction = self._make_prediction(train_data, game, window)

            # Compare to actual result
            actual_value = game[self.stat_type]
            line = self.line_value or prediction['expected_value']

            outcome = self._evaluate_outcome(actual_value, line, prediction['prob_over'])

            # Store result
            result = {
                'date': game_date,
                'player_id': player_id,
                'player_name': player_name,
                'predicted_mean': prediction['expected_value'],
                'predicted_std': prediction['std_dev'],
                'prob_over': prediction['prob_over'],
                'line': line,
                'actual_value': actual_value,
                'outcome': outcome['result'],  # 'over', 'under', 'push'
                'bet_recommendation': outcome['bet'],  # 'over', 'under', 'none'
                'correct': outcome['correct'],
                'profit': outcome['profit']
            }

            self.results.append(result)

            # Optionally save to database
            self._save_prediction_to_db(result)

        # Calculate aggregate metrics
        return self._calculate_metrics()

    def _make_prediction(self, historical_data, upcoming_game, window):
        """
        Generate prediction for a single game using only historical data.

        Args:
            historical_data: DataFrame of past games
            upcoming_game: Series representing the upcoming game
            window: Rolling window size

        Returns:
            dict: prediction with expected_value, std_dev, prob_over
        """
        historical_data = self._normalize_columns(historical_data)

        # Calculate rolling stats
        stats = add_rolling_stats(historical_data.copy(), window=window)
        mean_col = f'rolling_mean_{self.stat_type}'
        std_col = f'rolling_std_{self.stat_type}'

        if stats.empty or len(stats) < window or mean_col not in stats.columns or std_col not in stats.columns:
            # Fallback to simple average
            mean_stat = historical_data[self.stat_type].mean()
            std_stat = historical_data[self.stat_type].std()
        else:
            latest_valid = stats.dropna(subset=[mean_col, std_col])
            if latest_valid.empty:
                mean_stat = historical_data[self.stat_type].mean()
                std_stat = historical_data[self.stat_type].std()
            else:
                latest_stats = latest_valid.iloc[-1]
                mean_stat = latest_stats[mean_col]
                std_stat = latest_stats[std_col]

        # Apply adjustments (defense, minutes, etc.)
        # Note: You'll need to implement these if you want full accuracy
        expected_value = float(mean_stat)
        std_stat = float(std_stat) if pd.notna(std_stat) else 0.0

        # Calculate probability of exceeding the line
        line = self.line_value or expected_value
        prob_over1 = prob_over(line, expected_value, std_stat)

        return {
            'expected_value': expected_value,
            'std_dev': std_stat,
            'prob_over': prob_over1
        }

    def _evaluate_outcome(self, actual, line, prob_over):
        """
        Evaluate prediction outcome and calculate profit.

        Args:
            actual: Actual stat value
            line: Betting line
            prob_over: Model's probability of going over

        Returns:
            dict with result, bet, correct, profit
        """
        # Determine actual result
        if actual > line:
            result = 'over'
        elif actual < line:
            result = 'under'
        else:
            result = 'push'

        # Betting strategy: bet when model has >55% confidence
        edge_threshold = 0.55

        if prob_over > edge_threshold:
            bet = 'over'
        elif prob_over < (1 - edge_threshold):
            bet = 'under'
        else:
            bet = 'none'

        # Calculate if we were correct
        correct = False
        profit = 0.0

        if bet != 'none' and result != 'push':
            if bet == result:
                correct = True
                profit = 100.0  # Win $100 (assuming -110 odds)
            else:
                profit = -110.0  # Lose $110

        return {
            'result': result,
            'bet': bet,
            'correct': correct,
            'profit': profit
        }

    def _calculate_metrics(self):
        """
        Calculate aggregate performance metrics.

        Returns:
            dict with accuracy, roi, sharpe_ratio, etc.
        """
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        # Filter to only bets made (exclude 'none')
        bets = df[df['bet_recommendation'] != 'none']

        if len(bets) == 0:
            logger.warning("No bets made in backtest period")
            return {
                'total_games': len(df),
                'bets_made': 0
            }

        # Accuracy
        accuracy = bets['correct'].mean()

        # ROI
        total_risked = len(bets) * 110  # Risk $110 per bet at -110 odds
        total_profit = bets['profit'].sum()
        roi = (total_profit / total_risked) * 100

        # Win rate
        wins = (bets['correct'] == True).sum()
        losses = (bets['correct'] == False).sum()
        win_rate = wins / len(bets) if len(bets) > 0 else 0

        # Sharpe ratio (risk-adjusted returns)
        if len(bets) > 1:
            returns = bets['profit'] / 110  # Normalize by bet size
            sharpe = (returns.mean() / returns.std()) * np.sqrt(len(bets)) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Brier score (probability calibration)
        binary_outcomes = (df['actual_value'] > df['line']).astype(int)
        brier_score = np.mean((df['prob_over'] - binary_outcomes) ** 2)

        metrics = {
            'total_games': len(df),
            'bets_made': len(bets),
            'wins': wins,
            'losses': losses,
            'pushes': (df['outcome'] == 'push').sum(),
            'accuracy': accuracy,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'roi': roi,
            'sharpe_ratio': sharpe,
            'brier_score': brier_score,
            'avg_prob_over': df['prob_over'].mean(),
        }

        return metrics

    def _save_prediction_to_db(self, result):
        """Save prediction to database for future analysis."""
        try:
            prediction_data = {
                'player_id': result['player_id'],
                'game_date': result['date'].strftime('%Y-%m-%d'),
                'stat_type': self.stat_type,
                'predicted_mean': result['predicted_mean'],
                'predicted_std': result['predicted_std'],
                'prob_over': result['prob_over'],
                'line_value': result['line'],
                'expected_value': None,  # Could calculate EV here
                'book_odds': -110,  # Assuming standard odds
            }
            self.db.insert_prediction(prediction_data)
        except Exception as e:
            logger.warning(f"Could not save prediction to DB: {e}")

    def get_results_df(self):
        """Return results as DataFrame for analysis."""
        return pd.DataFrame(self.results)

    def print_summary(self, metrics):
        """Print nicely formatted summary of backtest results."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Stat Type: {self.stat_type}")
        print(f"Line: {self.line_value or 'Dynamic (rolling average)'}")
        print("-" * 60)

        # Handle case where no bets were made
        if not metrics or 'total_games' not in metrics:
            print("⚠ No predictions made in this period")
            print("Possible reasons:")
            print("  - Date range is in the future")
            print("  - No games played in this period")
            print("  - Insufficient historical data")
            print("=" * 60 + "\n")
            return

        print(f"Total Games: {metrics['total_games']}")
        print(f"Bets Made: {metrics['bets_made']}")

        # Only show detailed stats if bets were made
        if metrics['bets_made'] > 0:
            print(f"Wins: {metrics['wins']}")
            print(f"Losses: {metrics['losses']}")
            print(f"Pushes: {metrics['pushes']}")
            print("-" * 60)
            print(f"Accuracy: {metrics['accuracy']:.1%}")
            print(f"Win Rate: {metrics['win_rate']:.1%}")
            print(f"ROI: {metrics['roi']:.2f}%")
            print(f"Total Profit: ${metrics['total_profit']:.2f}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"Brier Score: {metrics['brier_score']:.4f} (lower is better)")
        else:
            print("\n⚠ No bets recommended (model had low confidence on all games)")

        print("=" * 60 + "\n")
