from nba_model.evaluation.backtest import Backtester
from datetime import datetime, timedelta

# Run backtest on recent games
end_date = datetime.now()
start_date = end_date - timedelta(days=60)  # Last 2 months

backtester = Backtester(
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d'),
    line_value=27.5,  # Test against a fixed line
    stat_type='points'
)

# Test on LeBron James
metrics = backtester.run_backtest("LeBron James", window=10)

# Print results
backtester.print_summary(metrics)

# Get detailed results
results_df = backtester.get_results_df()
print("\nSample predictions:")
print(results_df[['date', 'predicted_mean', 'actual_value', 'prob_over', 'outcome', 'profit']].head(10))

# Save results to CSV for analysis
results_df.to_csv('backtest_results.csv', index=False)
print("\n✓ Results saved to backtest_results.csv")