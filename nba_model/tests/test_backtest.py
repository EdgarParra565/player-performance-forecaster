from nba_model.evaluation.backtest import Backtester
from datetime import datetime, timedelta

# IMPORTANT: Use dates in the PAST for historical data
# NBA 2024-25 season started October 22, 2024
# Use recent games that have already been played

start_date = '2024-11-01'  # November 1, 2024
end_date = '2025-01-15'  # January 15, 2025 (or whatever today is)

backtester = Backtester(
    start_date=start_date,
    end_date=end_date,
    line_value=25.5,  # LeBron's typical line this season
    stat_type='points'
)

# Test on LeBron James
metrics = backtester.run_backtest("LeBron James", window=10)

# Print results
backtester.print_summary(metrics)

# Get detailed results
results_df = backtester.get_results_df()

if len(results_df) > 0:
    print("\nSample predictions:")
    print(results_df[['date', 'predicted_mean', 'actual_value', 'prob_over', 'outcome', 'profit']].head(10))

    # Save results to CSV for analysis
    results_df.to_csv('backtest_results.csv', index=False)
    print("\n✓ Results saved to backtest_results.csv")
else:
    print("\n⚠ No predictions made - check date range or data availability")