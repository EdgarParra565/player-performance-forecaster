# Baseline Benchmark Summary

This benchmark uses deterministic synthetic player logs to validate the end-to-end backtest pipeline offline.
- Players: LeBron James, Stephen Curry, Nikola Jokic
- Windows tested: 7, 10
- Period: 2024-12-01 to 2025-03-15
- Best window by avg ROI: 10

## Window Averages
 window   avg_roi  avg_win_rate  total_bets  total_games
     10 14.267677      0.598545         252          315
      7 10.676345      0.579733         256          315

## Per-Player Results
  player_name  window  line_value  total_games  bets_made  wins  losses  win_rate       roi  total_profit  brier_score
 LeBron James       7        25.5          105         95    65      30  0.684211 30.622010        3200.0     0.220628
 Nikola Jokic       7        26.5          105         83    45      38  0.542169  3.504929         320.0     0.278212
Stephen Curry       7        27.5          105         78    40      38  0.512821 -2.097902        -180.0     0.278379
 LeBron James      10        25.5          105         96    68      28  0.708333 35.227273        3720.0     0.214593
 Nikola Jokic      10        26.5          105         84    47      37  0.559524  6.818182         630.0     0.263367
Stephen Curry      10        27.5          105         72    38      34  0.527778  0.757576          60.0     0.266143
