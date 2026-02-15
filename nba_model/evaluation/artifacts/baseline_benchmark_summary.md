# Baseline Benchmark Summary

This benchmark uses deterministic synthetic player logs to validate the end-to-end backtest pipeline offline.
- Players: LeBron James, Stephen Curry, Nikola Jokic
- Windows tested: 7, 10
- Period: 2024-12-01 to 2025-03-15
- Best window by avg ROI: 10

## Window Averages
 window   avg_roi  avg_win_rate  total_bets  total_games
     10  1.099578      0.529569         233          315
      7 -0.260349      0.522446         255          315

## Per-Player Results
  player_name  window  line_value  total_games  bets_made  wins  losses  win_rate        roi  total_profit  brier_score
 LeBron James       7        25.5          105         78    50      28  0.641026  22.377622        1920.0     0.236606
 Nikola Jokic       7        26.5          105         81    32      49  0.395062 -24.579125       -2190.0     0.296868
Stephen Curry       7        27.5          105         96    51      45  0.531250   1.420455         150.0     0.293746
 LeBron James      10        25.5          105         74    51      23  0.689189  31.572482        2570.0     0.228668
 Nikola Jokic      10        26.5          105         70    26      44  0.371429 -29.090909       -2240.0     0.285626
Stephen Curry      10        27.5          105         89    47      42  0.528090   0.817160          80.0     0.281897
