# Baseline Benchmark Summary

This benchmark uses deterministic synthetic player logs to validate the end-to-end backtest pipeline offline.
- Players: LeBron James, Stephen Curry, Nikola Jokic
- Stat types: points, assists, rebounds, pra
- Windows tested: 7, 10
- Period: 2024-12-01 to 2025-03-15
- Best stat/window by avg ROI: {'stat_type': 'rebounds', 'window': 10, 'avg_roi': 44.42234848484849}

## Window Averages
stat_type  window   avg_roi  avg_win_rate  total_bets  total_games
  assists      10 32.052233      0.691702         281          315
  assists       7 26.783022      0.664102         279          315
   points      10  0.969962      0.528890         230          315
   points       7  0.097438      0.524320         243          315
      pra      10 36.559910      0.715314         299          315
      pra       7 35.396824      0.709221         293          315
 rebounds      10 44.422348      0.756498         274          315
 rebounds       7 43.374283      0.751008         284          315

## Per-Player Results
  player_name stat_type  window  line_value  total_games  bets_made  wins  losses  win_rate        roi  total_profit  brier_score
 LeBron James   assists       7         7.5          105         88    50      38  0.568182   8.471074         820.0     0.258929
 Nikola Jokic   assists       7         8.5          105         96    68      28  0.708333  35.227273        3720.0     0.209232
Stephen Curry   assists       7         5.5          105         95    68      27  0.715789  36.650718        3830.0     0.217508
 LeBron James   assists      10         7.5          105         85    54      31  0.635294  21.283422        1990.0     0.243598
 Nikola Jokic   assists      10         8.5          105        101    71      30  0.702970  34.203420        3800.0     0.211578
Stephen Curry   assists      10         5.5          105         95    70      25  0.736842  40.669856        4250.0     0.219822
 LeBron James    points       7        25.5          105         72    47      25  0.652778  24.621212        1950.0     0.239948
 Nikola Jokic    points       7        26.5          105         78    29      49  0.371795 -29.020979       -2490.0     0.294985
Stephen Curry    points       7        27.5          105         93    51      42  0.548387   4.692082         480.0     0.290648
 LeBron James    points      10        25.5          105         68    45      23  0.661765  26.336898        1970.0     0.232133
 Nikola Jokic    points      10        26.5          105         68    26      42  0.382353 -27.005348       -2020.0     0.283991
Stephen Curry    points      10        27.5          105         94    51      43  0.542553   3.578337         370.0     0.280046
 LeBron James       pra       7        40.5          105         93    63      30  0.677419  29.325513        3000.0     0.242549
 Nikola Jokic       pra       7        46.5          105         99    73      26  0.737374  40.771350        4440.0     0.206054
Stephen Curry       pra       7        37.5          105        101    72      29  0.712871  36.093609        4010.0     0.214878
 LeBron James       pra      10        40.5          105         97    66      31  0.680412  29.896907        3190.0     0.235692
 Nikola Jokic       pra      10        46.5          105         99    73      26  0.737374  40.771350        4440.0     0.199762
Stephen Curry       pra      10        37.5          105        103    75      28  0.728155  39.011474        4420.0     0.207839
 LeBron James  rebounds       7         7.5          105         74    30      44  0.405405 -22.604423       -1840.0     0.288448
 Nikola Jokic  rebounds       7        11.5          105        105   100       5  0.952381  81.818182        9450.0     0.047160
Stephen Curry  rebounds       7         4.5          105        105    94      11  0.895238  70.909091        8190.0     0.103150
 LeBron James  rebounds      10         7.5          105         64    27      37  0.421875 -19.460227       -1370.0     0.266084
 Nikola Jokic  rebounds      10        11.5          105        105   100       5  0.952381  81.818182        9450.0     0.045549
Stephen Curry  rebounds      10         4.5          105        105    94      11  0.895238  70.909091        8190.0     0.100067
