# NBA Player Performance Forecaster

Probabilistic NBA player-props modeling project focused on points forecasts, EV estimation, and backtesting.

The current baseline is designed to be reproducible offline (synthetic benchmark harness) while still supporting real NBA API data through caching when available.

## Current Status

- Core data contract is unified around lowercase stat columns: `points`, `assists`, `rebounds`, `minutes`.
- Backtest pipeline is wired end-to-end without import/signature mismatches.
- `run_model` supports executable demo modes for single props and correlated parlays.
- Deterministic smoke tests are in place for features, probability math, and backtest integration.
- Baseline benchmark artifacts are generated and saved under `nba_model/evaluation/artifacts/`.

## Repository Layout

- `nba_model/data/` - API loading, caching, and database utilities
- `nba_model/model/` - feature engineering, probability, simulation, EV, parlay modeling
- `nba_model/evaluation/` - backtesting and benchmark runners
- `nba_model/tests/` - smoke and integration-style tests
- `nba_model/visualization/` - distribution plotting

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run Commands

### 1) Single Prop or Parlay Demo

```bash
python3 -m nba_model.run_model --mode single --player "LeBron James" --n-games 100
python3 -m nba_model.run_model --mode parlay --player "LeBron James" --n-games 100
python3 -m nba_model.run_model --mode both --player "LeBron James" --n-games 100
```

### 1b) Simple Desktop UI

Run from a lightweight local UI (Single Prop + Parlay with PrizePicks/Underdog inputs):

```bash
python3 -m nba_model.simple_ui
```

In the Parlay tab you can input:
- sportsbook (`prizepicks`, `underdog`, or `custom`)
- leg stats and lines
- parlay odds and odds format (`american`, `decimal`, or `multiplier`)
- bounded sliders for correlation/volatility severity, plus reset-to-default

In the Single Prop tab you can also tune:
- opponent/league defense inputs and defense sensitivity
- blowout threshold + blowout penalty
- bounded severity sliders (defense, minutes penalty, and sigma)
- selectable distribution family (`normal`, `student_t`, `binomial`, `poisson`, `exponential`, `uniform`, `lognormal`, `power_law`)
- reset-to-default button for one-click baseline restore

In the `Manual Lines Import` tab you can paste sportsbook rows and write them into
`betting_lines` for market-line backtests. Supported row formats (pipe/csv/tab):
- `player | stat | line | over_odds | under_odds`
- `player | game_date | book | stat | line | over_odds | under_odds`
- raw board text paste is also supported (auto-extracts player + matchup + line + stat blocks)
After parsing, use the table preview to verify rows and delete selected entries before saving.

Optional flags:
- `--window` (rolling window)
- `--line` (prop line)
- `--odds` (American odds)
- `--opp-def-rating`
- `--spread`
- `--league-avg-def-rating`
- `--defense-sensitivity`
- `--blowout-threshold`
- `--blowout-penalty`
- `--distribution`
- `--defense-severity`
- `--minutes-penalty-severity`
- `--sigma-severity`
- `--correlation-severity`
- `--volatility-severity`
- `--plot` (single mode only)

### 2) Deterministic Smoke Tests (No Live API Required)

```bash
python3 -m unittest nba_model.tests.test_smoke
```

### 3) Baseline Benchmark (Deterministic, Offline)

```bash
python3 -m nba_model.evaluation.run_baseline_benchmark
```

Outputs:
- `nba_model/evaluation/artifacts/baseline_benchmark.csv`
- `nba_model/evaluation/artifacts/baseline_benchmark_summary.md`

### 4) Batch Backtest (Multi-Player / Multi-Window)

Run real-data batch backtests with significance stats (Wilson CI + z-test vs breakeven):

```bash
python3 -m nba_model.evaluation.run_batch_backtest \
  --players "LeBron James" "Stephen Curry" "Nikola Jokic" \
  --windows 5 7 10 15 \
  --stat-types points \
  --distributions normal student_t lognormal \
  --start-date 2024-11-01 \
  --end-date 2025-03-15
```

Notes:
- For points backtests, minutes projection uses spread from game rows when available, then falls back to spread aliases in `betting_lines` (`spread`, `game_spread`, `line_spread`, `vegas_spread`, etc.).

Artifacts are saved under `nba_model/evaluation/artifacts/` with timestamped filenames.

### 5) Daily ETL Runner (Game Logs + Defense + Odds)

Run a single daily pipeline job with retry + JSON report output:

```bash
python3 -m nba_model.data.daily_etl \
  --players "LeBron James" "Stephen Curry" \
  --season 2024-25 \
  --retries 2 \
  --retry-delay-seconds 1 \
  --report-dir nba_model/data/artifacts
```

Notes:
- By default, it force-refreshes game logs, updates `team_defense`, and attempts odds ingestion.
- Odds step auto-skips when no API key is set (`ODDS_API_KEY`/`THE_ODDS_API_KEY`).
- Use `--strict` for cron jobs to return non-zero exit code on failed/partial runs.

### 6) Real-Data Multi-Player Benchmark (Player/Window CIs)

```bash
python3 -m nba_model.evaluation.run_real_data_benchmark \
  --players "LeBron James" "Stephen Curry" "Nikola Jokic" "Luka Doncic" \
           "Jayson Tatum" "Giannis Antetokounmpo" "Shai Gilgeous-Alexander" \
           "Kevin Durant" "Anthony Edwards" "Damian Lillard" \
  --windows 5 7 10 15 \
  --stat-types points \
  --distributions normal \
  --start-date 2024-11-01 \
  --end-date 2025-03-15
```

Exports include per-player/window confidence interval artifacts.

### 7) Distribution Sweep Benchmark (ROI + Calibration + Significance)

```bash
python3 -m nba_model.evaluation.run_distribution_sweep \
  --windows 5 7 10 15 \
  --stat-types points assists rebounds pra \
  --distributions normal student_t binomial poisson exponential uniform lognormal power_law \
  --start-date 2024-11-01 \
  --end-date 2025-03-15
```

Exports include distribution/stat summaries for ROI, Brier score, and significance.

### 8) Line Comparison + Monthly Diagnostics

```bash
# Cross-book and model-vs-book value comparison
python3 -m nba_model.evaluation.line_comparison \
  --start-date 2024-11-01 \
  --end-date 2025-03-15 \
  --stat-types points assists rebounds pra \
  --edge-threshold 0.02

# Monthly drift, drawdown, and calibration diagnostics
python3 -m nba_model.evaluation.monthly_diagnostics \
  --start-date 2024-11-01 \
  --end-date 2025-03-15 \
  --stat-types points assists rebounds pra
```

## Production defaults (from benchmarks)

Default distribution per stat type is defined in `nba_model/model/simulation.py` as `DEFAULT_DISTRIBUTION_BY_STAT`. These should be set after running the real-data benchmark and distribution sweep and reviewing artifacts:

1. **Real-data benchmark** (player/window CIs):
   ```bash
   python3 -m nba_model.evaluation.run_real_data_benchmark \
     --windows 5 7 10 15 \
     --stat-types points \
     --distributions normal \
     --start-date 2024-11-01 \
     --end-date 2025-03-15
   ```

2. **Distribution sweep** (ROI/calibration/significance by distribution and stat):
   ```bash
   python3 -m nba_model.evaluation.run_distribution_sweep \
     --windows 5 7 10 15 \
     --stat-types points assists rebounds pra \
     --start-date 2024-11-01 \
     --end-date 2025-03-15
   ```

3. Review artifacts in `nba_model/evaluation/artifacts/` (e.g. `distribution_sweep_distribution_summary_*.csv`, `distribution_sweep_summary_*.md`).
4. Pick the best distribution per stat (e.g. by `avg_roi` or significance) and update `DEFAULT_DISTRIBUTION_BY_STAT` in `nba_model/model/simulation.py`. Use `get_default_distribution(stat_type)` in code when a single default is needed for a given stat.

## Baseline Benchmark Results

Baseline benchmark now runs with the same template across:
- `points`
- `assists`
- `rebounds`
- `pra` (points + rebounds + assists)

Latest results are generated into:
- `nba_model/evaluation/artifacts/baseline_benchmark.csv`
- `nba_model/evaluation/artifacts/baseline_benchmark_summary.md`

## Known Limitations

- Benchmark metrics are synthetic and validate pipeline behavior, not live betting edge.
- Live NBA API calls require network access when cache/database does not already contain player games.
- Cross-book/model-vs-book and monthly diagnostics depend on having populated `betting_lines` and `predictions` data.
- Historical open/close line snapshots are not yet stored, so line-move analysis is still limited.

## Next Expansion Targets

- Add real multi-player historical benchmarks from cached/live NBA data.
- Expand contextual modeling beyond current rest/travel/injury-proxy heuristics.
- Extend line comparison to include richer execution constraints and portfolio-level filters.
