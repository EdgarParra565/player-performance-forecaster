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

Optional flags:
- `--window` (rolling window)
- `--line` (prop line)
- `--odds` (American odds)
- `--opp-def-rating`
- `--spread`
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

## Baseline Benchmark Results

These results come from deterministic synthetic game logs to validate the full backtest pipeline offline:

| Window | Avg ROI | Avg Win Rate | Total Bets | Total Games |
|---|---:|---:|---:|---:|
| 10 | 14.27% | 59.85% | 252 | 315 |
| 7 | 10.68% | 57.97% | 256 | 315 |

Per-player detail is in `nba_model/evaluation/artifacts/baseline_benchmark.csv`.

## Known Limitations

- Benchmark metrics are synthetic and validate pipeline behavior, not live betting edge.
- Live NBA API calls require network access when cache/database does not already contain player games.
- Defense and minutes adjustments are available in model modules, but can still be expanded in backtest logic.
- Odds ingestion is currently a placeholder integration and not yet productionized.

## Next Expansion Targets

- Add real multi-player historical benchmarks from cached/live NBA data.
- Integrate richer context features (rest, home/away, opponent splits, injuries).
- Expand from points-only baseline to assists/rebounds/PRA evaluation.
