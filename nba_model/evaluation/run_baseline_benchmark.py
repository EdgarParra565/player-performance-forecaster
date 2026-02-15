import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from nba_model.evaluation.backtest import Backtester


ARTIFACT_DIR = Path("nba_model/evaluation/artifacts")

PLAYER_PROFILES = {
    "LeBron James": {"line": 25.5, "minutes": 34.5, "ppm": 0.79},
    "Stephen Curry": {"line": 27.5, "minutes": 33.2, "ppm": 0.83},
    "Nikola Jokic": {"line": 26.5, "minutes": 35.0, "ppm": 0.78},
}


def _stable_seed(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def _build_synthetic_games(player_name: str, profile: dict, n_games: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(_stable_seed(player_name))
    dates = pd.date_range("2024-10-01", periods=n_games, freq="D")

    minutes = np.clip(rng.normal(profile["minutes"], 2.5, n_games), 24.0, 42.0)
    ppm = np.clip(rng.normal(profile["ppm"], 0.08, n_games), 0.45, 1.15)
    points = np.clip(ppm * minutes + rng.normal(0.0, 3.2, n_games), 0.0, None)
    assists = np.clip(rng.normal(7.0, 2.0, n_games), 0.0, None)
    rebounds = np.clip(rng.normal(7.5, 2.3, n_games), 0.0, None)

    return pd.DataFrame(
        {
            "game_date": dates,
            "points": points,
            "assists": assists,
            "rebounds": rebounds,
            "minutes": minutes,
        }
    )


class _SyntheticLoader:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames
        self.player_ids = {name: idx + 1000 for idx, name in enumerate(frames.keys())}

    def get_player_id(self, player_name: str) -> int:
        return self.player_ids[player_name]

    def load_player_data(self, player_name: str, n_games: int = 200):
        return self.frames[player_name].tail(n_games).copy()


class _NoOpDatabase:
    @staticmethod
    def insert_prediction(prediction_data):
        del prediction_data


def run_baseline_benchmark(
    windows: tuple[int, ...] = (7, 10),
    start_date: str = "2024-12-01",
    end_date: str = "2025-03-15",
):
    frames = {
        name: _build_synthetic_games(name, profile)
        for name, profile in PLAYER_PROFILES.items()
    }
    loader = _SyntheticLoader(frames)

    rows = []
    for window in windows:
        for player_name, profile in PLAYER_PROFILES.items():
            backtester = Backtester(
                start_date=start_date,
                end_date=end_date,
                line_value=profile["line"],
                stat_type="points",
            )
            backtester.loader = loader
            backtester.db = _NoOpDatabase()

            metrics = backtester.run_backtest(player_name, window=window)
            rows.append(
                {
                    "player_name": player_name,
                    "window": window,
                    "line_value": profile["line"],
                    "total_games": metrics.get("total_games", 0),
                    "bets_made": metrics.get("bets_made", 0),
                    "wins": metrics.get("wins", 0),
                    "losses": metrics.get("losses", 0),
                    "win_rate": metrics.get("win_rate", 0.0),
                    "roi": metrics.get("roi", 0.0),
                    "total_profit": metrics.get("total_profit", 0.0),
                    "brier_score": metrics.get("brier_score", 0.0),
                }
            )

    results_df = pd.DataFrame(rows).sort_values(["window", "player_name"]).reset_index(drop=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = ARTIFACT_DIR / "baseline_benchmark.csv"
    results_df.to_csv(csv_path, index=False)

    summary_df = (
        results_df.groupby("window", as_index=False)
        .agg(
            avg_roi=("roi", "mean"),
            avg_win_rate=("win_rate", "mean"),
            total_bets=("bets_made", "sum"),
            total_games=("total_games", "sum"),
        )
        .sort_values("avg_roi", ascending=False)
    )
    best_window = int(summary_df.iloc[0]["window"])

    summary_lines = [
        "# Baseline Benchmark Summary",
        "",
        "This benchmark uses deterministic synthetic player logs to validate the end-to-end backtest pipeline offline.",
        f"- Players: {', '.join(PLAYER_PROFILES.keys())}",
        f"- Windows tested: {', '.join(str(v) for v in windows)}",
        f"- Period: {start_date} to {end_date}",
        f"- Best window by avg ROI: {best_window}",
        "",
        "## Window Averages",
        summary_df.to_string(index=False),
        "",
        "## Per-Player Results",
        results_df.to_string(index=False),
        "",
    ]

    summary_path = ARTIFACT_DIR / "baseline_benchmark_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return results_df, summary_df, csv_path, summary_path


if __name__ == "__main__":
    results, summary, csv_file, summary_file = run_baseline_benchmark()
    print(f"Saved benchmark CSV: {csv_file}")
    print(f"Saved benchmark summary: {summary_file}")
    print("\nWindow summary:")
    print(summary.to_string(index=False))
