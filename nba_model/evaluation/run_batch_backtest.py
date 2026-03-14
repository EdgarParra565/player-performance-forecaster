import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from nba_model.data.data_loader import DataLoader
from nba_model.data.database.db_manager import DatabaseManager
from nba_model.evaluation.backtest import Backtester
from nba_model.evaluation.significance import win_rate_significance_summary
from nba_model.model.simulation import SUPPORTED_DISTRIBUTIONS, normalize_distribution_name

ARTIFACT_DIR = Path("nba_model/evaluation/artifacts")
DEFAULT_METRIC_COLUMNS = {
    "total_games": 0,
    "bets_made": 0,
    "wins": 0,
    "losses": 0,
    "pushes": 0,
    "accuracy": float("nan"),
    "win_rate": float("nan"),
    "total_profit": 0.0,
    "roi": float("nan"),
    "sharpe_ratio": float("nan"),
    "brier_score": float("nan"),
    "avg_prob_over": float("nan"),
    "market_line_games": 0,
    "fixed_line_games": 0,
    "model_line_games": 0,
    "market_spread_games": 0,
    "row_spread_games": 0,
    "default_spread_games": 0,
}
DEFAULT_PLAYERS = [
    "LeBron James",
    "Stephen Curry",
    "Nikola Jokic",
    "Luka Doncic",
    "Jayson Tatum",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch backtest across players/windows/stats.")
    parser.add_argument("--players", nargs="+", default=DEFAULT_PLAYERS, help="List of player full names")
    parser.add_argument("--windows", nargs="+", type=int, default=[5, 7, 10, 15])
    parser.add_argument("--stat-types", nargs="+", default=["points", "assists", "rebounds", "pra"])
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=["normal"],
        help=f"Single-prop distribution(s). Supported: {', '.join(SUPPORTED_DISTRIBUTIONS)}",
    )
    parser.add_argument("--start-date", default="2024-11-01")
    parser.add_argument("--end-date", default="2025-03-15")
    parser.add_argument("--line", type=float, default=None, help="Optional fixed line for all runs")
    parser.add_argument("--use-market-lines", action="store_true", help="Use betting_lines table values by game date.")
    parser.add_argument(
        "--require-market-line",
        action="store_true",
        help="Skip games that do not have a market line when --use-market-lines is enabled.",
    )
    parser.add_argument("--market-book", default=None, help="Optional sportsbook name filter for market lines.")
    parser.add_argument("--market-line-agg", choices=["median", "mean", "min", "max"], default="median")
    parser.add_argument(
        "--history-games",
        type=int,
        default=120,
        help="How many recent games to load per player into cache for batch runs.",
    )
    parser.add_argument("--output-prefix", default="batch_backtest")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--american-odds", type=int, default=-110)
    return parser


def run_batch_backtest(
    players: list[str],
    windows: list[int],
    stat_types: list[str],
    start_date: str,
    end_date: str,
    line: float = None,
    use_market_lines: bool = False,
    require_market_line: bool = False,
    market_book: str = None,
    market_line_agg: str = "median",
    history_games: int = 120,
    confidence: float = 0.95,
    american_odds: int = -110,
    distributions: list[str] = None,
):
    if history_games < 20:
        raise ValueError("history_games must be >= 20")
    requested_distributions = distributions or ["normal"]
    normalized_distributions: list[str] = []
    for dist in requested_distributions:
        canonical = normalize_distribution_name(dist)
        if canonical not in normalized_distributions:
            normalized_distributions.append(canonical)

    # Fast fail when caller requires market lines but no candidate rows exist.
    if use_market_lines and require_market_line and line is None:
        stat_placeholders = ",".join(["?"] * len(stat_types))
        query = f"""
            SELECT COUNT(*)
            FROM betting_lines
            WHERE game_date BETWEEN ? AND ?
              AND stat_type IN ({stat_placeholders})
        """
        params = [start_date, end_date, *stat_types]
        if market_book:
            query += " AND book = ?"
            params.append(market_book)
        with DatabaseManager() as db:
            matching_lines = db.conn.execute(query, params).fetchone()[0]
        if matching_lines == 0:
            raise RuntimeError(
                "No betting_lines found for requested date/stat filters. "
                "Run odds ingestion first or disable --require-market-line. "
                "Note: current odds ingestion fetches upcoming markets; historical backtest windows "
                "need historical line snapshots in betting_lines."
            )

    class _CachedLoader:
        def __init__(self, base_loader: DataLoader, requested_games: int):
            self.base_loader = base_loader
            self.requested_games = requested_games
            self.player_ids = {}
            self.cache = {}

        def warm(self, player_name: str):
            if player_name in self.cache:
                return
            player_id = self.base_loader.get_player_id(player_name)
            data = self.base_loader.load_player_data(player_name, n_games=self.requested_games)
            self.player_ids[player_name] = player_id
            self.cache[player_name] = data

        def get_player_id(self, player_name: str):
            if player_name not in self.player_ids:
                self.warm(player_name)
            return self.player_ids[player_name]

        def load_player_data(self, player_name: str, n_games: int = 200, force_refresh: bool = False):
            del force_refresh
            if player_name not in self.cache:
                self.warm(player_name)
            return self.cache[player_name].tail(n_games).copy()

    base_loader = DataLoader()
    cached_loader = _CachedLoader(base_loader=base_loader, requested_games=history_games)
    for player in players:
        cached_loader.warm(player)

    rows = []
    failures = []

    for distribution in normalized_distributions:
        for stat_type in stat_types:
            for window in windows:
                for player_name in players:
                    try:
                        backtester = Backtester(
                            start_date=start_date,
                            end_date=end_date,
                            line_value=line,
                            stat_type=stat_type,
                            distribution=distribution,
                            use_market_lines=use_market_lines,
                            require_market_line=require_market_line,
                            market_book=market_book,
                            market_line_agg=market_line_agg,
                        )
                        backtester.loader = cached_loader
                        metrics = backtester.run_backtest(player_name, window=window)
                    except Exception as exc:
                        failures.append(
                            {
                                "player_name": player_name,
                                "distribution": distribution,
                                "stat_type": stat_type,
                                "window": window,
                                "error": str(exc),
                            }
                        )
                        continue

                    if not metrics:
                        continue

                    wins = int(metrics.get("wins", 0))
                    bets = int(metrics.get("bets_made", 0))
                    sig = win_rate_significance_summary(
                        wins=wins,
                        bets=bets,
                        confidence=confidence,
                        american_odds=american_odds,
                    )

                    row = {
                        "player_name": player_name,
                        "distribution": distribution,
                        "stat_type": stat_type,
                        "window": window,
                        "line_value": line,
                        "use_market_lines": use_market_lines,
                        "market_book": market_book,
                        "market_line_agg": market_line_agg,
                        "history_games": history_games,
                        "start_date": start_date,
                        "end_date": end_date,
                    }
                    row.update(metrics)
                    row.update(sig)
                    rows.append(row)

    results_df = pd.DataFrame(rows)
    failures_df = pd.DataFrame(failures)
    for column_name, default_value in DEFAULT_METRIC_COLUMNS.items():
        if column_name not in results_df.columns:
            results_df[column_name] = default_value
    if not results_df.empty:
        preferred_sort_cols = ["distribution", "stat_type", "window", "roi", "win_rate"]
        sort_cols = [col for col in preferred_sort_cols if col in results_df.columns]
        ascending_map = {
            "distribution": True,
            "stat_type": True,
            "window": True,
            "roi": False,
            "win_rate": False,
        }
        results_df = results_df.sort_values(
            sort_cols,
            ascending=[ascending_map[col] for col in sort_cols],
            na_position="last",
        ).reset_index(drop=True)

    return results_df, failures_df


def _build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    return (
        results_df.groupby(["distribution", "stat_type", "window"], as_index=False)
        .agg(
            avg_roi=("roi", "mean"),
            avg_win_rate=("win_rate", "mean"),
            total_bets=("bets_made", "sum"),
            runs=("player_name", "count"),
            significant_runs=("significant_at_5pct", "sum"),
        )
        .sort_values(["stat_type", "avg_roi"], ascending=[True, False])
    )


def _write_artifacts(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    output_prefix: str,
    args,
):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = ARTIFACT_DIR / f"{output_prefix}_results_{ts}.csv"
    summary_path = ARTIFACT_DIR / f"{output_prefix}_summary_{ts}.csv"
    failures_path = ARTIFACT_DIR / f"{output_prefix}_failures_{ts}.csv"
    markdown_path = ARTIFACT_DIR / f"{output_prefix}_summary_{ts}.md"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    failures_df.to_csv(failures_path, index=False)

    md_lines = [
        "# Batch Backtest Summary",
        "",
        f"- Players: {', '.join(args.players)}",
        f"- Stat types: {', '.join(args.stat_types)}",
        f"- Distributions: {', '.join(args.distributions)}",
        f"- Windows: {', '.join(str(w) for w in args.windows)}",
        f"- Date range: {args.start_date} to {args.end_date}",
        f"- Use market lines: {args.use_market_lines}",
        f"- Require market line: {args.require_market_line}",
        f"- Market book: {args.market_book or 'all books'}",
        f"- Market line aggregation: {args.market_line_agg}",
        f"- History games loaded per player: {args.history_games}",
        f"- Confidence level: {args.confidence}",
        f"- Breakeven odds assumption: {args.american_odds}",
        "",
        "## Summary by Stat/Window",
        summary_df.to_string(index=False) if not summary_df.empty else "No successful runs.",
        "",
        "## Top Rows",
        results_df.head(20).to_string(index=False) if not results_df.empty else "No successful runs.",
        "",
        "## Failures",
        failures_df.to_string(index=False) if not failures_df.empty else "None",
    ]
    markdown_path.write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "results_csv": results_path,
        "summary_csv": summary_path,
        "failures_csv": failures_path,
        "summary_md": markdown_path,
    }


def main():
    args = _build_parser().parse_args()
    results_df, failures_df = run_batch_backtest(
        players=args.players,
        windows=args.windows,
        stat_types=[s.lower() for s in args.stat_types],
        distributions=[normalize_distribution_name(d) for d in args.distributions],
        start_date=args.start_date,
        end_date=args.end_date,
        line=args.line,
        use_market_lines=args.use_market_lines,
        require_market_line=args.require_market_line,
        market_book=args.market_book,
        market_line_agg=args.market_line_agg,
        history_games=args.history_games,
        confidence=args.confidence,
        american_odds=args.american_odds,
    )
    summary_df = _build_summary(results_df)
    paths = _write_artifacts(results_df, summary_df, failures_df, args.output_prefix, args)

    print("\nBatch backtest complete.")
    for key, value in paths.items():
        print(f"{key}: {value}")
    if not summary_df.empty:
        print("\nSummary preview:")
        print(summary_df.to_string(index=False))
    if not failures_df.empty:
        print(f"\nFailures: {len(failures_df)} (see {paths['failures_csv']})")


if __name__ == "__main__":
    main()
