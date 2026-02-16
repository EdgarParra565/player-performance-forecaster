import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from nba_model.evaluation.backtest import Backtester
from nba_model.evaluation.significance import win_rate_significance_summary

ARTIFACT_DIR = Path("nba_model/evaluation/artifacts")
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
    parser.add_argument("--start-date", default="2024-11-01")
    parser.add_argument("--end-date", default="2025-03-15")
    parser.add_argument("--line", type=float, default=None, help="Optional fixed line for all runs")
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
    confidence: float = 0.95,
    american_odds: int = -110,
):
    rows = []
    failures = []

    for stat_type in stat_types:
        for window in windows:
            for player_name in players:
                try:
                    backtester = Backtester(
                        start_date=start_date,
                        end_date=end_date,
                        line_value=line,
                        stat_type=stat_type,
                    )
                    metrics = backtester.run_backtest(player_name, window=window)
                except Exception as exc:
                    failures.append(
                        {
                            "player_name": player_name,
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
                    "stat_type": stat_type,
                    "window": window,
                    "line_value": line,
                    "start_date": start_date,
                    "end_date": end_date,
                }
                row.update(metrics)
                row.update(sig)
                rows.append(row)

    results_df = pd.DataFrame(rows)
    failures_df = pd.DataFrame(failures)
    if not results_df.empty:
        results_df = results_df.sort_values(
            ["stat_type", "window", "roi", "win_rate"],
            ascending=[True, True, False, False],
        ).reset_index(drop=True)

    return results_df, failures_df


def _build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    return (
        results_df.groupby(["stat_type", "window"], as_index=False)
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
        f"- Windows: {', '.join(str(w) for w in args.windows)}",
        f"- Date range: {args.start_date} to {args.end_date}",
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
        start_date=args.start_date,
        end_date=args.end_date,
        line=args.line,
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
