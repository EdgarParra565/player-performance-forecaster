"""Run real-data multi-player benchmark and export player/window confidence intervals."""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from nba_model.evaluation.run_batch_backtest import run_batch_backtest
from nba_model.evaluation.significance import win_rate_significance_summary

ARTIFACT_DIR = Path("nba_model/evaluation/artifacts")
DEFAULT_PLAYERS = [
    "LeBron James",
    "Stephen Curry",
    "Nikola Jokic",
    "Luka Doncic",
    "Jayson Tatum",
    "Giannis Antetokounmpo",
    "Shai Gilgeous-Alexander",
    "Kevin Durant",
    "Anthony Edwards",
    "Damian Lillard",
    "Devin Booker",
    "Jalen Brunson",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-data benchmark (10+ players) with CI exports.")
    parser.add_argument("--players", nargs="+", default=DEFAULT_PLAYERS)
    parser.add_argument("--windows", nargs="+", type=int, default=[5, 7, 10, 15])
    parser.add_argument("--stat-types", nargs="+", default=["points"])
    parser.add_argument("--distributions", nargs="+", default=["normal"])
    parser.add_argument("--start-date", default="2024-11-01")
    parser.add_argument("--end-date", default="2025-03-15")
    parser.add_argument("--line", type=float, default=None)
    parser.add_argument("--use-market-lines", action="store_true")
    parser.add_argument("--require-market-line", action="store_true")
    parser.add_argument("--market-book", default=None)
    parser.add_argument("--market-line-agg", choices=["median", "mean", "min", "max"], default="median")
    parser.add_argument("--history-games", type=int, default=120)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--american-odds", type=int, default=-110)
    parser.add_argument("--output-prefix", default="real_data_benchmark")
    return parser


def build_player_window_ci_summary(
    results_df: pd.DataFrame,
    confidence: float = 0.95,
    american_odds: int = -110,
) -> pd.DataFrame:
    """Aggregate benchmark runs to player/window confidence intervals."""
    if results_df.empty:
        return pd.DataFrame()

    grouped = (
        results_df.groupby(["player_name", "window"], as_index=False)
        .agg(
            runs=("stat_type", "count"),
            total_games=("total_games", "sum"),
            bets_made=("bets_made", "sum"),
            wins=("wins", "sum"),
            losses=("losses", "sum"),
            avg_roi=("roi", "mean"),
            avg_brier=("brier_score", "mean"),
            avg_win_rate=("win_rate", "mean"),
        )
        .sort_values(["player_name", "window"])
        .reset_index(drop=True)
    )

    ci_rows = []
    for _, row in grouped.iterrows():
        sig = win_rate_significance_summary(
            wins=int(row["wins"]),
            bets=int(row["bets_made"]),
            confidence=confidence,
            american_odds=american_odds,
        )
        ci_rows.append(
            {
                "player_name": row["player_name"],
                "window": int(row["window"]),
                "win_rate_ci_lower": sig["win_rate_ci_lower"],
                "win_rate_ci_upper": sig["win_rate_ci_upper"],
                "breakeven_prob": sig["breakeven_prob"],
                "z_score_vs_breakeven": sig["z_score_vs_breakeven"],
                "p_value_vs_breakeven": sig["p_value_vs_breakeven"],
                "significant_at_5pct": sig["significant_at_5pct"],
            }
        )

    ci_df = pd.DataFrame(ci_rows)
    return grouped.merge(ci_df, on=["player_name", "window"], how="left")


def _write_artifacts(
    results_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    player_window_df: pd.DataFrame,
    output_prefix: str,
    args,
) -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = ARTIFACT_DIR / f"{output_prefix}_results_{ts}.csv"
    failures_path = ARTIFACT_DIR / f"{output_prefix}_failures_{ts}.csv"
    player_window_path = ARTIFACT_DIR / f"{output_prefix}_player_window_ci_{ts}.csv"
    markdown_path = ARTIFACT_DIR / f"{output_prefix}_summary_{ts}.md"

    results_df.to_csv(results_path, index=False)
    failures_df.to_csv(failures_path, index=False)
    player_window_df.to_csv(player_window_path, index=False)

    md_lines = [
        "# Real-Data Benchmark Summary",
        "",
        f"- Players ({len(args.players)}): {', '.join(args.players)}",
        f"- Windows: {', '.join(str(w) for w in args.windows)}",
        f"- Stat types: {', '.join(args.stat_types)}",
        f"- Distributions: {', '.join(args.distributions)}",
        f"- Date range: {args.start_date} to {args.end_date}",
        f"- Confidence: {args.confidence}",
        "",
        "## Player/Window CI Summary",
        player_window_df.to_string(index=False) if not player_window_df.empty else "No successful runs.",
        "",
        "## Top Benchmark Rows",
        results_df.head(20).to_string(index=False) if not results_df.empty else "No successful runs.",
        "",
        "## Failures",
        failures_df.to_string(index=False) if not failures_df.empty else "None",
    ]
    markdown_path.write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "results_csv": str(results_path),
        "player_window_ci_csv": str(player_window_path),
        "failures_csv": str(failures_path),
        "summary_md": str(markdown_path),
    }


def main():
    args = _build_parser().parse_args()
    results_df, failures_df = run_batch_backtest(
        players=args.players,
        windows=args.windows,
        stat_types=[s.lower() for s in args.stat_types],
        distributions=[d.lower() for d in args.distributions],
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

    player_window_df = build_player_window_ci_summary(
        results_df=results_df,
        confidence=args.confidence,
        american_odds=args.american_odds,
    )
    paths = _write_artifacts(
        results_df=results_df,
        failures_df=failures_df,
        player_window_df=player_window_df,
        output_prefix=args.output_prefix,
        args=args,
    )

    print("\nReal-data benchmark complete.")
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
