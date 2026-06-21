"""Run distribution sweep benchmark and export ROI/calibration/significance summaries."""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from nba_model.evaluation.run_batch_backtest import run_batch_backtest
from nba_model.evaluation.significance import win_rate_significance_summary
from nba_model.model.simulation import SUPPORTED_DISTRIBUTIONS

ARTIFACT_DIR = Path("nba_model/evaluation/artifacts")

# Realistic per-stat lines a book would actually post. Used as the default
# when the caller supplies neither --use-market-lines nor an explicit --line,
# so symmetric distributions aren't settled at line == model mean (which makes
# P(over) ≈ 0.5, the 0.55 edge filter rejects every wager, and the sweep
# returns zero bets — the degenerate case the per-stat sweep was built to fix).
# Kept here as the canonical source; run_per_stat_sweep imports it.
DEFAULT_LINES_BY_STAT: dict[str, float] = {
    "points":   24.5,
    "assists":   6.5,
    "rebounds":  7.5,
    "pra":      35.5,
    "ra":       11.5,
    "three_pointers_made": 2.5,
    "field_goals_made":    9.5,
    "minutes": 33.5,
}

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
    parser = argparse.ArgumentParser(description="Run multi-player distribution sweep benchmark.")
    parser.add_argument("--players", nargs="+", default=DEFAULT_PLAYERS)
    parser.add_argument("--windows", nargs="+", type=int, default=[5, 7, 10, 15])
    parser.add_argument("--stat-types", nargs="+", default=["points", "assists", "rebounds", "pra"])
    parser.add_argument("--distributions", nargs="+", default=list(SUPPORTED_DISTRIBUTIONS))
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
    parser.add_argument("--output-prefix", default="distribution_sweep")
    return parser


def _resolve_stat_lines(
    stat_types: list[str],
    line: "float | None",
    use_market_lines: bool,
) -> dict:
    """Resolve the per-stat line to settle on.

    - ``use_market_lines``: every stat → ``line`` (may be ``None``, meaning the
      backtest pulls the book's market line).
    - explicit ``--line``: every stat → that fixed line.
    - neither: each stat → its realistic ``DEFAULT_LINES_BY_STAT`` value so we
      never silently fall back to ``line == model mean``. Stats without a
      default map to ``None`` (caller warns/skips).
    """
    out: dict = {}
    for stat in stat_types:
        s = stat.lower()
        if use_market_lines:
            out[s] = line
        elif line is not None:
            out[s] = float(line)
        else:
            out[s] = DEFAULT_LINES_BY_STAT.get(s)
    return out


def build_distribution_summary(
    results_df: pd.DataFrame,
    confidence: float = 0.95,
    american_odds: int = -110,
) -> pd.DataFrame:
    """Aggregate sweep metrics by distribution/stat_type."""
    if results_df.empty:
        return pd.DataFrame()

    grouped = (
        results_df.groupby(["distribution", "stat_type"], as_index=False)
        .agg(
            runs=("player_name", "count"),
            total_games=("total_games", "sum"),
            total_bets=("bets_made", "sum"),
            wins=("wins", "sum"),
            losses=("losses", "sum"),
            avg_roi=("roi", "mean"),
            median_roi=("roi", "median"),
            avg_win_rate=("win_rate", "mean"),
            avg_brier_score=("brier_score", "mean"),
            significant_runs=("significant_at_5pct", "sum"),
        )
        .sort_values(["stat_type", "avg_roi"], ascending=[True, False])
        .reset_index(drop=True)
    )

    grouped["significant_rate"] = grouped.apply(
        lambda row: (float(row["significant_runs"]) / float(row["runs"])) if row["runs"] else 0.0,
        axis=1,
    )

    ci_rows = []
    for _, row in grouped.iterrows():
        sig = win_rate_significance_summary(
            wins=int(row["wins"]),
            bets=int(row["total_bets"]),
            confidence=confidence,
            american_odds=american_odds,
        )
        ci_rows.append(
            {
                "distribution": row["distribution"],
                "stat_type": row["stat_type"],
                "win_rate_ci_lower": sig["win_rate_ci_lower"],
                "win_rate_ci_upper": sig["win_rate_ci_upper"],
                "z_score_vs_breakeven": sig["z_score_vs_breakeven"],
                "p_value_vs_breakeven": sig["p_value_vs_breakeven"],
                "aggregate_significant": sig["significant_at_5pct"],
            }
        )
    ci_df = pd.DataFrame(ci_rows)
    return grouped.merge(ci_df, on=["distribution", "stat_type"], how="left")


def _write_artifacts(
    results_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    distribution_df: pd.DataFrame,
    output_prefix: str,
    args,
) -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = ARTIFACT_DIR / f"{output_prefix}_results_{ts}.csv"
    failures_path = ARTIFACT_DIR / f"{output_prefix}_failures_{ts}.csv"
    summary_path = ARTIFACT_DIR / f"{output_prefix}_distribution_summary_{ts}.csv"
    markdown_path = ARTIFACT_DIR / f"{output_prefix}_summary_{ts}.md"

    results_df.to_csv(results_path, index=False)
    failures_df.to_csv(failures_path, index=False)
    distribution_df.to_csv(summary_path, index=False)

    md_lines = [
        "# Distribution Sweep Summary",
        "",
        f"- Players ({len(args.players)}): {', '.join(args.players)}",
        f"- Windows: {', '.join(str(w) for w in args.windows)}",
        f"- Stat types: {', '.join(args.stat_types)}",
        f"- Distributions: {', '.join(args.distributions)}",
        f"- Date range: {args.start_date} to {args.end_date}",
        f"- Confidence: {args.confidence}",
        "",
        "## ROI / Calibration / Significance by Distribution",
        distribution_df.to_string(index=False) if not distribution_df.empty else "No successful runs.",
        "",
        "## Top Sweep Rows",
        results_df.head(25).to_string(index=False) if not results_df.empty else "No successful runs.",
        "",
        "## Failures",
        failures_df.to_string(index=False) if not failures_df.empty else "None",
    ]
    markdown_path.write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "results_csv": str(results_path),
        "distribution_summary_csv": str(summary_path),
        "failures_csv": str(failures_path),
        "summary_md": str(markdown_path),
    }


def main():
    args = _build_parser().parse_args()
    stat_types = [s.lower() for s in args.stat_types]
    stat_lines = _resolve_stat_lines(
        stat_types, args.line, args.use_market_lines,
    )
    if not args.use_market_lines and args.line is None:
        print(
            "No --use-market-lines / --line given: settling each stat at its "
            "realistic default line (avoids the line==mean degeneracy). "
            f"Lines: {stat_lines}"
        )

    # Run each stat at its own line so symmetric distributions aren't forced
    # to settle at the model mean. Concatenate into one results/failures frame.
    results_frames: list[pd.DataFrame] = []
    failures_frames: list[pd.DataFrame] = []
    for stat in stat_types:
        line = stat_lines.get(stat)
        if line is None and not args.use_market_lines:
            print(f"  skipping {stat!r}: no default line (pass --line).")
            continue
        res, fail = run_batch_backtest(
            players=args.players,
            windows=args.windows,
            stat_types=[stat],
            distributions=[d.lower() for d in args.distributions],
            start_date=args.start_date,
            end_date=args.end_date,
            line=line,
            use_market_lines=args.use_market_lines,
            require_market_line=args.require_market_line,
            market_book=args.market_book,
            market_line_agg=args.market_line_agg,
            history_games=args.history_games,
            confidence=args.confidence,
            american_odds=args.american_odds,
        )
        if res is not None and not res.empty:
            results_frames.append(res)
        if fail is not None and not fail.empty:
            failures_frames.append(fail)

    results_df = (
        pd.concat(results_frames, ignore_index=True)
        if results_frames else pd.DataFrame()
    )
    failures_df = (
        pd.concat(failures_frames, ignore_index=True)
        if failures_frames else pd.DataFrame()
    )
    distribution_df = build_distribution_summary(
        results_df=results_df,
        confidence=args.confidence,
        american_odds=args.american_odds,
    )
    paths = _write_artifacts(
        results_df=results_df,
        failures_df=failures_df,
        distribution_df=distribution_df,
        output_prefix=args.output_prefix,
        args=args,
    )

    print("\nDistribution sweep complete.")
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
