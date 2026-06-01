"""Per-stat distribution sweep: one realistic line per stat instead of 25.5.

The earlier sweep (``run_distribution_sweep.py``) pinned a single line
(25.5) for every stat, which made assists / rebounds trivially "win" by
being way above any reasonable threshold and made PRA trivially win by
being way below.  Either the model looks 100 % accurate or it looks 0 %
accurate depending on which side of the unrealistic line the truth sits.

This wrapper runs the same sweep but with a *per-stat* line that's drawn
from real-world ranges, so the resulting ROI / win-rate / Brier numbers
reflect calibration at lines a book would actually post.  Defaults are
chosen from common NBA market levels for star players; pass ``--lines
points=24.5 rebounds=7.5 ...`` to override.

CLI:
    python -m nba_model.evaluation.run_per_stat_sweep \
        --players "LeBron James" "Stephen Curry" ... \
        --windows 5 7 10 15 \
        --stat-types points assists rebounds pra

Writes one set of artifacts per stat alongside the original sweep, keyed
by ``<output_prefix>_<stat>_*`` so they don't clobber each other.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from nba_model.evaluation.run_distribution_sweep import DEFAULT_PLAYERS
from nba_model.evaluation.run_batch_backtest import run_batch_backtest

DEFAULT_WINDOWS = [5, 7, 10, 15]
# The original sweep tests `normal`/`poisson`; keep that as the default
# here so output is comparable.  Pass `--distributions normal poisson
# negative_binomial` to expand.
DEFAULT_DISTRIBUTIONS = ["normal", "poisson"]

logger = logging.getLogger(__name__)


# Realistic per-stat lines.  These are the kind of numbers a book would
# actually price for star NBA players; the line=25.5 default in the
# vanilla sweep made the sweep meaningless for assists / rebounds / PRA.
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


def _parse_line_overrides(items: list[str] | None) -> dict[str, float]:
    """Parse ``--lines points=24.5 rebounds=7.5`` into a dict."""
    out: dict[str, float] = {}
    for raw in items or []:
        if "=" not in raw:
            raise SystemExit(
                f"--lines argument {raw!r} must be of the form stat=value"
            )
        k, v = raw.split("=", 1)
        try:
            out[k.strip().lower()] = float(v)
        except ValueError as exc:
            raise SystemExit(
                f"--lines value for {k!r} must be a number; got {v!r}"
            ) from exc
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Distribution sweep with per-stat realistic lines (avoid the "
            "fixed-25.5 pitfall in run_distribution_sweep)."
        ),
    )
    parser.add_argument("--players", nargs="+", default=DEFAULT_PLAYERS)
    parser.add_argument("--windows", nargs="+", type=int, default=DEFAULT_WINDOWS)
    parser.add_argument(
        "--stat-types", nargs="+",
        default=["points", "assists", "rebounds", "pra"],
    )
    parser.add_argument(
        "--distributions", nargs="+", default=DEFAULT_DISTRIBUTIONS,
    )
    parser.add_argument("--start-date", default="2024-11-01")
    parser.add_argument("--end-date", default="2025-03-15")
    # ``history_games`` is the size of the per-player cache the underlying
    # ``_CachedLoader`` warms before running.  It MUST be at least ~120 so
    # the backtest has enough rows to scan inside the configured date
    # window (the cached loader doesn't re-fetch; whatever's loaded once
    # is all the backtest will ever see).  ``run_batch_backtest`` also
    # rejects values below 20 outright.
    parser.add_argument("--history-games", type=int, default=120)
    parser.add_argument("--american-odds", type=int, default=-110)
    parser.add_argument(
        "--lines", nargs="+", default=None,
        help=(
            "Override per-stat lines via stat=value pairs "
            "(e.g. points=24.5 rebounds=7.5).  Unspecified stats use the "
            "module-level defaults in DEFAULT_LINES_BY_STAT."
        ),
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--output-prefix",
        default=f"per_stat_sweep_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    overrides = _parse_line_overrides(args.lines)
    lines_by_stat = {**DEFAULT_LINES_BY_STAT, **overrides}

    summaries = []
    for stat in args.stat_types:
        line = lines_by_stat.get(stat.lower())
        if line is None:
            logger.warning(
                "No default line for stat %r — skipping. Pass via --lines.",
                stat,
            )
            continue
        logger.info("Running sweep for %s at line %s", stat, line)
        results_df, failures_df = run_batch_backtest(
            players=args.players,
            windows=args.windows,
            stat_types=[stat],
            distributions=args.distributions,
            start_date=args.start_date,
            end_date=args.end_date,
            line=line,
            use_market_lines=False,
            require_market_line=False,
            market_book=None,
            market_line_agg="median",
            history_games=args.history_games,
            confidence=args.confidence,
            american_odds=args.american_odds,
        )
        # Mirror the existing sweep's artifact layout — write one CSV per
        # stat so reviewers can diff them side-by-side.
        from pathlib import Path
        out_dir = Path("nba_model/evaluation/artifacts")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        results_path = (
            out_dir / f"{args.output_prefix}_{stat}_results_{ts}.csv"
        )
        failures_path = (
            out_dir / f"{args.output_prefix}_{stat}_failures_{ts}.csv"
        )
        results_df.to_csv(results_path, index=False)
        if not failures_df.empty:
            failures_df.to_csv(failures_path, index=False)

        # Quick on-screen summary per stat: best distribution by avg ROI.
        if not results_df.empty and "roi" in results_df.columns:
            best = (
                results_df.groupby("distribution", as_index=False)["roi"]
                .mean()
                .sort_values("roi", ascending=False)
            )
            top = best.iloc[0]
            summaries.append({
                "stat": stat,
                "line": line,
                "best_distribution": top["distribution"],
                "avg_roi": float(top["roi"]),
                "runs": int(len(results_df)),
                "results_path": str(results_path),
            })
        else:
            summaries.append({
                "stat": stat,
                "line": line,
                "best_distribution": None,
                "avg_roi": None,
                "runs": int(len(results_df)) if results_df is not None else 0,
                "results_path": str(results_path),
            })

    print("\nPer-stat sweep summary:")
    for row in summaries:
        roi = row.get("avg_roi")
        roi_str = f"{roi:.2f}" if roi is not None else "n/a"
        print(
            f"  {row['stat']:>10}  line={row['line']:>6.2f}  "
            f"best={row['best_distribution'] or '-':<10}  "
            f"avg_roi={roi_str}  runs={row['runs']}  "
            f"  ({row['results_path']})"
        )


if __name__ == "__main__":
    main()
