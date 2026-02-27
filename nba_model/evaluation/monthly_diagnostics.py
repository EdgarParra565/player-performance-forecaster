"""Monthly performance diagnostics: drift, drawdown, and calibration trends."""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from nba_model.data.database.db_manager import DatabaseManager

ARTIFACT_DIR = Path("nba_model/evaluation/artifacts")


def load_prediction_actuals(
    db_path: str = "data/database/nba_data.db",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stat_types: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load predictions joined to realized outcomes from game logs."""
    query = """
        SELECT
            p.prediction_id,
            p.player_id,
            COALESCE(pl.name, '') AS player_name,
            p.game_date,
            lower(p.stat_type) AS stat_type,
            p.predicted_mean,
            p.predicted_std,
            p.prob_over,
            p.line_value,
            p.created_at,
            CASE
                WHEN lower(p.stat_type) = 'points' THEN gl.points
                WHEN lower(p.stat_type) = 'assists' THEN gl.assists
                WHEN lower(p.stat_type) = 'rebounds' THEN gl.rebounds
                WHEN lower(p.stat_type) = 'pra' THEN (gl.points + gl.assists + gl.rebounds)
                ELSE NULL
            END AS actual_value
        FROM predictions p
        LEFT JOIN players pl ON pl.player_id = p.player_id
        LEFT JOIN game_logs gl
            ON gl.player_id = p.player_id
           AND gl.game_date = p.game_date
        WHERE p.prob_over IS NOT NULL
          AND p.line_value IS NOT NULL
          AND p.predicted_mean IS NOT NULL
    """
    params: list = []
    if start_date:
        query += " AND p.game_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND p.game_date <= ?"
        params.append(end_date)
    if stat_types:
        normalized = [str(s).strip().lower() for s in stat_types]
        placeholders = ",".join(["?"] * len(normalized))
        query += f" AND lower(p.stat_type) IN ({placeholders})"
        params.extend(normalized)

    with DatabaseManager(db_path=db_path) as db:
        df = pd.read_sql_query(query, db.conn, params=params)

    if df.empty:
        return df

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values(["created_at", "prediction_id"]).drop_duplicates(
        subset=["player_id", "game_date", "stat_type"],
        keep="last",
    )
    return df.reset_index(drop=True)


def _bet_side(prob_over: float, edge_threshold: float = 0.55) -> str:
    if prob_over > edge_threshold:
        return "over"
    if prob_over < (1.0 - edge_threshold):
        return "under"
    return "none"


def _actual_side(actual_value: float, line_value: float) -> str:
    if actual_value > line_value:
        return "over"
    if actual_value < line_value:
        return "under"
    return "push"


def _profit_for_row(bet_side: str, actual_side: str) -> float:
    if bet_side == "none" or actual_side == "push":
        return 0.0
    if bet_side == actual_side:
        return 100.0
    return -110.0


def build_monthly_diagnostics(
    prediction_df: pd.DataFrame,
    edge_threshold: float = 0.55,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compute monthly diagnostics and equity drawdown trends."""
    if prediction_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    df = prediction_df.copy()
    df = df.dropna(subset=["game_date", "line_value", "predicted_mean", "prob_over", "actual_value"])
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    df["line_value"] = pd.to_numeric(df["line_value"], errors="coerce")
    df["predicted_mean"] = pd.to_numeric(df["predicted_mean"], errors="coerce")
    df["prob_over"] = pd.to_numeric(df["prob_over"], errors="coerce")
    df["actual_value"] = pd.to_numeric(df["actual_value"], errors="coerce")
    df = df.dropna(subset=["line_value", "predicted_mean", "prob_over", "actual_value"]).copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    df["actual_over"] = (df["actual_value"] > df["line_value"]).astype(int)
    df["actual_side"] = df.apply(lambda row: _actual_side(row["actual_value"], row["line_value"]), axis=1)
    df["bet_recommendation"] = df["prob_over"].apply(lambda p: _bet_side(float(p), edge_threshold=edge_threshold))
    df["profit"] = df.apply(
        lambda row: _profit_for_row(row["bet_recommendation"], row["actual_side"]),
        axis=1,
    )
    df["brier_score"] = (df["prob_over"] - df["actual_over"]) ** 2
    df["prediction_error"] = df["actual_value"] - df["predicted_mean"]
    df["abs_error"] = df["prediction_error"].abs()
    df["month"] = df["game_date"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby("month", as_index=False)
        .agg(
            predictions=("prediction_id", "count"),
            bets_made=("bet_recommendation", lambda s: int((s != "none").sum())),
            wins=("profit", lambda s: int((s == 100.0).sum())),
            losses=("profit", lambda s: int((s == -110.0).sum())),
            pushes=("actual_side", lambda s: int((s == "push").sum())),
            total_profit=("profit", "sum"),
            avg_prob_over=("prob_over", "mean"),
            actual_over_rate=("actual_over", "mean"),
            avg_brier_score=("brier_score", "mean"),
            mean_prediction_error=("prediction_error", "mean"),
            mean_abs_error=("abs_error", "mean"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )

    monthly["win_rate"] = monthly.apply(
        lambda row: (float(row["wins"]) / float(row["bets_made"])) if row["bets_made"] else 0.0,
        axis=1,
    )
    monthly["roi"] = monthly.apply(
        lambda row: (float(row["total_profit"]) / (float(row["bets_made"]) * 110.0) * 100.0)
        if row["bets_made"]
        else 0.0,
        axis=1,
    )
    monthly["calibration_gap"] = (monthly["avg_prob_over"] - monthly["actual_over_rate"]).abs()

    bets = df[df["bet_recommendation"] != "none"].copy().sort_values("game_date")
    if bets.empty:
        equity = pd.DataFrame(
            columns=["game_date", "month", "profit", "cumulative_profit", "running_peak", "drawdown"]
        )
        monthly["month_min_drawdown"] = 0.0
        max_drawdown = 0.0
    else:
        bets["cumulative_profit"] = bets["profit"].cumsum()
        bets["running_peak"] = bets["cumulative_profit"].cummax()
        bets["drawdown"] = bets["cumulative_profit"] - bets["running_peak"]
        equity = bets[["game_date", "month", "profit", "cumulative_profit", "running_peak", "drawdown"]].copy()
        monthly_drawdown = (
            equity.groupby("month", as_index=False)
            .agg(month_min_drawdown=("drawdown", "min"))
            .sort_values("month")
        )
        monthly = monthly.merge(monthly_drawdown, on="month", how="left")
        monthly["month_min_drawdown"] = monthly["month_min_drawdown"].fillna(0.0)
        max_drawdown = float(equity["drawdown"].min())

    overall = {
        "predictions": int(len(df)),
        "bets_made": int((df["bet_recommendation"] != "none").sum()),
        "total_profit": float(df["profit"].sum()),
        "overall_brier_score": float(df["brier_score"].mean()),
        "overall_calibration_gap": float(abs(df["prob_over"].mean() - df["actual_over"].mean())),
        "overall_mean_prediction_error": float(df["prediction_error"].mean()),
        "overall_mean_abs_error": float(df["abs_error"].mean()),
        "max_drawdown": max_drawdown,
        "start_date": str(df["game_date"].min().date()),
        "end_date": str(df["game_date"].max().date()),
    }
    return monthly, equity, overall


def run_monthly_diagnostics(
    db_path: str = "data/database/nba_data.db",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stat_types: Optional[list[str]] = None,
    edge_threshold: float = 0.55,
    output_prefix: str = "monthly_diagnostics",
) -> dict:
    """Load prediction history, compute diagnostics, and write artifacts."""
    prediction_df = load_prediction_actuals(
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        stat_types=stat_types,
    )
    monthly_df, equity_df, overall = build_monthly_diagnostics(
        prediction_df=prediction_df,
        edge_threshold=edge_threshold,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    monthly_path = ARTIFACT_DIR / f"{output_prefix}_monthly_{ts}.csv"
    equity_path = ARTIFACT_DIR / f"{output_prefix}_equity_{ts}.csv"
    summary_path = ARTIFACT_DIR / f"{output_prefix}_summary_{ts}.md"

    monthly_df.to_csv(monthly_path, index=False)
    equity_df.to_csv(equity_path, index=False)

    summary_lines = [
        "# Monthly Diagnostics Summary",
        "",
        f"- Date range: {start_date or 'all'} to {end_date or 'all'}",
        f"- Stat types: {', '.join(stat_types) if stat_types else 'all'}",
        f"- Edge threshold: {edge_threshold}",
        "",
        "## Overall",
    ]
    if overall:
        for key, value in overall.items():
            summary_lines.append(f"- {key}: {value}")
    else:
        summary_lines.append("No diagnostic rows available.")
    summary_lines.extend(
        [
            "",
            "## Monthly Trend Table",
            monthly_df.to_string(index=False) if not monthly_df.empty else "No rows.",
        ]
    )
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "monthly_csv": str(monthly_path),
        "equity_csv": str(equity_path),
        "summary_md": str(summary_path),
        "monthly_rows": int(len(monthly_df)),
        "equity_rows": int(len(equity_df)),
        "overall": overall,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run monthly drift/drawdown/calibration diagnostics.")
    parser.add_argument("--db-path", default="data/database/nba_data.db")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--stat-types", nargs="*", default=None)
    parser.add_argument("--edge-threshold", type=float, default=0.55)
    parser.add_argument("--output-prefix", default="monthly_diagnostics")
    return parser


def main():
    args = _build_parser().parse_args()
    outputs = run_monthly_diagnostics(
        db_path=args.db_path,
        start_date=args.start_date,
        end_date=args.end_date,
        stat_types=args.stat_types,
        edge_threshold=args.edge_threshold,
        output_prefix=args.output_prefix,
    )

    print("\nMonthly diagnostics complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
