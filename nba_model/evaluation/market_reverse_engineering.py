"""Reverse-engineer implied volatility scales from sportsbook odds.

Given historical model predictions and book odds, this script infers, for each
row, a volatility multiplier `sigma_scale` such that:

  prob_over_distribution(line, mu, sigma * sigma_scale, distribution) ~= p_book

where `p_book` is the implied probability from the book's odds.

Aggregating these scales by (book, stat_type) gives a simple
backwards-engineering view of how aggressively each book prices volatility
relative to the model's baseline sigma.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import json

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model.odds import american_to_implied_prob
from nba_model.model.probability import prob_over_distribution

ARTIFACT_DIR = Path("nba_model/evaluation/artifacts")


def _safe_implied_prob(odds) -> Optional[float]:
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return None
    try:
        return float(american_to_implied_prob(int(odds)))
    except (TypeError, ValueError):
        return None


def _infer_sigma_scale_for_row(
    line: float,
    mu: float,
    sigma: float,
    p_book: float,
    distribution: str = "normal",
) -> Optional[float]:
    """Solve for sigma_scale so model over-probability matches book probability."""
    if sigma <= 0:
        return None
    if p_book <= 0.0 or p_book >= 1.0:
        return None

    def f(scale: float) -> float:
        try:
            p = prob_over_distribution(
                line=line,
                mu=mu,
                sigma=sigma * max(scale, 1e-6),
                distribution=distribution,
            )
        except ValueError:
            return 1.0  # arbitrary non-zero to avoid crashes
        return float(p - p_book)

    min_scale = 0.1
    max_scale = 5.0
    tol = 1e-4
    max_iter = 40

    low = float(min_scale)
    high = float(max_scale)
    f_low = f(low)
    f_high = f(high)

    # If no sign change, try to expand search range modestly.
    if f_low * f_high > 0:
        for factor in (0.05, 0.1, 0.2):
            low_candidate = max(1e-3, low * factor)
            if low_candidate != low:
                low = low_candidate
                f_low = f(low)
                if f_low * f_high <= 0:
                    break
        else:
            return None

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = f(mid)
        if abs(f_mid) < tol:
            return float(max(mid, 1e-6))
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return float(max(0.5 * (low + high), 1e-6))


def load_prediction_market_rows(
    db_path: str,
    start_date: Optional[str],
    end_date: Optional[str],
    stat_types: Optional[list[str]],
) -> pd.DataFrame:
    """Load predictions that include book odds and line values."""
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
            p.book_odds,
            pc.config_json
        FROM predictions p
        LEFT JOIN players pl ON pl.player_id = p.player_id
        LEFT JOIN prediction_configs pc ON pc.prediction_id = p.prediction_id
        WHERE p.predicted_mean IS NOT NULL
          AND p.predicted_std IS NOT NULL
          AND p.prob_over IS NOT NULL
          AND p.line_value IS NOT NULL
          AND p.book_odds IS NOT NULL
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

    df["game_date"] = pd.to_datetime(
        df["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["stat_type"] = df["stat_type"].astype(str).str.strip().str.lower()
    df["predicted_mean"] = pd.to_numeric(df["predicted_mean"], errors="coerce")
    df["predicted_std"] = pd.to_numeric(df["predicted_std"], errors="coerce")
    df["prob_over"] = pd.to_numeric(df["prob_over"], errors="coerce")
    df["line_value"] = pd.to_numeric(df["line_value"], errors="coerce")
    df["book_odds"] = pd.to_numeric(df["book_odds"], errors="coerce")
    df = df.dropna(
        subset=["predicted_mean", "predicted_std",
                "prob_over", "line_value", "book_odds"]
    ).reset_index(drop=True)
    return df


def _distribution_from_config(config_json: Optional[str]) -> str:
    if not config_json:
        return "normal"
    try:
        payload = json.loads(config_json)
        if isinstance(payload, dict):
            value = str(payload.get("distribution", "normal")).strip().lower()
            return value or "normal"
    except Exception:  # noqa: BLE001
        return "normal"


def build_sigma_scale_table(
    df: pd.DataFrame,
    min_sigma: float = 1e-3,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        mu = float(row["predicted_mean"])
        sigma = float(max(row["predicted_std"], min_sigma))
        line = float(row["line_value"])
        p_book = _safe_implied_prob(row["book_odds"])
        if p_book is None:
            continue
        distribution = _distribution_from_config(row.get("config_json"))

        scale = _infer_sigma_scale_for_row(
            line=line,
            mu=mu,
            sigma=sigma,
            p_book=p_book,
            distribution=distribution or "normal",
        )
        if scale is None:
            continue
        rows.append(
            {
                "prediction_id": row["prediction_id"],
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "game_date": row["game_date"],
                "stat_type": row["stat_type"],
                "book_odds": int(row["book_odds"]),
                "p_book": p_book,
                "distribution": distribution or "normal",
                "sigma_scale": float(scale),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def aggregate_sigma_scales(
    sigma_df: pd.DataFrame,
) -> pd.DataFrame:
    if sigma_df.empty:
        return pd.DataFrame()

    agg = (
        sigma_df.groupby(["stat_type", "distribution"], as_index=False)
        .agg(
            rows=("sigma_scale", "size"),
            sigma_scale_median=("sigma_scale", "median"),
            sigma_scale_mean=("sigma_scale", "mean"),
            sigma_scale_p25=(
                "sigma_scale", lambda s: float(np.percentile(s, 25))),
            sigma_scale_p75=(
                "sigma_scale", lambda s: float(np.percentile(s, 75))),
        )
        .sort_values(["stat_type", "distribution"])
        .reset_index(drop=True)
    )
    return agg


def run_market_reverse_engineering(
    db_path: str = "data/database/nba_data.db",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stat_types: Optional[list[str]] = None,
    output_prefix: str = "market_reverse_engineering",
) -> dict:
    df = load_prediction_market_rows(
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        stat_types=stat_types,
    )
    sigma_df = build_sigma_scale_table(df)
    agg_df = aggregate_sigma_scales(sigma_df)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    per_row_path = ARTIFACT_DIR / f"{output_prefix}_rows_{ts}.csv"
    agg_path = ARTIFACT_DIR / f"{output_prefix}_summary_{ts}.csv"

    sigma_df.to_csv(per_row_path, index=False)
    agg_df.to_csv(agg_path, index=False)

    return {
        "rows_csv": str(per_row_path),
        "summary_csv": str(agg_path),
        "rows": int(len(sigma_df)),
        "groups": int(len(agg_df)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reverse-engineer implied volatility scales from sportsbook odds."
    )
    parser.add_argument("--db-path", default="data/database/nba_data.db")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--stat-types", nargs="*", default=None)
    parser.add_argument("--output-prefix",
                        default="market_reverse_engineering")
    return parser


def main():
    args = _build_parser().parse_args()
    result = run_market_reverse_engineering(
        db_path=args.db_path,
        start_date=args.start_date,
        end_date=args.end_date,
        stat_types=args.stat_types,
        output_prefix=args.output_prefix,
    )
    print("Market reverse-engineering summary:")
    for key, value in result.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
