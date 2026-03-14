"""Reverse-engineer sportsbook pricing parameters from market rows.

This module joins model predictions with market quotes from:
  - betting_lines
  - betting_line_snapshots

For each quote row it infers a volatility multiplier (`sigma_scale`) such that:
  prob_over_distribution(line, mu, sigma * sigma_scale, distribution) ~= p_book

It then exports:
  1) per-row inference table
  2) per-book/per-stat summary
  3) per-book/per-stat/per-player summary (keeps door open for player-specific models)
  4) optional per-book/per-stat/per-market_key summary
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import time
from typing import Optional

import numpy as np
import pandas as pd

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model.odds import american_to_implied_prob
from nba_model.model.probability import prob_over_distribution

ARTIFACT_DIR = Path("nba_model/evaluation/artifacts")
STAT_TYPE_ALIASES = {
    "pts": "points",
    "point": "points",
    "points": "points",
    "ast": "assists",
    "assist": "assists",
    "assists": "assists",
    "reb": "rebounds",
    "rebound": "rebounds",
    "rebounds": "rebounds",
    "pra": "pra",
    "points_rebounds_assists": "pra",
}


def _normalize_stat_type(value) -> str:
    key = str(value or "").strip().lower()
    return STAT_TYPE_ALIASES.get(key, key)


def _safe_implied_prob(odds) -> Optional[float]:
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return None
    try:
        return float(american_to_implied_prob(int(odds)))
    except (TypeError, ValueError):
        return None


def _distribution_from_config(config_json: Optional[str]) -> str:
    if not config_json:
        return "normal"
    try:
        payload = json.loads(config_json)
        if isinstance(payload, dict):
            value = str(payload.get("distribution", "normal")).strip().lower()
            return value or "normal"
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return "normal"


def load_latest_predictions(
    db_path: str,
    start_date: Optional[str],
    end_date: Optional[str],
    stat_types: Optional[list[str]],
) -> pd.DataFrame:
    """Load latest prediction snapshot per player/date/stat with distribution metadata."""
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
            p.created_at,
            pc.config_json,
            pc.created_at AS config_created_at
        FROM predictions p
        LEFT JOIN players pl ON pl.player_id = p.player_id
        LEFT JOIN prediction_configs pc ON pc.prediction_id = p.prediction_id
        WHERE p.predicted_mean IS NOT NULL
          AND p.predicted_std IS NOT NULL
    """
    params: list = []
    if start_date:
        query += " AND p.game_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND p.game_date <= ?"
        params.append(end_date)
    if stat_types:
        normalized = [_normalize_stat_type(s) for s in stat_types]
        placeholders = ",".join(["?"] * len(normalized))
        query += f" AND lower(p.stat_type) IN ({placeholders})"
        params.extend(normalized)

    with DatabaseManager(db_path=db_path) as db:
        pred_df = pd.read_sql_query(query, db.conn, params=params)

    if pred_df.empty:
        return pred_df

    pred_df["game_date"] = pd.to_datetime(pred_df["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    pred_df["created_at"] = pd.to_datetime(pred_df["created_at"], errors="coerce")
    pred_df["config_created_at"] = pd.to_datetime(pred_df["config_created_at"], errors="coerce")
    pred_df["stat_type"] = pred_df["stat_type"].map(_normalize_stat_type)
    pred_df["predicted_mean"] = pd.to_numeric(pred_df["predicted_mean"], errors="coerce")
    pred_df["predicted_std"] = pd.to_numeric(pred_df["predicted_std"], errors="coerce")
    pred_df = pred_df.dropna(subset=["game_date", "predicted_mean", "predicted_std"])
    if pred_df.empty:
        return pred_df

    # One config row per prediction.
    pred_df = pred_df.sort_values(
        ["prediction_id", "config_created_at", "created_at"],
        ascending=[True, True, True],
    ).drop_duplicates(subset=["prediction_id"], keep="last")

    # Latest prediction snapshot per player/date/stat.
    pred_df = pred_df.sort_values(["created_at", "prediction_id"]).drop_duplicates(
        subset=["player_id", "game_date", "stat_type"],
        keep="last",
    )
    pred_df["distribution"] = pred_df["config_json"].apply(_distribution_from_config)
    return pred_df.reset_index(drop=True)


def load_betting_lines_quotes(
    db_path: str,
    start_date: Optional[str],
    end_date: Optional[str],
    stat_types: Optional[list[str]],
    books: Optional[list[str]],
) -> pd.DataFrame:
    """Load quotes from betting_lines."""
    query = """
        SELECT
            bl.line_id AS quote_id,
            bl.player_id,
            COALESCE(p.name, '') AS player_name,
            bl.game_date,
            lower(bl.stat_type) AS stat_type,
            bl.book,
            bl.line_value,
            bl.over_odds,
            bl.under_odds,
            bl.scraped_at AS quote_ts_utc,
            '' AS market_key,
            'betting_lines' AS source_table
        FROM betting_lines bl
        LEFT JOIN players p ON p.player_id = bl.player_id
        WHERE bl.line_value IS NOT NULL
    """
    params: list = []
    if start_date:
        query += " AND bl.game_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND bl.game_date <= ?"
        params.append(end_date)
    if stat_types:
        normalized = [_normalize_stat_type(s) for s in stat_types]
        placeholders = ",".join(["?"] * len(normalized))
        query += f" AND lower(bl.stat_type) IN ({placeholders})"
        params.extend(normalized)
    if books:
        normalized_books = [str(b).strip() for b in books if str(b).strip()]
        if normalized_books:
            placeholders = ",".join(["?"] * len(normalized_books))
            query += f" AND bl.book IN ({placeholders})"
            params.extend(normalized_books)

    with DatabaseManager(db_path=db_path) as db:
        df = pd.read_sql_query(query, db.conn, params=params)
    return _normalize_quote_df(df)


def load_snapshot_quotes(
    db_path: str,
    start_date: Optional[str],
    end_date: Optional[str],
    stat_types: Optional[list[str]],
    books: Optional[list[str]],
) -> pd.DataFrame:
    """Load quotes from betting_line_snapshots."""
    query = """
        SELECT
            s.snapshot_id AS quote_id,
            s.player_id,
            COALESCE(p.name, '') AS player_name,
            s.game_date,
            lower(s.stat_type) AS stat_type,
            s.book,
            s.line_value,
            s.over_odds,
            s.under_odds,
            s.snapshot_ts_utc AS quote_ts_utc,
            COALESCE(s.market_key, '') AS market_key,
            'betting_line_snapshots' AS source_table
        FROM betting_line_snapshots s
        LEFT JOIN players p ON p.player_id = s.player_id
        WHERE s.line_value IS NOT NULL
    """
    params: list = []
    if start_date:
        query += " AND s.game_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND s.game_date <= ?"
        params.append(end_date)
    if stat_types:
        normalized = [_normalize_stat_type(s) for s in stat_types]
        placeholders = ",".join(["?"] * len(normalized))
        query += f" AND lower(s.stat_type) IN ({placeholders})"
        params.extend(normalized)
    if books:
        normalized_books = [str(b).strip() for b in books if str(b).strip()]
        if normalized_books:
            placeholders = ",".join(["?"] * len(normalized_books))
            query += f" AND s.book IN ({placeholders})"
            params.extend(normalized_books)

    with DatabaseManager(db_path=db_path) as db:
        df = pd.read_sql_query(query, db.conn, params=params)
    return _normalize_quote_df(df)


def _normalize_quote_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize quote dataframe schema."""
    if df.empty:
        return df
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["quote_ts_utc"] = pd.to_datetime(out["quote_ts_utc"], errors="coerce", utc=True)
    out["book"] = out["book"].astype(str).str.strip()
    out["market_key"] = out["market_key"].astype(str).str.strip().str.lower()
    out["stat_type"] = out["stat_type"].map(_normalize_stat_type)
    out["line_value"] = pd.to_numeric(out["line_value"], errors="coerce")
    out["over_odds"] = pd.to_numeric(out["over_odds"], errors="coerce")
    out["under_odds"] = pd.to_numeric(out["under_odds"], errors="coerce")
    out = out.dropna(subset=["player_id", "game_date", "book", "stat_type", "line_value"]).copy()
    if out.empty:
        return out

    # De-dupe repeated lines while preserving time-series shape for snapshots.
    lines = out[out["source_table"] == "betting_lines"].copy()
    snapshots = out[out["source_table"] == "betting_line_snapshots"].copy()

    if not lines.empty:
        lines = lines.sort_values("quote_ts_utc").drop_duplicates(
            subset=["player_id", "game_date", "stat_type", "book", "line_value", "over_odds", "under_odds"],
            keep="last",
        )
    if not snapshots.empty:
        snapshots = snapshots.sort_values("quote_ts_utc").drop_duplicates(
            subset=[
                "player_id",
                "game_date",
                "stat_type",
                "book",
                "market_key",
                "line_value",
                "over_odds",
                "under_odds",
                "quote_ts_utc",
            ],
            keep="last",
        )

    out = pd.concat([lines, snapshots], ignore_index=True)
    return out.sort_values(["source_table", "quote_ts_utc", "game_date"]).reset_index(drop=True)


def load_market_quotes(
    db_path: str,
    source: str,
    start_date: Optional[str],
    end_date: Optional[str],
    stat_types: Optional[list[str]],
    books: Optional[list[str]],
) -> pd.DataFrame:
    """Load market quotes from requested source(s)."""
    source_key = str(source or "both").strip().lower()
    if source_key not in {"both", "lines", "snapshots"}:
        raise ValueError("source must be one of: both, lines, snapshots")

    frames = []
    if source_key in {"both", "lines"}:
        frames.append(load_betting_lines_quotes(db_path, start_date, end_date, stat_types, books))
    if source_key in {"both", "snapshots"}:
        frames.append(load_snapshot_quotes(db_path, start_date, end_date, stat_types, books))

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["source_table", "quote_ts_utc", "game_date"]).reset_index(drop=True)


def build_reverse_engineering_base_table(
    quotes_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join quotes to latest model predictions by player/date/stat."""
    if quotes_df.empty or predictions_df.empty:
        return pd.DataFrame()

    join_cols = ["player_id", "game_date", "stat_type"]
    merged = quotes_df.merge(
        predictions_df[
            [
                "prediction_id",
                "player_id",
                "player_name",
                "game_date",
                "stat_type",
                "predicted_mean",
                "predicted_std",
                "distribution",
            ]
        ],
        on=join_cols,
        how="inner",
        suffixes=("", "_pred"),
    )
    if merged.empty:
        return merged

    # Keep quote-side player name when available, fallback to prediction-side.
    merged["player_name"] = np.where(
        merged["player_name"].astype(str).str.strip() != "",
        merged["player_name"],
        merged["player_name_pred"],
    )
    merged["predicted_mean"] = pd.to_numeric(merged["predicted_mean"], errors="coerce")
    merged["predicted_std"] = pd.to_numeric(merged["predicted_std"], errors="coerce")
    merged = merged.dropna(subset=["predicted_mean", "predicted_std", "line_value"]).copy()
    return merged.reset_index(drop=True)


def _infer_sigma_scale_for_prob(
    line: float,
    mu: float,
    sigma: float,
    p_over_target: float,
    distribution: str,
) -> Optional[float]:
    """Solve for sigma scale so model over-probability matches target probability."""
    if sigma <= 0:
        return None
    if p_over_target <= 0.0 or p_over_target >= 1.0:
        return None

    def f(scale: float) -> float:
        try:
            p_model = prob_over_distribution(
                line=float(line),
                mu=float(mu),
                sigma=float(max(scale, 1e-6) * sigma),
                distribution=str(distribution or "normal"),
            )
        except ValueError:
            p_model = prob_over_distribution(
                line=float(line),
                mu=float(mu),
                sigma=float(max(scale, 1e-6) * sigma),
                distribution="normal",
            )
        return float(p_model - p_over_target)

    lo = 0.05
    hi = 6.0
    f_lo = f(lo)
    f_hi = f(hi)
    if f_lo * f_hi > 0:
        return None

    for _ in range(48):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < 1e-4:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return float(0.5 * (lo + hi))


def _median_from_candidates(values: list[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return None
    return float(np.median(clean))


def build_inferred_parameter_rows(
    base_df: pd.DataFrame,
    min_sigma: float = 1e-3,
) -> pd.DataFrame:
    """Infer per-row pricing parameters from quote + model rows."""
    if base_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in base_df.iterrows():
        mu = float(row["predicted_mean"])
        sigma = float(max(row["predicted_std"], min_sigma))
        line = float(row["line_value"])
        distribution = str(row.get("distribution") or "normal").strip().lower()

        p_over_raw = _safe_implied_prob(row.get("over_odds"))
        p_under_raw = _safe_implied_prob(row.get("under_odds"))
        has_two_sided = p_over_raw is not None and p_under_raw is not None
        overround = None
        p_over_no_vig = None
        p_under_no_vig = None
        if has_two_sided:
            denom = p_over_raw + p_under_raw
            if denom > 0:
                p_over_no_vig = float(p_over_raw / denom)
                p_under_no_vig = float(p_under_raw / denom)
                overround = float(denom - 1.0)

        # Convert under-side probabilities to over-side targets.
        p_over_from_under_raw = (1.0 - p_under_raw) if p_under_raw is not None else None
        p_over_from_under_no_vig = (1.0 - p_under_no_vig) if p_under_no_vig is not None else None

        scale_over_raw = (
            _infer_sigma_scale_for_prob(line, mu, sigma, p_over_raw, distribution)
            if p_over_raw is not None
            else None
        )
        scale_under_raw = (
            _infer_sigma_scale_for_prob(line, mu, sigma, p_over_from_under_raw, distribution)
            if p_over_from_under_raw is not None
            else None
        )
        scale_over_no_vig = (
            _infer_sigma_scale_for_prob(line, mu, sigma, p_over_no_vig, distribution)
            if p_over_no_vig is not None
            else None
        )
        scale_under_no_vig = (
            _infer_sigma_scale_for_prob(line, mu, sigma, p_over_from_under_no_vig, distribution)
            if p_over_from_under_no_vig is not None
            else None
        )

        consensus_scale = _median_from_candidates(
            [scale_over_no_vig, scale_under_no_vig, scale_over_raw, scale_under_raw]
        )
        over_under_gap = None
        if scale_over_no_vig is not None and scale_under_no_vig is not None:
            over_under_gap = float(abs(scale_over_no_vig - scale_under_no_vig))

        rows.append(
            {
                "quote_id": row.get("quote_id"),
                "source_table": row.get("source_table"),
                "quote_ts_utc": row.get("quote_ts_utc"),
                "prediction_id": row.get("prediction_id"),
                "player_id": row.get("player_id"),
                "player_name": row.get("player_name"),
                "game_date": row.get("game_date"),
                "book": row.get("book"),
                "market_key": row.get("market_key"),
                "stat_type": row.get("stat_type"),
                "distribution": distribution,
                "line_value": line,
                "over_odds": row.get("over_odds"),
                "under_odds": row.get("under_odds"),
                "predicted_mean": mu,
                "predicted_std": sigma,
                "line_minus_mu": float(line - mu),
                "line_minus_mu_sigma": float((line - mu) / sigma) if sigma > 0 else None,
                "p_over_raw": p_over_raw,
                "p_under_raw": p_under_raw,
                "p_over_no_vig": p_over_no_vig,
                "p_under_no_vig": p_under_no_vig,
                "vig_overround": overround,
                "has_two_sided_odds": int(bool(has_two_sided)),
                "sigma_scale_over_raw": scale_over_raw,
                "sigma_scale_under_raw": scale_under_raw,
                "sigma_scale_over_no_vig": scale_over_no_vig,
                "sigma_scale_under_no_vig": scale_under_no_vig,
                "sigma_scale_consensus": consensus_scale,
                "sigma_scale_over_under_gap": over_under_gap,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.dropna(subset=["sigma_scale_consensus"]).reset_index(drop=True)


def _p25(series: pd.Series) -> float:
    return float(np.percentile(series, 25))


def _p75(series: pd.Series) -> float:
    return float(np.percentile(series, 75))


def _aggregate_parameter_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = (
        df.groupby(group_cols, as_index=False)
        .agg(
            rows=("sigma_scale_consensus", "size"),
            unique_players=("player_id", "nunique"),
            unique_games=("game_date", "nunique"),
            sigma_scale_mean=("sigma_scale_consensus", "mean"),
            sigma_scale_median=("sigma_scale_consensus", "median"),
            sigma_scale_p25=("sigma_scale_consensus", _p25),
            sigma_scale_p75=("sigma_scale_consensus", _p75),
            line_minus_mu_sigma_mean=("line_minus_mu_sigma", "mean"),
            line_minus_mu_sigma_median=("line_minus_mu_sigma", "median"),
            vig_overround_mean=("vig_overround", "mean"),
            vig_overround_median=("vig_overround", "median"),
            over_under_scale_gap_mean=("sigma_scale_over_under_gap", "mean"),
            two_sided_quote_rate=("has_two_sided_odds", "mean"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    return out


def aggregate_inferred_parameters(
    inferred_df: pd.DataFrame,
    min_player_segment_rows: int = 8,
    include_market_segments: bool = True,
    min_market_segment_rows: int = 15,
) -> dict[str, pd.DataFrame]:
    """Build per-book summaries with optional player/stat and market-key segmentation."""
    if inferred_df.empty:
        return {
            "book_stat": pd.DataFrame(),
            "book_stat_player": pd.DataFrame(),
            "book_stat_market": pd.DataFrame(),
        }

    book_stat = _aggregate_parameter_table(
        inferred_df,
        ["source_table", "book", "stat_type", "distribution"],
    )

    book_stat_player = _aggregate_parameter_table(
        inferred_df,
        ["source_table", "book", "stat_type", "player_id", "player_name", "distribution"],
    )
    book_stat_player = book_stat_player[
        book_stat_player["rows"] >= max(1, int(min_player_segment_rows))
    ].reset_index(drop=True)

    if include_market_segments:
        tmp = inferred_df.copy()
        tmp["market_key"] = tmp["market_key"].fillna("")
        tmp = tmp[tmp["market_key"] != ""]
        book_stat_market = _aggregate_parameter_table(
            tmp,
            ["source_table", "book", "stat_type", "market_key", "distribution"],
        )
        book_stat_market = book_stat_market[
            book_stat_market["rows"] >= max(1, int(min_market_segment_rows))
        ].reset_index(drop=True)
    else:
        book_stat_market = pd.DataFrame()

    return {
        "book_stat": book_stat,
        "book_stat_player": book_stat_player,
        "book_stat_market": book_stat_market,
    }


def _run_market_reverse_engineering_once(
    db_path: str = "data/database/nba_data.db",
    source: str = "both",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stat_types: Optional[list[str]] = None,
    books: Optional[list[str]] = None,
    min_player_segment_rows: int = 8,
    include_market_segments: bool = True,
    min_market_segment_rows: int = 15,
    output_prefix: str = "market_reverse_engineering",
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Run one reverse-engineering pass and return metrics + summary tables."""
    quotes_df = load_market_quotes(
        db_path=db_path,
        source=source,
        start_date=start_date,
        end_date=end_date,
        stat_types=stat_types,
        books=books,
    )
    predictions_df = load_latest_predictions(
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        stat_types=stat_types,
    )
    base_df = build_reverse_engineering_base_table(quotes_df, predictions_df)
    inferred_df = build_inferred_parameter_rows(base_df)
    summaries = aggregate_inferred_parameters(
        inferred_df=inferred_df,
        min_player_segment_rows=min_player_segment_rows,
        include_market_segments=include_market_segments,
        min_market_segment_rows=min_market_segment_rows,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows_path = ARTIFACT_DIR / f"{output_prefix}_rows_{ts}.csv"
    book_stat_path = ARTIFACT_DIR / f"{output_prefix}_book_stat_{ts}.csv"
    player_path = ARTIFACT_DIR / f"{output_prefix}_book_stat_player_{ts}.csv"
    market_path = ARTIFACT_DIR / f"{output_prefix}_book_stat_market_{ts}.csv"

    inferred_df.to_csv(rows_path, index=False)
    summaries["book_stat"].to_csv(book_stat_path, index=False)
    summaries["book_stat_player"].to_csv(player_path, index=False)
    summaries["book_stat_market"].to_csv(market_path, index=False)

    result = {
        "rows_csv": str(rows_path),
        "book_stat_summary_csv": str(book_stat_path),
        "book_stat_player_summary_csv": str(player_path),
        "book_stat_market_summary_csv": str(market_path),
        "quote_rows_loaded": int(len(quotes_df)),
        "prediction_rows_loaded": int(len(predictions_df)),
        "joined_rows": int(len(base_df)),
        "inferred_rows": int(len(inferred_df)),
        "book_stat_groups": int(len(summaries["book_stat"])),
        "book_stat_player_groups": int(len(summaries["book_stat_player"])),
        "book_stat_market_groups": int(len(summaries["book_stat_market"])),
    }
    return result, summaries


def run_market_reverse_engineering(
    db_path: str = "data/database/nba_data.db",
    source: str = "both",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stat_types: Optional[list[str]] = None,
    books: Optional[list[str]] = None,
    min_player_segment_rows: int = 8,
    include_market_segments: bool = True,
    min_market_segment_rows: int = 15,
    output_prefix: str = "market_reverse_engineering",
) -> dict:
    """Run reverse-engineering pipeline directly from betting_lines/snapshots."""
    result, _ = _run_market_reverse_engineering_once(
        db_path=db_path,
        source=source,
        start_date=start_date,
        end_date=end_date,
        stat_types=stat_types,
        books=books,
        min_player_segment_rows=min_player_segment_rows,
        include_market_segments=include_market_segments,
        min_market_segment_rows=min_market_segment_rows,
        output_prefix=output_prefix,
    )
    return result


def _coverage_ready(
    result: dict,
    min_inferred_rows: int,
    min_book_stat_groups: int,
    min_player_segment_groups: int,
) -> bool:
    return (
        int(result.get("inferred_rows", 0)) >= max(0, int(min_inferred_rows))
        and int(result.get("book_stat_groups", 0)) >= max(0, int(min_book_stat_groups))
        and int(result.get("book_stat_player_groups", 0))
        >= max(0, int(min_player_segment_groups))
    )


def _build_book_stat_signature(
    book_stat_df: pd.DataFrame,
    min_group_rows_for_stability: int = 1,
) -> dict[tuple, float]:
    """Map group identity -> sigma median for convergence checks."""
    if book_stat_df.empty:
        return {}

    required_cols = [
        "source_table",
        "book",
        "stat_type",
        "distribution",
        "rows",
        "sigma_scale_median",
    ]
    if not all(col in book_stat_df.columns for col in required_cols):
        return {}

    filtered = book_stat_df[
        book_stat_df["rows"] >= max(1, int(min_group_rows_for_stability))
    ].copy()
    if filtered.empty:
        return {}

    signature: dict[tuple, float] = {}
    for _, row in filtered.iterrows():
        value = pd.to_numeric(row.get("sigma_scale_median"), errors="coerce")
        if pd.isna(value):
            continue
        key = (
            row.get("source_table"),
            row.get("book"),
            row.get("stat_type"),
            row.get("distribution"),
        )
        signature[key] = float(value)
    return signature


def _max_relative_signature_shift(
    previous_signature: dict[tuple, float],
    current_signature: dict[tuple, float],
) -> Optional[float]:
    """Compute largest relative sigma-median shift across overlapping groups."""
    if not previous_signature or not current_signature:
        return None
    shared = set(previous_signature.keys()).intersection(current_signature.keys())
    if not shared:
        return None

    shifts = []
    for key in shared:
        prev_value = previous_signature[key]
        curr_value = current_signature[key]
        denom = max(abs(prev_value), 1e-6)
        shifts.append(abs(curr_value - prev_value) / denom)
    if not shifts:
        return None
    return float(max(shifts))


def run_market_reverse_engineering_continuous(
    db_path: str = "data/database/nba_data.db",
    source: str = "both",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stat_types: Optional[list[str]] = None,
    books: Optional[list[str]] = None,
    min_player_segment_rows: int = 8,
    include_market_segments: bool = True,
    min_market_segment_rows: int = 15,
    output_prefix: str = "market_reverse_engineering",
    poll_seconds: float = 300.0,
    max_runs: Optional[int] = None,
    max_wait_minutes: Optional[float] = None,
    min_inferred_rows: int = 25,
    min_book_stat_groups: int = 2,
    min_player_segment_groups: int = 5,
    require_stability_runs: int = 2,
    stability_tolerance: float = 0.10,
    min_group_rows_for_stability: int = 10,
) -> dict:
    """
    Re-run reverse-engineering until inferred values are sufficiently found/stable.

    Readiness requires both:
      1) coverage thresholds (rows/group counts)
      2) sigma-median stability across consecutive runs
    """
    run_cap = int(max_runs) if max_runs is not None else None
    if run_cap is not None and run_cap <= 0:
        run_cap = None

    wait_cap_seconds = None
    if max_wait_minutes is not None and float(max_wait_minutes) > 0:
        wait_cap_seconds = float(max_wait_minutes) * 60.0

    poll_seconds = max(0.0, float(poll_seconds))
    require_stability_runs = max(1, int(require_stability_runs))
    stability_tolerance = max(0.0, float(stability_tolerance))

    start_ts = time.monotonic()
    run_index = 0
    stable_comparison_streak = 0
    previous_signature: dict[tuple, float] = {}
    last_result: dict = {}
    last_shift: Optional[float] = None

    while True:
        run_index += 1
        run_output_prefix = f"{output_prefix}_run{run_index:03d}"
        result, summaries = _run_market_reverse_engineering_once(
            db_path=db_path,
            source=source,
            start_date=start_date,
            end_date=end_date,
            stat_types=stat_types,
            books=books,
            min_player_segment_rows=min_player_segment_rows,
            include_market_segments=include_market_segments,
            min_market_segment_rows=min_market_segment_rows,
            output_prefix=run_output_prefix,
        )

        coverage_is_ready = _coverage_ready(
            result=result,
            min_inferred_rows=min_inferred_rows,
            min_book_stat_groups=min_book_stat_groups,
            min_player_segment_groups=min_player_segment_groups,
        )

        current_signature = _build_book_stat_signature(
            summaries.get("book_stat", pd.DataFrame()),
            min_group_rows_for_stability=min_group_rows_for_stability,
        )

        if require_stability_runs == 1:
            stability_is_ready = True
            last_shift = None
            stable_comparison_streak = 0
        else:
            shift = _max_relative_signature_shift(previous_signature, current_signature)
            last_shift = shift
            if shift is not None and shift <= stability_tolerance:
                stable_comparison_streak += 1
            else:
                stable_comparison_streak = 0
            stability_is_ready = stable_comparison_streak >= (require_stability_runs - 1)

        previous_signature = current_signature
        elapsed_seconds = float(time.monotonic() - start_ts)
        last_result = {
            **result,
            "status": "waiting",
            "run_index": int(run_index),
            "runs_executed": int(run_index),
            "coverage_ready": bool(coverage_is_ready),
            "stability_ready": bool(stability_is_ready),
            "stable_comparison_streak": int(stable_comparison_streak),
            "max_relative_sigma_shift": (
                None if last_shift is None else float(last_shift)
            ),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "coverage_thresholds": {
                "min_inferred_rows": int(min_inferred_rows),
                "min_book_stat_groups": int(min_book_stat_groups),
                "min_player_segment_groups": int(min_player_segment_groups),
            },
            "stability_thresholds": {
                "require_stability_runs": int(require_stability_runs),
                "stability_tolerance": float(stability_tolerance),
                "min_group_rows_for_stability": int(min_group_rows_for_stability),
            },
        }

        if coverage_is_ready and stability_is_ready:
            last_result["status"] = "ready"
            return last_result

        if run_cap is not None and run_index >= run_cap:
            last_result["status"] = "max_runs_reached"
            return last_result

        if wait_cap_seconds is not None and elapsed_seconds >= wait_cap_seconds:
            last_result["status"] = "max_wait_reached"
            return last_result

        if poll_seconds > 0:
            time.sleep(poll_seconds)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reverse-engineer implied sportsbook pricing parameters directly from "
            "betting_lines/betting_line_snapshots per book."
        )
    )
    parser.add_argument("--db-path", default="data/database/nba_data.db")
    parser.add_argument("--source", choices=["both", "lines", "snapshots"], default="both")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--stat-types", nargs="*", default=None)
    parser.add_argument("--books", nargs="*", default=None)
    parser.add_argument("--min-player-segment-rows", type=int, default=8)
    parser.add_argument("--skip-market-segments", action="store_true")
    parser.add_argument("--min-market-segment-rows", type=int, default=15)
    parser.add_argument("--output-prefix", default="market_reverse_engineering")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Keep running until coverage/stability thresholds are met.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=300.0,
        help="Seconds to sleep between continuous runs.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Stop after N runs in continuous mode (0 = no run cap).",
    )
    parser.add_argument(
        "--max-wait-minutes",
        type=float,
        default=0.0,
        help="Stop after this many minutes in continuous mode (0 = no wait cap).",
    )
    parser.add_argument(
        "--min-inferred-rows",
        type=int,
        default=25,
        help="Coverage threshold: minimum inferred rows.",
    )
    parser.add_argument(
        "--min-book-stat-groups",
        type=int,
        default=2,
        help="Coverage threshold: minimum book/stat/distribution groups.",
    )
    parser.add_argument(
        "--min-player-segment-groups",
        type=int,
        default=5,
        help="Coverage threshold: minimum book/stat/player/distribution groups.",
    )
    parser.add_argument(
        "--require-stability-runs",
        type=int,
        default=2,
        help="Stability threshold: number of consecutive stable runs required.",
    )
    parser.add_argument(
        "--stability-tolerance",
        type=float,
        default=0.10,
        help="Max relative shift allowed for sigma medians between runs.",
    )
    parser.add_argument(
        "--min-group-rows-for-stability",
        type=int,
        default=10,
        help="Ignore small groups when checking stability.",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    if args.continuous:
        result = run_market_reverse_engineering_continuous(
            db_path=args.db_path,
            source=args.source,
            start_date=args.start_date,
            end_date=args.end_date,
            stat_types=args.stat_types,
            books=args.books,
            min_player_segment_rows=args.min_player_segment_rows,
            include_market_segments=not args.skip_market_segments,
            min_market_segment_rows=args.min_market_segment_rows,
            output_prefix=args.output_prefix,
            poll_seconds=args.poll_seconds,
            max_runs=(args.max_runs if args.max_runs > 0 else None),
            max_wait_minutes=(
                args.max_wait_minutes if args.max_wait_minutes > 0 else None
            ),
            min_inferred_rows=args.min_inferred_rows,
            min_book_stat_groups=args.min_book_stat_groups,
            min_player_segment_groups=args.min_player_segment_groups,
            require_stability_runs=args.require_stability_runs,
            stability_tolerance=args.stability_tolerance,
            min_group_rows_for_stability=args.min_group_rows_for_stability,
        )
    else:
        result = run_market_reverse_engineering(
            db_path=args.db_path,
            source=args.source,
            start_date=args.start_date,
            end_date=args.end_date,
            stat_types=args.stat_types,
            books=args.books,
            min_player_segment_rows=args.min_player_segment_rows,
            include_market_segments=not args.skip_market_segments,
            min_market_segment_rows=args.min_market_segment_rows,
            output_prefix=args.output_prefix,
        )
    print("Market reverse-engineering summary:")
    for key, value in result.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
