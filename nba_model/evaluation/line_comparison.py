"""Cross-book and model-vs-book line comparison engine."""

import argparse
import json
from datetime import datetime
from pathlib import Path
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


def _prob_to_american(prob: float) -> Optional[int]:
    if prob is None or not np.isfinite(prob):
        return None
    p = float(prob)
    if p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


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


def load_betting_lines(
    db_path: str = "data/database/nba_data.db",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stat_types: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load betting lines from DB with optional filters."""
    query = """
        SELECT
            bl.player_id,
            COALESCE(p.name, '') AS player_name,
            bl.game_date,
            lower(bl.stat_type) AS stat_type,
            bl.book,
            bl.line_value,
            bl.over_odds,
            bl.under_odds
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

    with DatabaseManager(db_path=db_path) as db:
        lines_df = pd.read_sql_query(query, db.conn, params=params)

    if lines_df.empty:
        return lines_df

    lines_df["game_date"] = pd.to_datetime(lines_df["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    lines_df["book"] = lines_df["book"].astype(str).str.strip()
    lines_df["stat_type"] = lines_df["stat_type"].map(_normalize_stat_type)
    lines_df["line_value"] = pd.to_numeric(lines_df["line_value"], errors="coerce")
    lines_df["over_odds"] = pd.to_numeric(lines_df["over_odds"], errors="coerce")
    lines_df["under_odds"] = pd.to_numeric(lines_df["under_odds"], errors="coerce")
    lines_df = lines_df.dropna(subset=["player_id", "game_date", "stat_type", "book", "line_value"])
    return lines_df.reset_index(drop=True)


def load_latest_predictions(
    db_path: str = "data/database/nba_data.db",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stat_types: Optional[list[str]] = None,
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
            p.line_value AS model_line,
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
    pred_df = pred_df.sort_values(
        ["prediction_id", "config_created_at", "created_at"],
        ascending=[True, True, True],
    ).drop_duplicates(subset=["prediction_id"], keep="last")

    pred_df = pred_df.sort_values(["created_at", "prediction_id"]).drop_duplicates(
        subset=["player_id", "game_date", "stat_type"],
        keep="last",
    )
    pred_df["distribution"] = pred_df["config_json"].apply(_distribution_from_config)
    return pred_df.reset_index(drop=True)


def build_book_vs_book_comparison(lines_df: pd.DataFrame, min_books: int = 2) -> pd.DataFrame:
    """Compare books against each other and surface best available lines/prices."""
    if lines_df.empty:
        return pd.DataFrame()

    df = lines_df.copy()
    df["over_implied_prob"] = df["over_odds"].apply(_safe_implied_prob)
    df["under_implied_prob"] = df["under_odds"].apply(_safe_implied_prob)

    rows = []
    keys = ["player_id", "player_name", "game_date", "stat_type"]
    for key, group in df.groupby(keys):
        if group["book"].nunique() < max(1, int(min_books)):
            continue

        row = {
            "player_id": key[0],
            "player_name": key[1],
            "game_date": key[2],
            "stat_type": key[3],
            "book_count": int(group["book"].nunique()),
            "line_min": float(group["line_value"].min()),
            "line_max": float(group["line_value"].max()),
            "line_range": float(group["line_value"].max() - group["line_value"].min()),
            "line_median": float(group["line_value"].median()),
        }

        best_over_line = group.loc[group["line_value"].idxmin()]
        best_under_line = group.loc[group["line_value"].idxmax()]
        row.update(
            {
                "best_over_line_book": best_over_line["book"],
                "best_over_line_value": float(best_over_line["line_value"]),
                "best_under_line_book": best_under_line["book"],
                "best_under_line_value": float(best_under_line["line_value"]),
            }
        )

        over_price_rows = group.dropna(subset=["over_implied_prob"])
        if not over_price_rows.empty:
            best_over_price = over_price_rows.loc[over_price_rows["over_implied_prob"].idxmin()]
            row["best_over_price_book"] = best_over_price["book"]
            row["best_over_price_odds"] = int(best_over_price["over_odds"])
            row["best_over_price_implied_prob"] = float(best_over_price["over_implied_prob"])
        else:
            row["best_over_price_book"] = None
            row["best_over_price_odds"] = None
            row["best_over_price_implied_prob"] = None

        under_price_rows = group.dropna(subset=["under_implied_prob"])
        if not under_price_rows.empty:
            best_under_price = under_price_rows.loc[under_price_rows["under_implied_prob"].idxmin()]
            row["best_under_price_book"] = best_under_price["book"]
            row["best_under_price_odds"] = int(best_under_price["under_odds"])
            row["best_under_price_implied_prob"] = float(best_under_price["under_implied_prob"])
        else:
            row["best_under_price_book"] = None
            row["best_under_price_odds"] = None
            row["best_under_price_implied_prob"] = None

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["game_date", "player_name", "stat_type"]).reset_index(drop=True)


def _model_prob_at_line(row) -> Optional[float]:
    try:
        line = float(row["line_value"])
        mu = float(row["predicted_mean"])
        sigma = float(row["predicted_std"])
    except (TypeError, ValueError):
        return None

    if sigma > 0:
        try:
            return float(
                prob_over_distribution(
                    line=line,
                    mu=mu,
                    sigma=sigma,
                    distribution=str(row.get("distribution") or "normal"),
                )
            )
        except ValueError:
            return float(prob_over_distribution(line=line, mu=mu, sigma=sigma, distribution="normal"))

    # Fallback when sigma is invalid and book line equals model line.
    model_line = row.get("model_line")
    prob_over = row.get("prob_over")
    if model_line is not None and prob_over is not None:
        try:
            if abs(float(model_line) - line) < 1e-9:
                return float(prob_over)
        except (TypeError, ValueError):
            pass
    return None


def build_model_vs_book_comparison(
    lines_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    edge_threshold: float = 0.0,
) -> pd.DataFrame:
    """Compare model fair probabilities against each available book line."""
    if lines_df.empty or predictions_df.empty:
        return pd.DataFrame()

    merge_keys = ["player_id", "game_date", "stat_type"]
    merged = lines_df.merge(predictions_df, on=merge_keys, how="inner", suffixes=("", "_pred"))
    if merged.empty:
        return pd.DataFrame()

    merged["model_prob_over_at_line"] = merged.apply(_model_prob_at_line, axis=1)
    merged["model_prob_under_at_line"] = merged["model_prob_over_at_line"].apply(
        lambda p: None if p is None or pd.isna(p) else float(1.0 - p)
    )
    merged["over_implied_prob"] = merged["over_odds"].apply(_safe_implied_prob)
    merged["under_implied_prob"] = merged["under_odds"].apply(_safe_implied_prob)
    merged["edge_over"] = merged.apply(
        lambda row: (
            row["model_prob_over_at_line"] - row["over_implied_prob"]
            if row["model_prob_over_at_line"] is not None and row["over_implied_prob"] is not None
            else np.nan
        ),
        axis=1,
    )
    merged["edge_under"] = merged.apply(
        lambda row: (
            row["model_prob_under_at_line"] - row["under_implied_prob"]
            if row["model_prob_under_at_line"] is not None and row["under_implied_prob"] is not None
            else np.nan
        ),
        axis=1,
    )

    def _best_side(row):
        edge_over = row["edge_over"]
        edge_under = row["edge_under"]
        if pd.isna(edge_over) and pd.isna(edge_under):
            return "none", np.nan
        if pd.isna(edge_under) or (not pd.isna(edge_over) and edge_over >= edge_under):
            return "over", float(edge_over)
        return "under", float(edge_under)

    best_side = merged.apply(_best_side, axis=1)
    merged["recommended_side"] = [item[0] for item in best_side]
    merged["best_edge"] = [item[1] for item in best_side]
    merged["model_fair_over_odds"] = merged["model_prob_over_at_line"].apply(_prob_to_american)
    merged["model_fair_under_odds"] = merged["model_prob_under_at_line"].apply(_prob_to_american)

    group_keys = ["player_id", "player_name", "game_date", "stat_type"]
    merged["best_edge_rank"] = merged["best_edge"].fillna(-1e9)
    idx = merged.groupby(group_keys)["best_edge_rank"].idxmax()
    best_rows = merged.loc[idx].copy()

    spread = (
        merged.groupby(group_keys, as_index=False)
        .agg(
            books_compared=("book", "nunique"),
            line_median=("line_value", "median"),
            line_range=("line_value", lambda x: float(max(x) - min(x))),
        )
    )
    best_rows = best_rows.merge(spread, on=group_keys, how="left")
    best_rows["value_flag"] = best_rows["best_edge"].fillna(-1.0) >= float(edge_threshold)

    keep_cols = [
        "player_id",
        "player_name",
        "game_date",
        "stat_type",
        "books_compared",
        "line_median",
        "line_range",
        "book",
        "line_value",
        "recommended_side",
        "best_edge",
        "model_prob_over_at_line",
        "model_prob_under_at_line",
        "over_odds",
        "under_odds",
        "model_fair_over_odds",
        "model_fair_under_odds",
        "value_flag",
    ]
    return best_rows[keep_cols].sort_values(["game_date", "player_name", "stat_type"]).reset_index(drop=True)


def run_line_comparison(
    db_path: str = "data/database/nba_data.db",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stat_types: Optional[list[str]] = None,
    min_books: int = 2,
    edge_threshold: float = 0.02,
    output_prefix: str = "line_comparison",
) -> dict:
    """Run book-vs-book and model-vs-book comparison, writing artifacts to disk."""
    lines_df = load_betting_lines(
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        stat_types=stat_types,
    )
    predictions_df = load_latest_predictions(
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        stat_types=stat_types,
    )
    book_vs_book_df = build_book_vs_book_comparison(lines_df=lines_df, min_books=min_books)
    model_vs_book_df = build_model_vs_book_comparison(
        lines_df=lines_df,
        predictions_df=predictions_df,
        edge_threshold=edge_threshold,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    book_vs_book_path = ARTIFACT_DIR / f"{output_prefix}_book_vs_book_{ts}.csv"
    model_vs_book_path = ARTIFACT_DIR / f"{output_prefix}_model_vs_book_{ts}.csv"
    summary_md_path = ARTIFACT_DIR / f"{output_prefix}_summary_{ts}.md"

    book_vs_book_df.to_csv(book_vs_book_path, index=False)
    model_vs_book_df.to_csv(model_vs_book_path, index=False)

    if not model_vs_book_df.empty and "best_edge" in model_vs_book_df.columns:
        top_value = model_vs_book_df.sort_values("best_edge", ascending=False).head(25)
    else:
        top_value = pd.DataFrame()
    summary_lines = [
        "# Line Comparison Summary",
        "",
        f"- Date range: {start_date or 'all'} to {end_date or 'all'}",
        f"- Stat types: {', '.join(stat_types) if stat_types else 'all'}",
        f"- Min books for book-vs-book: {min_books}",
        f"- Edge threshold: {edge_threshold}",
        f"- Book-vs-book rows: {len(book_vs_book_df)}",
        f"- Model-vs-book rows: {len(model_vs_book_df)}",
        "",
        "## Top Model-vs-Book Opportunities",
        top_value.to_string(index=False) if not top_value.empty else "No rows.",
    ]
    summary_md_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "book_vs_book_csv": str(book_vs_book_path),
        "model_vs_book_csv": str(model_vs_book_path),
        "summary_md": str(summary_md_path),
        "book_vs_book_rows": int(len(book_vs_book_df)),
        "model_vs_book_rows": int(len(model_vs_book_df)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-book and model-vs-book line comparison.")
    parser.add_argument("--db-path", default="data/database/nba_data.db")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--stat-types", nargs="*", default=None)
    parser.add_argument("--min-books", type=int, default=2)
    parser.add_argument("--edge-threshold", type=float, default=0.02)
    parser.add_argument("--output-prefix", default="line_comparison")
    return parser


def main():
    args = _build_parser().parse_args()
    paths = run_line_comparison(
        db_path=args.db_path,
        start_date=args.start_date,
        end_date=args.end_date,
        stat_types=args.stat_types,
        min_books=args.min_books,
        edge_threshold=args.edge_threshold,
        output_prefix=args.output_prefix,
    )
    print("\nLine comparison complete.")
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
