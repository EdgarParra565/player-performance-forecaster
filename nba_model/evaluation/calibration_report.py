"""Calibration / reliability reporting (WS10 Phase 1 — measurement).

Builds a reliability table per stat type — predicted-probability buckets vs the
realized hit rate — plus a per-stat Brier score, from settled ``predictions`` or
``bet_log`` rows. No plotting dependency: the "curve" is the bucket table (a
downstream notebook or the web app can plot ``mean_pred`` vs ``realized_rate``).

A well-calibrated model has ``realized_rate ≈ mean_pred`` in every bucket and a
low Brier score.

CLI:
    python -m nba_model.evaluation.calibration_report \\
        --db data/database/nba_data.db --source predictions
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

ARTIFACT_DIR = Path("nba_model/evaluation/artifacts")

RELIABILITY_COLUMNS = [
    "stat_type", "bucket", "bucket_low", "bucket_high",
    "n", "mean_pred", "realized_rate", "calibration_gap",
]


def _bucket_index(prob: float, n_buckets: int) -> int:
    """Equal-width bucket index in ``[0, n_buckets - 1]`` for ``prob`` in [0, 1]."""
    p = min(max(float(prob), 0.0), 1.0)
    return min(int(p * n_buckets), n_buckets - 1)


def build_reliability_table(
    df: pd.DataFrame,
    *,
    prob_col: str = "pred_prob",
    hit_col: str = "hit",
    stat_col: str = "stat_type",
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Reliability table: one row per (stat_type, probability bucket).

    ``mean_pred`` is the average predicted prob in the bucket, ``realized_rate``
    the fraction of hits, ``calibration_gap`` their difference (realized −
    predicted). Empty buckets are omitted."""
    if df is None or df.empty:
        return pd.DataFrame(columns=RELIABILITY_COLUMNS)
    if int(n_buckets) < 1:
        raise ValueError("n_buckets must be >= 1")

    work = df[[stat_col, prob_col, hit_col]].copy()
    work[prob_col] = pd.to_numeric(work[prob_col], errors="coerce")
    work[hit_col] = pd.to_numeric(work[hit_col], errors="coerce")
    work = work.dropna(subset=[prob_col, hit_col])
    if work.empty:
        return pd.DataFrame(columns=RELIABILITY_COLUMNS)

    work["bucket"] = work[prob_col].map(lambda p: _bucket_index(p, n_buckets))

    rows = []
    for (stat, bucket), grp in work.groupby([stat_col, "bucket"]):
        mean_pred = float(grp[prob_col].mean())
        realized = float(grp[hit_col].mean())
        rows.append({
            "stat_type": stat,
            "bucket": int(bucket),
            "bucket_low": round(bucket / n_buckets, 4),
            "bucket_high": round((bucket + 1) / n_buckets, 4),
            "n": int(len(grp)),
            "mean_pred": round(mean_pred, 4),
            "realized_rate": round(realized, 4),
            "calibration_gap": round(realized - mean_pred, 4),
        })
    return pd.DataFrame(rows, columns=RELIABILITY_COLUMNS).sort_values(
        ["stat_type", "bucket"]).reset_index(drop=True)


def brier_by_stat(
    df: pd.DataFrame,
    *,
    prob_col: str = "pred_prob",
    hit_col: str = "hit",
    stat_col: str = "stat_type",
) -> pd.DataFrame:
    """Per-stat Brier score = mean((pred − hit)²) over settled rows."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["stat_type", "n", "brier_score", "mean_pred",
                                     "realized_rate"])
    work = df[[stat_col, prob_col, hit_col]].copy()
    work[prob_col] = pd.to_numeric(work[prob_col], errors="coerce")
    work[hit_col] = pd.to_numeric(work[hit_col], errors="coerce")
    work = work.dropna(subset=[prob_col, hit_col])
    if work.empty:
        return pd.DataFrame(columns=["stat_type", "n", "brier_score", "mean_pred",
                                     "realized_rate"])

    rows = []
    for stat, grp in work.groupby(stat_col):
        pred = grp[prob_col].to_numpy(dtype=float)
        hit = grp[hit_col].to_numpy(dtype=float)
        rows.append({
            "stat_type": stat,
            "n": int(len(grp)),
            "brier_score": round(float(np.mean((pred - hit) ** 2)), 4),
            "mean_pred": round(float(np.mean(pred)), 4),
            "realized_rate": round(float(np.mean(hit)), 4),
        })
    return pd.DataFrame(rows).sort_values("stat_type").reset_index(drop=True)


def load_calibration_frame(
    db_path: str,
    source: str = "predictions",
) -> pd.DataFrame:
    """Load settled rows as ``(stat_type, pred_prob, hit)`` for calibration.

    ``source='predictions'``: uses ``prob_over`` vs ``outcome`` ('over' → hit=1,
    'under' → hit=0); pushes / pending excluded.
    ``source='bet_log'``: uses ``model_prob`` vs ``status`` ('won' → hit=1,
    'lost' → hit=0); push / void / pending excluded."""
    src = str(source or "predictions").strip().lower()
    with DatabaseManager(db_path=db_path) as db:
        if src == "bet_log":
            raw = pd.read_sql_query(
                "SELECT stat_type, model_prob AS pred_prob, status "
                "FROM bet_log WHERE status IN ('won','lost') "
                "AND model_prob IS NOT NULL",
                db.conn,
            )
            if raw.empty:
                return pd.DataFrame(columns=["stat_type", "pred_prob", "hit"])
            raw["hit"] = (raw["status"] == "won").astype(int)
            return raw[["stat_type", "pred_prob", "hit"]]

        if src == "predictions":
            raw = pd.read_sql_query(
                "SELECT stat_type, prob_over AS pred_prob, outcome "
                "FROM predictions WHERE outcome IN ('over','under') "
                "AND prob_over IS NOT NULL",
                db.conn,
            )
            if raw.empty:
                return pd.DataFrame(columns=["stat_type", "pred_prob", "hit"])
            raw["hit"] = (raw["outcome"] == "over").astype(int)
            return raw[["stat_type", "pred_prob", "hit"]]

    raise ValueError(f"Unknown source '{source}' (use 'predictions' or 'bet_log').")


def _reliability_markdown(reliability: pd.DataFrame, brier: pd.DataFrame,
                          source: str) -> str:
    lines = [f"# Calibration report ({source})", ""]
    lines.append("## Brier score per stat")
    lines.append("")
    if brier.empty:
        lines.append("_No settled rows._")
    else:
        lines.append("| stat | n | brier | mean_pred | realized |")
        lines.append("|---|---|---|---|---|")
        for r in brier.itertuples(index=False):
            lines.append(
                f"| {r.stat_type} | {r.n} | {r.brier_score:.4f} | "
                f"{r.mean_pred:.4f} | {r.realized_rate:.4f} |")
    lines.append("")
    lines.append("## Reliability buckets")
    lines.append("")
    if reliability.empty:
        lines.append("_No settled rows._")
    else:
        lines.append("| stat | bucket | n | mean_pred | realized | gap |")
        lines.append("|---|---|---|---|---|---|")
        for r in reliability.itertuples(index=False):
            lines.append(
                f"| {r.stat_type} | [{r.bucket_low:.1f},{r.bucket_high:.1f}) | "
                f"{r.n} | {r.mean_pred:.4f} | {r.realized_rate:.4f} | "
                f"{r.calibration_gap:+.4f} |")
    lines.append("")
    return "\n".join(lines)


def run_calibration_report(
    db_path: str = "data/database/nba_data.db",
    source: str = "predictions",
    n_buckets: int = 10,
    output_prefix: str = "calibration",
    artifact_dir: str | Path = ARTIFACT_DIR,
) -> dict:
    """Load settled rows, build reliability + Brier tables, write CSV + md."""
    frame = load_calibration_frame(db_path, source=source)
    reliability = build_reliability_table(frame, n_buckets=n_buckets)
    brier = brier_by_stat(frame)

    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reliability_csv = out_dir / f"{output_prefix}_reliability_{source}_{ts}.csv"
    brier_csv = out_dir / f"{output_prefix}_brier_{source}_{ts}.csv"
    md_path = out_dir / f"{output_prefix}_{source}_{ts}.md"

    reliability.to_csv(reliability_csv, index=False)
    brier.to_csv(brier_csv, index=False)
    md_path.write_text(
        _reliability_markdown(reliability, brier, source), encoding="utf-8")

    return {
        "source": source,
        "settled_rows": int(len(frame)),
        "reliability_rows": int(len(reliability)),
        "reliability_csv": str(reliability_csv),
        "brier_csv": str(brier_csv),
        "md_path": str(md_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a calibration / reliability report per stat type.")
    p.add_argument("--db", default="data/database/nba_data.db", dest="db_path")
    p.add_argument("--source", choices=["predictions", "bet_log"],
                   default="predictions")
    p.add_argument("--n-buckets", type=int, default=10)
    p.add_argument("--output-prefix", default="calibration")
    p.add_argument("--artifact-dir", default=str(ARTIFACT_DIR))
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging()
    result = run_calibration_report(
        db_path=args.db_path, source=args.source, n_buckets=args.n_buckets,
        output_prefix=args.output_prefix, artifact_dir=args.artifact_dir,
    )
    logger.info(
        "calibration report complete",
        extra={"source": args.source, "n_buckets": int(args.n_buckets),
               "result_keys": sorted(result.keys())},
    )
    print("Calibration report:")
    for key, value in result.items():
        print(f"- {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
