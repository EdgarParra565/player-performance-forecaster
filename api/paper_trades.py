"""Read-only paper-trade + calibration views for the flagship /paper-trades page.

- Picks come straight from ``bet_log`` (written by the ``bet_slip`` CLI; empty
  until the season starts). Nothing here writes.
- Calibration wraps ``nba_model.evaluation.calibration_report`` — the SAME
  ``load_calibration_frame`` / ``build_reliability_table`` / ``brier_by_stat``
  path the CLI report uses (WS10 Phase 1 measurement).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.evaluation import calibration_report as cr

SETTLED = ("won", "lost", "push", "void")
STATUS_FILTERS = ("all", "pending", "settled")
CALIBRATION_SOURCES = ("bet_log", "predictions")

_COLUMNS = (
    "log_id", "created_at_utc", "game_date", "player_id", "player_name",
    "stat_type", "book", "line", "side", "model_prob", "implied_prob", "edge",
    "model_mode", "stake_units", "status", "settled_at_utc", "actual_value",
    "clv_delta",
)


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if pd.notna(x) and x not in (float("inf"), float("-inf")) else None


def _est_profit(row: dict) -> Optional[float]:
    """Units won/lost at the logged implied price (estimate: implied_prob
    includes the book's vig, so it slightly understates the payout)."""
    stake = _f(row.get("stake_units")) or 1.0
    implied = _f(row.get("implied_prob"))
    status = row.get("status")
    if status == "lost":
        return -stake
    if status in ("push", "void"):
        return 0.0
    if status == "won" and implied and 0 < implied < 1:
        return stake * (1.0 / implied - 1.0)
    return None


def list_paper_trades(db_path: str, status: str = "all", limit: int = 200) -> dict:
    where = ""
    if status == "pending":
        where = "WHERE status = 'pending'"
    elif status == "settled":
        where = "WHERE status IN ('won','lost','push','void')"
    with DatabaseManager(db_path=db_path) as db:
        rows = db.conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM bet_log {where} "  # noqa: S608 — fixed fragments only
            "ORDER BY created_at_utc DESC, log_id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        counts = dict(db.conn.execute(
            "SELECT status, COUNT(*) FROM bet_log GROUP BY status").fetchall())
        clv = db.conn.execute(
            "SELECT COUNT(clv_delta), AVG(clv_delta), "
            "SUM(CASE WHEN clv_delta > 0 THEN 1 ELSE 0 END) "
            "FROM bet_log WHERE clv_delta IS NOT NULL").fetchone()
        settled_rows = db.conn.execute(
            "SELECT status, stake_units, implied_prob FROM bet_log "
            "WHERE status IN ('won','lost','push','void')").fetchall()

    out_rows = []
    for r in rows:
        d = dict(zip(_COLUMNS, r))
        for k in ("line", "model_prob", "implied_prob", "edge", "stake_units",
                  "actual_value", "clv_delta"):
            d[k] = _f(d.get(k))
        d["est_profit_units"] = _est_profit(d)
        out_rows.append(d)

    won, lost = int(counts.get("won", 0)), int(counts.get("lost", 0))
    profits = [_est_profit(dict(zip(("status", "stake_units", "implied_prob"), r)))
               for r in settled_rows]
    known = [p for p in profits if p is not None]
    n_clv = int(clv[0] or 0)
    return {
        "status_filter": status,
        "rows": out_rows,
        "summary": {
            "total": int(sum(counts.values())),
            "pending": int(counts.get("pending", 0)),
            "won": won,
            "lost": lost,
            "push": int(counts.get("push", 0)),
            "void": int(counts.get("void", 0)),
            "win_rate": (won / (won + lost)) if (won + lost) else None,
            "n_clv": n_clv,
            "mean_clv": _f(clv[1]) if n_clv else None,
            "positive_clv_rate": (int(clv[2] or 0) / n_clv) if n_clv else None,
            "est_units": round(sum(known), 4) if known else None,
        },
    }


def calibration(
    db_path: str,
    source: str = "bet_log",
    stat: Optional[str] = None,
    n_buckets: int = 10,
) -> dict:
    frame = cr.load_calibration_frame(db_path, source=source)
    stats_available = sorted(frame["stat_type"].dropna().unique().tolist()) if not frame.empty else []
    if stat:
        frame = frame[frame["stat_type"] == stat]
    work = frame.copy()
    # Pool across stats for the headline curve ("all").
    work["stat_type"] = stat or "all"
    reliability = cr.build_reliability_table(work, n_buckets=n_buckets)
    brier_all = cr.brier_by_stat(work)
    brier_per_stat = cr.brier_by_stat(frame)

    def _records(df: pd.DataFrame) -> list[dict]:
        return [{k: (_f(v) if k not in ("stat_type",) else v) for k, v in rec.items()}
                for rec in df.to_dict("records")]

    overall = _records(brier_all)[0] if not brier_all.empty else None
    return {
        "source": source,
        "stat": stat or "all",
        "n_buckets": int(n_buckets),
        "n_settled": int(len(frame)),
        "stats_available": stats_available,
        "reliability": _records(reliability),
        "brier": overall,
        "brier_by_stat": _records(brier_per_stat),
    }
