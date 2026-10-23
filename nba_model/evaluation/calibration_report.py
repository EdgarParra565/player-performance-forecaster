"""Calibration / reliability reporting (WS10 Phase 1 — measurement).

Builds a reliability table per stat type — predicted-probability buckets vs the
realized hit rate — plus a per-stat Brier score, from settled ``predictions`` or
``bet_log`` rows. No plotting dependency: the "curve" is the bucket table (a
downstream notebook or the web app can plot ``mean_pred`` vs ``realized_rate``).

A well-calibrated model has ``realized_rate ≈ mean_pred`` in every bucket and a
low Brier score.

``--source llm`` (WS11) scores one ``llm_forecasts`` run: every LLM next to the
model, the market and a constant 0.5 on the exact same settled props, with the
Brier decomposition (reliability / resolution / uncertainty) and expected
calibration error — so "calibrated but no resolution" (hedging at 0.5) shows.
Historical (retro) rows whose model may have trained on the result are
reported separately, never as the headline.

CLI:
    python -m nba_model.evaluation.calibration_report \\
        --db data/database/nba_data.db --source predictions
    python -m nba_model.evaluation.calibration_report --source llm --run-id RUN
"""
from __future__ import annotations

import argparse
import re
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


DECOMPOSITION_COLUMNS = [
    "n", "brier", "reliability", "resolution", "uncertainty", "within_bin",
    "ece", "mean_pred", "base_rate",
]


def brier_decomposition(pred, hit, *, n_buckets: int = 10) -> dict:
    """Murphy decomposition of the Brier score over the reliability buckets,
    plus expected calibration error (ECE).

    ``brier = reliability − resolution + uncertainty + within_bin`` exactly:
    reliability = Σ n_k (mean_pred_k − realized_k)² / N (lower is better),
    resolution = Σ n_k (realized_k − base_rate)² / N (higher is better),
    uncertainty = base_rate (1 − base_rate) (the props, not the forecaster),
    within_bin = the remainder the buckets can't attribute: within-bucket
    forecast variance minus twice its covariance with the outcomes (0 when
    every bucket holds one value; negative when the forecaster ranks props
    correctly inside a bucket). ECE = Σ n_k |realized_k − mean_pred_k| / N.
    A forecaster that always says 0.5 has resolution 0."""
    p = np.asarray(pred, dtype=float)
    h = np.asarray(hit, dtype=float)
    keep = ~(np.isnan(p) | np.isnan(h))
    p, h = p[keep], h[keep]
    n = int(len(p))
    if n == 0:
        return {c: (0 if c == "n" else float("nan")) for c in DECOMPOSITION_COLUMNS}
    if int(n_buckets) < 1:
        raise ValueError("n_buckets must be >= 1")
    buckets = np.array([_bucket_index(v, n_buckets) for v in p])
    base = float(h.mean())
    rel = res = ece = 0.0
    for b in np.unique(buckets):
        mask = buckets == b
        n_k = int(mask.sum())
        f_k = float(p[mask].mean())
        o_k = float(h[mask].mean())
        rel += n_k * (f_k - o_k) ** 2
        res += n_k * (o_k - base) ** 2
        ece += n_k * abs(o_k - f_k)
    rel, res, ece = rel / n, res / n, ece / n
    brier = float(np.mean((p - h) ** 2))
    unc = base * (1.0 - base)
    return {
        "n": n, "brier": brier, "reliability": rel, "resolution": res,
        "uncertainty": unc, "within_bin": brier - (rel - res + unc), "ece": ece,
        "mean_pred": float(p.mean()), "base_rate": base,
    }


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
    run_id: Optional[str] = None,
) -> dict:
    """Load settled rows, build reliability + Brier tables, write CSV + md.
    ``source='llm'`` scores one ``llm_forecasts`` run (``run_id``; newest
    when None) — see ``run_llm_calibration_report``."""
    if str(source or "").strip().lower() == "llm":
        return run_llm_calibration_report(
            db_path, run_id=run_id, n_buckets=n_buckets,
            output_prefix=output_prefix, artifact_dir=artifact_dir)
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


# ---------------------------------------------------------------------------
# --source llm (WS11): LLM forecasters vs model vs market vs constant 0.5
# ---------------------------------------------------------------------------

PROP_KEY = ["player_id", "game_date", "stat_type", "line_value"]
BASELINES = ("model", "market", "constant 0.5")
COVERAGE_COLUMNS = ["forecaster", "rows", "ok", "invalid", "refusal", "scored",
                    "push", "pending", "leak_risk_rows"]


def _llm_label(model_name: str, arm: str) -> str:
    return str(model_name) if arm == "blind" else f"{model_name} [{arm} arm]"


def load_llm_calibration_frame(db_path: str, run_id: Optional[str] = None) -> dict:
    """One ``llm_forecasts`` run (the newest when ``run_id`` is None) as
    ``{"run_id", "forecasts", "long", "market_real"}``.

    ``long`` has one row per (forecaster, prop): each LLM answer plus the
    baselines on the same props — ``model`` (``prob_over`` of the forecast's
    ``prediction_id``), ``market`` (de-vigged real ``betting_lines`` price at
    the line, else the -110 assumption) and ``constant 0.5``. ``hit`` is 1 for
    over, 0 for under, NaN for push / pending. ``leak_risk`` is per prop:
    some LLM row on it is retro with an unknown or on/after-game training
    cutoff. ``market_real`` counts props with a real price. Read-only open;
    a DB without the table yields empty frames."""
    from nba_model.evaluation.llm_forecast import (
        ASSUMED_PRICE_PROB, _parse_config, market_prob_over,
    )

    empty = {"run_id": run_id, "forecasts": pd.DataFrame(),
             "long": pd.DataFrame(), "market_real": 0, "model_lookahead_props": 0}
    with DatabaseManager(db_path=db_path, read_only=True) as db:
        if not db.has_table("llm_forecasts"):
            return empty
        if run_id is None:
            row = db.conn.execute(
                "SELECT run_id FROM llm_forecasts "
                "ORDER BY created_at_utc DESC, forecast_id DESC LIMIT 1").fetchone()
            if row is None:
                return empty
            run_id = row[0]
        forecasts = pd.read_sql_query(
            "SELECT forecast_id, run_id, provider, model_name, prompt_version, arm, "
            "player_id, player_name, game_date, stat_type, line_value, prob_over, "
            "parse_status, retro, training_cutoff, prediction_id, outcome, "
            "actual_result FROM llm_forecasts WHERE run_id = ? ORDER BY forecast_id",
            db.conn, params=(run_id,))
        if forecasts.empty:
            return {**empty, "run_id": run_id}
        props = (forecasts.sort_values("prediction_id")
                 .drop_duplicates(PROP_KEY)[PROP_KEY + ["prediction_id", "outcome"]]
                 .reset_index(drop=True))
        ids = [int(x) for x in props["prediction_id"].dropna().unique()]
        model_prob: dict = {}
        context_ids: set = set()   # backtest rows built with context features
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            for pid, prob, cfg in db.conn.execute(
                    "SELECT p.prediction_id, p.prob_over, "
                    "(SELECT c.config_json FROM prediction_configs c "
                    "  WHERE c.prediction_id = p.prediction_id "
                    "  ORDER BY c.config_id DESC LIMIT 1) "
                    f"FROM predictions p WHERE p.prediction_id IN ({', '.join('?' * len(part))})",
                    part):
                model_prob[int(pid)] = prob
                if _parse_config(cfg).get("context_features_enabled"):
                    context_ids.add(int(pid))
        market = [market_prob_over(db.conn, r.player_id, r.game_date,
                                   r.stat_type, r.line_value)[0]
                  for r in props.itertuples(index=False)]

    hit_map = {"over": 1.0, "under": 0.0}
    cutoff = pd.to_datetime(forecasts["training_cutoff"], errors="coerce")
    game_day = pd.to_datetime(forecasts["game_date"], errors="coerce")
    forecasts["leak_risk"] = ((forecasts["retro"] == 1)
                              & (cutoff.isna() | (cutoff >= game_day)))
    prop_leak = (forecasts.groupby(PROP_KEY)["leak_risk"].any()
                 .rename("prop_leak_risk").reset_index())
    props = props.merge(prop_leak, on=PROP_KEY, how="left")
    props["hit"] = props["outcome"].map(hit_map)
    props["model"] = props["prediction_id"].map(
        lambda x: model_prob.get(int(x)) if pd.notna(x) else None)
    props["market"] = [ASSUMED_PRICE_PROB if p is None else p for p in market]
    props["constant 0.5"] = 0.5

    llm = forecasts.merge(props[PROP_KEY + ["prop_leak_risk"]], on=PROP_KEY, how="left")
    llm_long = pd.DataFrame({
        "forecaster": [_llm_label(m, a) for m, a in zip(llm["model_name"], llm["arm"])],
        "kind": "llm",
        **{k: llm[k] for k in PROP_KEY},
        "pred_prob": pd.to_numeric(llm["prob_over"], errors="coerce").where(
            llm["parse_status"] == "ok"),
        "hit": llm["outcome"].map(hit_map),
        "leak_risk": llm["prop_leak_risk"].astype(bool),
    })
    base_long = [pd.DataFrame({
        "forecaster": name, "kind": "baseline", **{k: props[k] for k in PROP_KEY},
        "pred_prob": pd.to_numeric(props[name], errors="coerce"),
        "hit": props["hit"], "leak_risk": props["prop_leak_risk"].astype(bool),
    }) for name in BASELINES]
    long = pd.concat([llm_long, *base_long], ignore_index=True)
    # The backtest's points mean applies injury_proxy, which includes the
    # target game's own minutes — those model numbers are optimistic.
    lookahead = sum(1 for stat, pid in zip(props["stat_type"], props["prediction_id"])
                    if stat == "points" and pd.notna(pid) and int(pid) in context_ids)
    return {"run_id": run_id, "forecasts": forecasts, "long": long,
            "market_real": int(sum(p is not None for p in market)),
            "model_lookahead_props": int(lookahead)}


def common_prop_set(long: pd.DataFrame) -> pd.DataFrame:
    """Scored rows (over/under) restricted to props where EVERY forecaster
    that answered anything has a probability — the like-for-like set. A
    baseline with no value anywhere (e.g. no model row) is left out instead
    of emptying the set."""
    scored = long[long["hit"].notna() & long["pred_prob"].notna()]
    if scored.empty:
        return scored
    n_forecasters = scored["forecaster"].nunique()
    per_prop = scored.groupby(PROP_KEY)["forecaster"].nunique()
    keep = per_prop[per_prop == n_forecasters].reset_index()[PROP_KEY]
    return scored.merge(keep, on=PROP_KEY, how="inner")


def _forecaster_order(frame: pd.DataFrame) -> list:
    """LLMs alphabetically, then the baselines in ``BASELINES`` order."""
    names = set(frame["forecaster"])
    return sorted(names - set(BASELINES)) + [b for b in BASELINES if b in names]


def score_forecasters(frame: pd.DataFrame, *, n_buckets: int = 10) -> pd.DataFrame:
    """One row per forecaster: Brier decomposition + ECE."""
    cols = ["forecaster", "kind"] + DECOMPOSITION_COLUMNS
    if frame is None or frame.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for name in _forecaster_order(frame):
        grp = frame[frame["forecaster"] == name]
        rows.append({"forecaster": name, "kind": grp["kind"].iloc[0],
                     **brier_decomposition(grp["pred_prob"], grp["hit"],
                                           n_buckets=n_buckets)})
    return pd.DataFrame(rows, columns=cols)


def _n_props(frame: pd.DataFrame) -> int:
    return 0 if frame.empty else int(len(frame.drop_duplicates(PROP_KEY)))


def _llm_coverage(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, arm), g in forecasts.groupby(["model_name", "arm"], sort=False):
        status = g["parse_status"]
        rows.append({
            "forecaster": _llm_label(model, arm), "rows": int(len(g)),
            "ok": int((status == "ok").sum()),
            "invalid": int((status == "invalid").sum()),
            "refusal": int((status == "refusal").sum()),
            "scored": int(((status == "ok") & g["outcome"].isin(["over", "under"])).sum()),
            "push": int((g["outcome"] == "push").sum()),
            "pending": int(g["outcome"].isna().sum()),
            "leak_risk_rows": int(g["leak_risk"].sum()),
        })
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def _concat(frames: list, columns: list) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _llm_section(frame: pd.DataFrame, n_buckets: int):
    """(summary, brier per stat, reliability) for one block of props."""
    summary = score_forecasters(frame, n_buckets=n_buckets)
    if frame.empty:
        per_stat = pd.DataFrame(columns=["forecaster", "stat_type", "n", "brier_score",
                                         "mean_pred", "realized_rate"])
        reliability = pd.DataFrame(columns=["forecaster"] + RELIABILITY_COLUMNS[1:])
        return summary, per_stat, reliability
    order = _forecaster_order(frame)
    per_stat = _concat(
        [brier_by_stat(frame[frame["forecaster"] == name]).assign(forecaster=name)
         for name in order], [])
    per_stat = per_stat[["forecaster", "stat_type", "n", "brier_score",
                         "mean_pred", "realized_rate"]]
    reliability = build_reliability_table(
        frame, stat_col="forecaster", n_buckets=n_buckets
    ).rename(columns={"stat_type": "forecaster"})
    rank = {name: i for i, name in enumerate(order)}
    reliability = reliability.sort_values(
        ["forecaster", "bucket"], key=lambda s: s.map(rank) if s.name == "forecaster" else s
    ).reset_index(drop=True)
    return summary, per_stat, reliability


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{round(float(v), 4) + 0.0:.4f}"   # + 0.0: no "-0.0000" from float noise


def _llm_markdown(meta: dict, coverage: pd.DataFrame, sections: dict) -> str:
    n_scored = meta["scored_props"]
    lines = [f"# LLM calibration report — run `{meta['run_id']}`", ""]
    lines.append(
        f"Forecasters: {', '.join(meta['llm_forecasters']) or 'none'} "
        f"(prompt {', '.join(meta['prompt_versions']) or '—'}). Baselines on the same "
        "props: **model** (`predictions.prob_over`, production config), **market** "
        "(de-vigged `betting_lines` price at the line; else -110 → 0.5238), "
        "**constant 0.5**.")
    lines.append("")
    lines.append(
        f"Scored on the common prop set: **{n_scored}** settled props where every "
        "forecaster has a probability (pushes, pending and unreadable replies are "
        "excluded — see Coverage).")
    lines.append("")
    if not coverage.empty and n_scored < int(coverage["scored"].max()):
        low = coverage[coverage["scored"] < coverage["scored"].max()]
        who = ", ".join(f"{r.forecaster} ({r.scored} scored)" for r in low.itertuples())
        lines.append(
            f"> **Smaller common set:** some forecasters scored "
            f"{int(coverage['scored'].max())} props, but the set every forecaster "
            f"answered is {n_scored} — limited by {who or 'baseline coverage'}.")
        lines.append("")
    if meta.get("model_lookahead_props"):
        lines.append(
            f"> **Model baseline caveat:** {meta['model_lookahead_props']} points props "
            "use backtest `predictions` rows built with context features; that "
            "harness's `injury_proxy` reads the target game's own minutes (look-ahead), "
            "so the model's points numbers are optimistic.")
        lines.append("")
    if meta["leak_risk_props"]:
        lines.append(
            f"> **Possible training-data leakage:** {meta['leak_risk_props']} of "
            f"{n_scored} props are historical (`retro=1`) with an unknown or "
            "on/after-game model training cutoff. They are reported separately below "
            "and are **not a headline result**.")
        lines.append("")
    lines.append(
        f"Market baseline: real two-sided price on {meta['market_real']} of "
        f"{meta['props_in_run']} props; the rest use the -110 assumption "
        "(the same 0.5238 for every prop — no resolution by construction).")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    if coverage.empty:
        lines.append("_No forecasts in this run._")
    else:
        lines.append("| forecaster | rows | ok | invalid | refusal | scored | push | "
                     "pending | leak-risk rows |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in coverage.itertuples(index=False):
            lines.append(f"| {r.forecaster} | {r.rows} | {r.ok} | {r.invalid} | "
                         f"{r.refusal} | {r.scored} | {r.push} | {r.pending} | "
                         f"{r.leak_risk_rows} |")
    lines.append("")
    titles = {
        "headline": "Headline (leak-safe)",
        "leak_risk": "Reported separately — possible training-data leakage",
    }
    for key, (summary, per_stat, reliability) in sections.items():
        n_props = 0 if summary.empty else int(summary["n"].max())
        lines.append(f"## {titles[key]} — {n_props} props")
        lines.append("")
        if summary.empty:
            lines.append("_None._" if key == "leak_risk" else
                         "_None: no scored prop is leak-safe (forward rows, or retro rows "
                         "after a known training cutoff)._")
            lines.append("")
            continue
        lines.append("| forecaster | n | brier | reliability | resolution | "
                     "uncertainty | within_bin | ece | mean_pred | base_rate |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in summary.itertuples(index=False):
            lines.append(
                f"| {r.forecaster} | {r.n} | {_fmt(r.brier)} | {_fmt(r.reliability)} | "
                f"{_fmt(r.resolution)} | {_fmt(r.uncertainty)} | {_fmt(r.within_bin)} | "
                f"{_fmt(r.ece)} | {_fmt(r.mean_pred)} | {_fmt(r.base_rate)} |")
        lines.append("")
        lines.append("Brier per stat:")
        lines.append("")
        lines.append("| forecaster | stat | n | brier | mean_pred | realized |")
        lines.append("|---|---|---|---|---|---|")
        for r in per_stat.itertuples(index=False):
            lines.append(f"| {r.forecaster} | {r.stat_type} | {r.n} | "
                         f"{r.brier_score:.4f} | {r.mean_pred:.4f} | {r.realized_rate:.4f} |")
        lines.append("")
        lines.append("Reliability buckets:")
        lines.append("")
        lines.append("| forecaster | bucket | n | mean_pred | realized | gap |")
        lines.append("|---|---|---|---|---|---|")
        for r in reliability.itertuples(index=False):
            lines.append(
                f"| {r.forecaster} | [{r.bucket_low:.1f},{r.bucket_high:.1f}) | {r.n} | "
                f"{r.mean_pred:.4f} | {r.realized_rate:.4f} | {r.calibration_gap:+.4f} |")
        lines.append("")
    lines.append(
        "How to read: brier = reliability − resolution + uncertainty + within_bin "
        "(within_bin = what the buckets can't attribute: forecast variance inside a "
        "bucket minus twice its covariance with the outcomes — 0 when each bucket "
        "holds one value, negative when props are ranked well inside a bucket). "
        "Lower brier / reliability / ECE is better; higher resolution is better; "
        "uncertainty depends only on the props. A forecaster that hedges at 0.5 can "
        "be well calibrated with zero resolution.")
    lines.append("")
    return "\n".join(lines)


def run_llm_calibration_report(
    db_path: str = "data/database/nba_data.db",
    run_id: Optional[str] = None,
    n_buckets: int = 10,
    output_prefix: str = "calibration",
    artifact_dir: str | Path = ARTIFACT_DIR,
) -> dict:
    """Score one ``llm_forecasts`` run vs model / market / 0.5 on the common
    prop set, split into leak-safe (headline) and possible-leakage blocks;
    write summary / brier / reliability / coverage CSVs + one markdown."""
    loaded = load_llm_calibration_frame(db_path, run_id=run_id)
    forecasts, long = loaded["forecasts"], loaded["long"]
    common = common_prop_set(long) if not long.empty else long
    sections = {}
    for key, mask in (("headline", False), ("leak_risk", True)):
        block = common[common["leak_risk"] == mask] if not common.empty else common
        sections[key] = _llm_section(block, n_buckets)
    coverage = _llm_coverage(forecasts) if not forecasts.empty else pd.DataFrame(
        columns=COVERAGE_COLUMNS)
    meta = {
        "run_id": loaded["run_id"],
        "llm_forecasters": list(coverage["forecaster"]),
        "prompt_versions": (sorted(forecasts["prompt_version"].unique())
                            if not forecasts.empty else []),
        "props_in_run": _n_props(forecasts),
        "scored_props": _n_props(common),
        "leak_risk_props": _n_props(common[common["leak_risk"]]) if not common.empty else 0,
        "market_real": loaded["market_real"],
        "model_lookahead_props": loaded["model_lookahead_props"],
    }

    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", str(loaded["run_id"] or "none"))[:80]
    paths = {name: out_dir / f"{output_prefix}_llm_{name}_{tag}_{ts}.csv"
             for name in ("summary", "brier", "reliability", "coverage")}
    for idx, name in enumerate(("summary", "brier", "reliability")):
        blocks = [frames[idx].assign(section=key) for key, frames in sections.items()]
        cols = ["section"] + [c for c in blocks[0].columns if c != "section"]
        _concat(blocks, cols).reindex(columns=cols).to_csv(paths[name], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    md_path = out_dir / f"{output_prefix}_llm_{tag}_{ts}.md"
    md_path.write_text(_llm_markdown(meta, coverage, sections), encoding="utf-8")
    return {
        "source": "llm",
        **meta,
        "forecast_rows": int(len(forecasts)),
        **{f"{name}_csv": str(path) for name, path in paths.items()},
        "md_path": str(md_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a calibration / reliability report per stat type.")
    p.add_argument("--db", default="data/database/nba_data.db", dest="db_path")
    p.add_argument("--source", choices=["predictions", "bet_log", "llm"],
                   default="predictions")
    p.add_argument("--run-id", default=None,
                   help="llm source: the llm_forecasts run to score (default newest).")
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
        run_id=args.run_id,
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
