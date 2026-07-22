"""Paper-trading bet-slip exporter (WS10 Phase 2 — measurement, NOT execution).

Pulls the slate's model edges through Edge Scanner **full** mode (rolling μ/σ +
team-prior blend + per-stat distribution — the same math the hourly predictions
use), applies staking gates, sizes each pick with fractional Kelly (capped), and

  * writes a timestamped CSV + JSON slip under
    ``nba_model/evaluation/artifacts/``, and
  * records each pick into the ``bet_log`` table as ``status='pending'`` so it
    can later be settled + calibrated.

This is the paper-trade entry point that gates any future auto-betting. There is
no order placement here. ``--dry-run`` computes and prints the slip without
touching the DB or writing artifacts.

CLI:
    python -m nba_model.evaluation.bet_slip \\
        --db data/database/nba_data.db --books underdog prizepicks \\
        --min-edge 0.03 --min-p 0.55 --max-picks 10 --dry-run
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.logging_utils import configure_logging, get_logger
from nba_model.model import edge_scanner as es
from nba_model.model.odds import american_to_implied_prob
from nba_model.visualization.player_charts import kelly_stake

logger = get_logger(__name__)

ARTIFACT_DIR = Path("nba_model/evaluation/artifacts")

DEFAULT_MIN_EDGE = 0.02
DEFAULT_MIN_P = 0.53
DEFAULT_MAX_PICKS = 10
DEFAULT_KELLY_FRACTION = 0.5      # half-Kelly (risk-tempered default)
DEFAULT_KELLY_CAP = 0.05         # never stake > 5% of bankroll on one pick
DEFAULT_BANKROLL_UNITS = 1.0

SLIP_COLUMNS = [
    "game_date", "player_name", "player_id", "stat_type", "book", "line",
    "side", "model_prob", "implied_prob", "edge", "model_mode", "distribution",
    "kelly_fraction", "stake_units",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def build_bet_slip(
    scored_df: pd.DataFrame,
    *,
    game_date: str,
    min_edge: float = DEFAULT_MIN_EDGE,
    min_p: float = DEFAULT_MIN_P,
    max_picks: int = DEFAULT_MAX_PICKS,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    kelly_cap: float = DEFAULT_KELLY_CAP,
    bankroll_units: float = DEFAULT_BANKROLL_UNITS,
    default_american_odds: int = es.DEFAULT_AMERICAN_ODDS,
    player_id_by_name: Optional[dict] = None,
) -> pd.DataFrame:
    """Gate + size a scored (full-mode) frame into a staked bet slip.

    Gates: ``model_edge >= min_edge`` and best-side prob ``>= min_p``. Each
    surviving pick is staked with capped fractional Kelly
    (``min(kelly_fraction·Kelly, kelly_cap)``); picks with a non-positive Kelly
    stake are dropped. Ranked by edge desc (prob desc tiebreak) and truncated to
    ``max_picks``. Pure — no DB writes."""
    if scored_df is None or scored_df.empty:
        return pd.DataFrame(columns=SLIP_COLUMNS)

    df = scored_df.copy()
    # Best side prob and the staked side. (Column names avoid leading
    # underscores so itertuples keeps them — same gotcha as edge_scanner.)
    df["p_best"] = df[["p_over", "p_under"]].max(axis=1)
    df["staked_side"] = df.apply(
        lambda r: "over" if r["p_over"] >= r["p_under"] else "under", axis=1)

    df = df[df["model_edge"] >= float(min_edge)]
    df = df[df["p_best"] >= float(min_p)]
    if df.empty:
        return pd.DataFrame(columns=SLIP_COLUMNS)

    implied = float(american_to_implied_prob(int(default_american_odds)))
    id_map = {str(k).lower(): v for k, v in (player_id_by_name or {}).items()}

    picks = []
    for r in df.itertuples(index=False):
        model_prob = float(r.p_best)
        kelly_raw = kelly_stake(
            model_prob, int(default_american_odds), fraction=float(kelly_fraction))
        kelly_capped = min(float(kelly_raw), float(kelly_cap))
        if kelly_capped <= 0:
            continue
        picks.append({
            "game_date": str(game_date)[:10],
            "player_name": r.player_name,
            "player_id": id_map.get(str(r.player_name).lower()),
            "stat_type": r.stat_type,
            "book": r.book,
            "line": float(r.book_line),
            "side": r.staked_side,
            "model_prob": round(model_prob, 4),
            "implied_prob": round(implied, 4),
            "edge": round(float(r.model_edge), 4),
            "model_mode": getattr(r, "model_mode", "full"),
            "distribution": getattr(r, "distribution", None),
            "kelly_fraction": round(kelly_capped, 4),
            "stake_units": round(kelly_capped * float(bankroll_units), 4),
        })

    if not picks:
        return pd.DataFrame(columns=SLIP_COLUMNS)

    out = pd.DataFrame(picks, columns=SLIP_COLUMNS)
    out = out.sort_values(
        ["edge", "model_prob"], ascending=[False, False]
    ).reset_index(drop=True)
    if max_picks is not None and max_picks >= 0:
        out = out.head(int(max_picks))
    return out


def _resolve_player_ids(db: DatabaseManager, names: Sequence[str]) -> dict:
    """Map player_name → player_id via the active-players reference."""
    out: dict = {}
    for name in {str(n) for n in names if n}:
        row = db.conn.execute(
            "SELECT player_id FROM nba_active_players_ref "
            "WHERE lower(player_name) = lower(?)",
            (name,),
        ).fetchone()
        if row and row[0] is not None:
            try:
                out[name] = int(row[0])
            except (TypeError, ValueError):
                pass
    return out


def generate_bet_slip(
    db_path: str = es.DEFAULT_DB_PATH,
    *,
    books: Optional[Sequence[str]] = None,
    stat_types: Optional[Sequence[str]] = None,
    since_hours: float = es.DEFAULT_SINCE_HOURS,
    n_games: int = es.DEFAULT_N_GAMES,
    rolling_window: int = 10,
    game_date: Optional[str] = None,
    min_edge: float = DEFAULT_MIN_EDGE,
    min_p: float = DEFAULT_MIN_P,
    max_picks: int = DEFAULT_MAX_PICKS,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    kelly_cap: float = DEFAULT_KELLY_CAP,
    bankroll_units: float = DEFAULT_BANKROLL_UNITS,
    artifact_dir: str | Path = ARTIFACT_DIR,
    dry_run: bool = False,
) -> dict:
    """Score the slate (full mode), build the slip, and (unless ``dry_run``)
    write artifacts + insert ``bet_log`` pending rows. Returns a summary dict
    including the picks DataFrame under ``"slip"``."""
    game_date = game_date or _today_utc()

    lines = es.fetch_latest_prop_lines(
        db_path, books=books, stat_types=stat_types, since_hours=since_hours)
    scored = es.score_prop_edges(
        lines, db_path=db_path, n_games=n_games, model_mode="full",
        rolling_window=rolling_window,
    )

    with DatabaseManager(db_path=db_path) as db:
        id_map = (
            _resolve_player_ids(db, list(scored["player_name"].unique()))
            if not scored.empty else {}
        )

    slip = build_bet_slip(
        scored, game_date=game_date, min_edge=min_edge, min_p=min_p,
        max_picks=max_picks, kelly_fraction=kelly_fraction, kelly_cap=kelly_cap,
        bankroll_units=bankroll_units, player_id_by_name=id_map,
    )

    summary = {
        "game_date": game_date,
        "scored_rows": int(len(scored)),
        "picks": int(len(slip)),
        "dry_run": bool(dry_run),
        "bet_log_inserted": 0,
        "csv_path": None,
        "json_path": None,
        "slip": slip,
    }

    if dry_run or slip.empty:
        return summary

    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"bet_slip_{ts}.csv"
    json_path = out_dir / f"bet_slip_{ts}.json"
    slip.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {"generated_at_utc": _utc_now_iso(), "game_date": game_date,
             "picks": slip.to_dict(orient="records")},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )

    created = _utc_now_iso()
    rows = []
    for pick in slip.to_dict(orient="records"):
        rows.append({**pick, "created_at_utc": created, "status": "pending"})
    with DatabaseManager(db_path=db_path) as db:
        insert_result = db.insert_bet_log_rows(rows)

    summary["csv_path"] = str(csv_path)
    summary["json_path"] = str(json_path)
    summary["bet_log_inserted"] = int(insert_result.get("inserted", 0))
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export a paper-trading bet slip from full-model edges.")
    p.add_argument("--db", default=es.DEFAULT_DB_PATH, dest="db_path")
    p.add_argument("--books", nargs="*", default=None)
    p.add_argument("--stats", nargs="*", default=None, dest="stat_types")
    p.add_argument("--since-hours", type=float, default=es.DEFAULT_SINCE_HOURS)
    p.add_argument("--n-games", type=int, default=es.DEFAULT_N_GAMES)
    p.add_argument("--rolling-window", type=int, default=10)
    p.add_argument("--game-date", default=None)
    p.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE)
    p.add_argument("--min-p", type=float, default=DEFAULT_MIN_P)
    p.add_argument("--max-picks", type=int, default=DEFAULT_MAX_PICKS)
    p.add_argument("--kelly-fraction", type=float, default=DEFAULT_KELLY_FRACTION)
    p.add_argument("--kelly-cap", type=float, default=DEFAULT_KELLY_CAP)
    p.add_argument("--bankroll-units", type=float, default=DEFAULT_BANKROLL_UNITS)
    p.add_argument("--artifact-dir", default=str(ARTIFACT_DIR))
    p.add_argument("--dry-run", action="store_true",
                   help="Compute + print the slip without writing artifacts or "
                        "inserting into bet_log.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging()
    summary = generate_bet_slip(
        db_path=args.db_path, books=args.books, stat_types=args.stat_types,
        since_hours=args.since_hours, n_games=args.n_games,
        rolling_window=args.rolling_window, game_date=args.game_date,
        min_edge=args.min_edge, min_p=args.min_p, max_picks=args.max_picks,
        kelly_fraction=args.kelly_fraction, kelly_cap=args.kelly_cap,
        bankroll_units=args.bankroll_units, artifact_dir=args.artifact_dir,
        dry_run=args.dry_run,
    )
    slip = summary["slip"]
    logger.info(
        "bet slip generated",
        extra={"game_date": summary["game_date"], "picks": int(len(slip)),
               "dry_run": bool(summary["dry_run"]),
               "bet_log_inserted": int(summary.get("bet_log_inserted") or 0)},
    )
    if slip.empty:
        print(f"No picks cleared the gates for {summary['game_date']}.")
        return 0
    cols = ["player_name", "stat_type", "book", "line", "side", "model_prob",
            "edge", "distribution", "kelly_fraction", "stake_units"]
    print(slip[cols].to_string(index=False))
    if summary["dry_run"]:
        print(f"\n[dry-run] {len(slip)} picks — not written to bet_log.")
    else:
        print(f"\nWrote {summary['bet_log_inserted']} pending bet_log rows; "
              f"CSV: {summary['csv_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
