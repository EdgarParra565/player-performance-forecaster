"""Book Edge Scanner — rank slate-wide props by model-vs-line edge.

This is NOT cross-book arbitrage (over at book A / under at book B). It is
**model-vs-line edge**: each scraped book line is compared against the player's
fitted-normal projection (μ = last-N mean, σ = sample std), exactly the same
math the Player Charts view uses (``player_charts.fitted_prob_over``).

User story this implements:
    Underdog posts LeBron points 17.5. The model's normal has μ = 19 (recent
    mean). Because 17.5 < μ, P(over at 17.5) > 50% — a soft line for overs.
    The scanner surfaces that edge across the whole slate, ranked.

Public API:
    fetch_latest_prop_lines(db_path, books, stat_types, since_hours)
        -> deduped latest line per (player, stat, book) from web_prop_cards
    score_prop_edges(lines_df, db_path=..., n_games=..., model_mode=...)
        -> per-row μ/σ, P(over), edge vs implied, EV (μ/σ memoized per player+stat)
    top_edges(scored_df, min_p_over=..., min_edge=..., only_positive_ev=...)
        -> filtered + sorted view

DFS books (Underdog / PrizePicks / Pick6 / ParlayPlay) rarely carry American
odds, so edge is shown against the -110 breakeven (52.38%) baseline unless a
real price is present.
"""
from __future__ import annotations

import argparse
import math
from typing import Optional, Sequence

import pandas as pd
from scipy.stats import norm

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model.odds import american_to_implied_prob
from nba_model.model.probability import prob_over_distribution
from nba_model.visualization import player_charts as pc

DEFAULT_DB_PATH = "data/database/nba_data.db"
DEFAULT_SINCE_HOURS = 48.0
DEFAULT_N_GAMES = 25
DEFAULT_AMERICAN_ODDS = -110
# Need at least this many games to fit a normal (σ needs ≥2 points).
MIN_GAMES_FOR_FIT = 2

LINE_COLUMNS = [
    "book", "player_name", "stat_type", "side", "line_value", "observed_at_utc",
]

SCORED_COLUMNS = [
    "book", "player_name", "stat_type", "book_line",
    "model_mu", "model_sigma", "line_vs_mu",
    "p_over", "p_under", "best_side",
    "model_edge", "ev_best",
    "consensus_mean", "pct_from_consensus",
    "observed_hours_ago", "observed_at_utc", "n_games_used",
]

# Columns appended (never inserted/reordered) so a scored frame always carries
# the fit provenance. ``SCORED_COLUMNS`` stays the canonical prefix that the
# UI (Agent B) relies on; ``SCORED_COLUMNS_FULL`` is the actual output shape.
APPENDED_COLUMNS = ["distribution", "model_mode"]
SCORED_COLUMNS_FULL = SCORED_COLUMNS + APPENDED_COLUMNS

# Accepted model modes. ``full`` runs the prop_board stack (rolling μ/σ +
# team-prior blend + per-stat distribution); the other two keep the legacy
# plain-normal fit.
MODEL_MODES = ("chart_mean", "rolling", "full")


# ---------------------------------------------------------------------------
# 1. Pull the latest deduped prop lines from the scraped DFS board
# ---------------------------------------------------------------------------

def fetch_latest_prop_lines(
    db_path: str,
    books: Optional[Sequence[str]] = None,
    stat_types: Optional[Sequence[str]] = None,
    since_hours: float = DEFAULT_SINCE_HOURS,
) -> pd.DataFrame:
    """Latest line per ``(player, stat, book, side)`` from ``web_prop_cards``.

    Mirrors the dedup window used by ``db_manager.get_consensus_prop_lines``
    and ``player_charts._fetch_latest_book_lines``: newest ``observed_at_utc``
    wins, restricted to ``active_nba`` rows and the lookback window. An empty
    ``books`` selection returns an empty (correctly-shaped) frame.
    """
    if books is not None and len(books) == 0:
        return pd.DataFrame(columns=LINE_COLUMNS)

    clauses = ["player_classification = 'active_nba'"]
    params: list = []
    if since_hours and since_hours > 0:
        clauses.append("observed_at_utc >= datetime('now', ?)")
        params.append(f"-{float(since_hours)} hours")
    if books:
        placeholders = ",".join("?" * len(books))
        clauses.append(f"lower(book) IN ({placeholders})")
        params.extend([str(b).lower() for b in books])
    where_sql = " AND ".join(clauses)

    query = f"""
        WITH latest AS (
            SELECT book, player_name, stat_type, side, line_value, observed_at_utc,
                   ROW_NUMBER() OVER (
                       PARTITION BY lower(player_name), lower(stat_type),
                                    lower(book), lower(side)
                       ORDER BY observed_at_utc DESC, card_id DESC
                   ) AS rn
            FROM web_prop_cards
            WHERE {where_sql}
        )
        SELECT book, player_name, stat_type, side, line_value, observed_at_utc
        FROM latest WHERE rn = 1
        ORDER BY player_name ASC, stat_type ASC, book ASC, side ASC
    """
    with DatabaseManager(db_path=db_path) as db:
        df = pd.read_sql_query(query, db.conn, params=tuple(params))

    if df.empty:
        return pd.DataFrame(columns=LINE_COLUMNS)

    if stat_types:
        wanted = {pc._canonical_stat_type(s) for s in stat_types}
        canon = df["stat_type"].map(pc._canonical_stat_type)
        df = df[canon.isin(wanted)].reset_index(drop=True)
        if df.empty:
            return pd.DataFrame(columns=LINE_COLUMNS)
    return df


# ---------------------------------------------------------------------------
# 2. Fit μ/σ (memoized) and score each line
# ---------------------------------------------------------------------------

def _normal_p_over(mu: float, sigma: float, line: float) -> float:
    """P(value > line) under N(mu, sigma). Mirrors ``fitted_prob_over``."""
    if sigma <= 0:
        return 1.0 if mu > line else (0.5 if mu == line else 0.0)
    return float(1.0 - norm.cdf(line, loc=mu, scale=sigma))


def _resolve_player_id(db: DatabaseManager, name: str) -> Optional[int]:
    row = db.conn.execute(
        "SELECT player_id FROM nba_active_players_ref WHERE lower(player_name) = lower(?)",
        (name,),
    ).fetchone()
    if row and row[0] is not None:
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return None
    return None


def _resolve_player_team(db: DatabaseManager, player_id: int) -> Optional[str]:
    """Player's team abbrev from the ``players`` table (or ``None`` if unknown).

    Mirrors the ``players.team`` source the hourly recompute uses (its rows come
    from ``betting_lines JOIN players``), so full-mode team-prior lookups line
    up with the hourly path."""
    row = db.conn.execute(
        "SELECT team FROM players WHERE player_id = ?", (int(player_id),)
    ).fetchone()
    if row and row[0] and str(row[0]).strip():
        return str(row[0]).strip()
    return None


def _fit_player_stat_full(
    db: DatabaseManager,
    name_to_id: dict,
    player_name: str,
    canonical_stat: str,
    n_games: int,
    rolling_window: int,
    team_prior_map: dict,
) -> Optional[dict]:
    """Full-model fit: rolling μ/σ + team-prior blend + per-stat distribution.

    Uses the shared ``prop_board`` projection helpers off the DB directly (no
    DataLoader / network), so it produces the same μ/σ/distribution the hourly
    ``_persist_predictions`` path stores for the same inputs. Returns ``None``
    (row dropped) for unprojectable stats or insufficient history."""
    from nba_model.model import prop_board

    if canonical_stat not in prop_board.PROJECTABLE_STATS:
        return None
    key = player_name.lower()
    if key not in name_to_id:
        name_to_id[key] = _resolve_player_id(db, player_name)
    pid = name_to_id[key]
    if pid is None:
        return None
    games = db.get_player_games(pid, n_games=max(1, int(n_games)))
    latest = prop_board.build_history_from_games(
        games, rolling_window=int(rolling_window))
    if latest is None:
        return None

    prior_inputs = None
    team = _resolve_player_team(db, pid)
    if team:
        prior_inputs = team_prior_map.get(team.upper())

    moments = prop_board.project_stat_moments(
        latest, canonical_stat, prior_inputs=prior_inputs)
    mu = float(moments["mu"])
    sigma = float(moments["sigma"])
    # A NULL stat value inside the rolling window used to make μ/σ NaN; the root
    # fix (prop_board.build_history_from_games) now recomputes μ/σ over the
    # surviving games, so this guard normally only fires when too few valid
    # games remained (moment still NaN) or for a degenerate blend. Drop rather
    # than persist a nonsensical NaN row that would otherwise sort to the top of
    # the edge ranking.
    if not (math.isfinite(mu) and math.isfinite(sigma)):
        return None
    return {
        "mu": mu,
        "sigma": sigma,
        "distribution": moments["distribution"],
        # The newest ``rolling_window`` games informed the projection.
        "n": int(rolling_window),
    }


def _fit_player_stat(
    db: DatabaseManager,
    name_to_id: dict,
    player_name: str,
    canonical_stat: str,
    n_games: int,
    model_mode: str,
    rolling_window: int,
    db_path: str,
    team_prior_map: Optional[dict] = None,
) -> Optional[dict]:
    """Return ``{"mu","sigma","n"[,"distribution"]}`` for a player+stat, or None
    when un-fittable."""
    if model_mode == "full":
        # Mirror the 'rolling' branch's swallow-and-drop: a single un-fittable
        # player (e.g. rolling_window < 2 → add_rolling_stats raises) must drop
        # that row, never abort the whole slate scan.
        try:
            return _fit_player_stat_full(
                db, name_to_id, player_name, canonical_stat,
                n_games, rolling_window, team_prior_map or {},
            )
        except Exception:
            return None

    if model_mode == "rolling":
        try:
            from nba_model.model import prop_board
            latest = prop_board._build_player_history(
                player_name, n_games=n_games,
                rolling_window=rolling_window, db_path=db_path,
            )
            mu, sigma = prop_board._project_stat_from_history(latest, canonical_stat)
            return {"mu": float(mu), "sigma": float(sigma), "n": int(n_games)}
        except Exception:
            return None

    # Default: chart-consistent last-N mean/std (same as Player Charts).
    key = player_name.lower()
    if key not in name_to_id:
        name_to_id[key] = _resolve_player_id(db, player_name)
    pid = name_to_id[key]
    if pid is None:
        return None
    games = db.get_player_games(pid, n_games=max(1, int(n_games)))
    if games is None or games.empty:
        return None
    games = games.sort_values("game_date").reset_index(drop=True)
    values = pc._series_for_stat(games, canonical_stat)
    if values.size < MIN_GAMES_FOR_FIT:
        return None
    data = pc.PlayerChartData(
        player_id=int(pid), player_name=player_name,
        stat_type=canonical_stat, games=games, values=values,
    )
    return {"mu": float(data.mu), "sigma": float(data.sigma), "n": int(values.size)}


def _hours_ago(observed_at_utc) -> Optional[float]:
    ts = pd.to_datetime(observed_at_utc, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    delta = pd.Timestamp.now(tz="UTC") - ts
    return float(delta.total_seconds() / 3600.0)


def score_prop_edges(
    lines_df: pd.DataFrame,
    *,
    db_path: str,
    n_games: int = DEFAULT_N_GAMES,
    model_mode: str = "chart_mean",
    rolling_window: int = 10,
    default_american_odds: int = DEFAULT_AMERICAN_ODDS,
) -> pd.DataFrame:
    """Score each book line against the fitted model. One row per (book,
    player, stat). μ/σ are memoized per (player, stat) so a 200-prop slate
    triggers one game-log fetch per unique player+stat, not per row.

    ``model_mode``:
      * ``chart_mean`` (default) — plain normal on the last-N mean/std.
      * ``rolling`` — rolling μ/σ from the prop_board stack (DataLoader-backed).
      * ``full`` — the full prop_board projection: rolling μ/σ + team-prior
        blend (when the player's team has a prior) + the per-stat default
        distribution via ``get_default_distribution``. Matches the numbers the
        hourly ``predictions`` recompute stores.

    Output keeps ``SCORED_COLUMNS`` as its ordered prefix and appends
    ``distribution`` + ``model_mode`` (see ``SCORED_COLUMNS_FULL``)."""
    if lines_df is None or lines_df.empty:
        return pd.DataFrame(columns=SCORED_COLUMNS_FULL)

    work = lines_df.copy()
    work["canonical_stat"] = work["stat_type"].map(pc._canonical_stat_type)
    work["player_key"] = work["player_name"].str.lower()
    work["book_key"] = work["book"].str.lower()

    # Collapse over/under sides into one line per (player, stat, book).
    # (Column names avoid leading underscores so itertuples keeps them.)
    per_book = (
        work.groupby(["player_key", "canonical_stat", "book_key"], as_index=False)
        .agg(
            player_name=("player_name", "first"),
            book=("book", "first"),
            book_line=("line_value", "median"),
            observed_at_utc=("observed_at_utc", "max"),
        )
    )

    # Cross-book consensus mean per (player, stat).
    consensus = (
        per_book.groupby(["player_key", "canonical_stat"])["book_line"]
        .mean()
        .to_dict()
    )

    implied_default = american_to_implied_prob(int(default_american_odds))

    memo: dict = {}
    name_to_id: dict = {}
    rows: list[dict] = []
    with DatabaseManager(db_path=db_path) as db:
        # Full mode pulls the whole slate's team priors once (pace + implied
        # team total per team) — the same map the hourly recompute blends.
        team_prior_map = (
            db.get_team_prior_inputs_map() if model_mode == "full" else {}
        )
        for r in per_book.itertuples(index=False):
            canonical_stat = r.canonical_stat
            fit_key = (r.player_key, canonical_stat)
            if fit_key not in memo:
                memo[fit_key] = _fit_player_stat(
                    db, name_to_id, r.player_name, canonical_stat,
                    n_games, model_mode, rolling_window, db_path,
                    team_prior_map=team_prior_map,
                )
            fit = memo[fit_key]
            if fit is None:
                continue

            mu, sigma = fit["mu"], fit["sigma"]
            distribution = fit.get("distribution", "normal")
            line = float(r.book_line)
            if model_mode == "full":
                # Per-stat distribution (rebounds → poisson, etc.), not a
                # hardcoded normal.
                p_over = float(prob_over_distribution(
                    line=line, mu=mu, sigma=sigma,
                    distribution=distribution, sample_size=int(rolling_window),
                ))
            else:
                p_over = _normal_p_over(mu, sigma, line)
            p_under = 1.0 - p_over
            best_side = "over" if p_over >= p_under else "under"
            p_best = max(p_over, p_under)
            # DFS cards carry no American price → both sides priced at -110.
            model_edge = p_best - implied_default
            ev_best = pc.expected_value(p_best, default_american_odds)

            cmean = consensus.get((r.player_key, canonical_stat))
            pct_from_consensus = (
                ((line - cmean) / cmean * 100.0)
                if cmean not in (None, 0) else None
            )

            rows.append({
                "book": r.book,
                "player_name": r.player_name,
                "stat_type": canonical_stat,
                "book_line": line,
                "model_mu": round(mu, 3),
                "model_sigma": round(sigma, 3),
                "line_vs_mu": round(line - mu, 3),
                "p_over": round(p_over, 4),
                "p_under": round(p_under, 4),
                "best_side": best_side,
                "model_edge": round(model_edge, 4),
                "ev_best": round(ev_best, 4) if ev_best is not None else None,
                "consensus_mean": round(cmean, 3) if cmean is not None else None,
                "pct_from_consensus": (
                    round(pct_from_consensus, 2)
                    if pct_from_consensus is not None else None
                ),
                "observed_hours_ago": (
                    round(_hours_ago(r.observed_at_utc), 2)
                    if _hours_ago(r.observed_at_utc) is not None else None
                ),
                "observed_at_utc": r.observed_at_utc,
                "n_games_used": fit["n"],
                "distribution": distribution,
                "model_mode": model_mode,
            })

    if not rows:
        return pd.DataFrame(columns=SCORED_COLUMNS_FULL)
    return pd.DataFrame(rows, columns=SCORED_COLUMNS_FULL)


# ---------------------------------------------------------------------------
# 3. Filter + rank
# ---------------------------------------------------------------------------

def top_edges(
    scored_df: pd.DataFrame,
    *,
    min_p_over: Optional[float] = None,
    min_edge: Optional[float] = None,
    only_positive_ev: bool = False,
    limit: int = 200,
) -> pd.DataFrame:
    """Filter and rank scored props. Default sort: ``model_edge`` desc, then
    ``p_over`` desc as a tiebreaker."""
    if scored_df is None or scored_df.empty:
        return pd.DataFrame(columns=SCORED_COLUMNS_FULL)

    df = scored_df.copy()
    if min_p_over is not None:
        df = df[df["p_over"] >= float(min_p_over)]
    if min_edge is not None:
        df = df[df["model_edge"] >= float(min_edge)]
    if only_positive_ev:
        df = df[df["ev_best"].fillna(-1) > 0]

    df = df.sort_values(
        ["model_edge", "p_over"], ascending=[False, False]
    ).reset_index(drop=True)
    if limit is not None and limit >= 0:
        df = df.head(int(limit))
    return df


# ---------------------------------------------------------------------------
# CLI helper (test without Streamlit)
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rank slate-wide props by model-vs-line edge.",
    )
    p.add_argument("--db", default=DEFAULT_DB_PATH, help="path to the SQLite DB")
    p.add_argument("--books", nargs="*", default=None,
                   help="book names to include (default: all)")
    p.add_argument("--stats", nargs="*", default=None,
                   help="stat types to include (default: all)")
    p.add_argument("--n-games", type=int, default=DEFAULT_N_GAMES)
    p.add_argument("--since-hours", type=float, default=DEFAULT_SINCE_HOURS)
    p.add_argument("--model-mode", choices=list(MODEL_MODES),
                   default="chart_mean")
    p.add_argument("--rolling-window", type=int, default=10)
    p.add_argument("--min-edge", type=float, default=None)
    p.add_argument("--min-p-over", type=float, default=None)
    p.add_argument("--only-positive-ev", action="store_true")
    p.add_argument("--limit", type=int, default=20)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    lines = fetch_latest_prop_lines(
        args.db, books=args.books, stat_types=args.stats,
        since_hours=args.since_hours,
    )
    if lines.empty:
        print("No prop lines found for that book/stat/lookback selection.")
        return 0
    scored = score_prop_edges(
        lines, db_path=args.db, n_games=args.n_games,
        model_mode=args.model_mode, rolling_window=args.rolling_window,
    )
    top = top_edges(
        scored, min_p_over=args.min_p_over, min_edge=args.min_edge,
        only_positive_ev=args.only_positive_ev, limit=args.limit,
    )
    if top.empty:
        print("No props cleared the edge/EV filters.")
        return 0
    cols = ["book", "player_name", "stat_type", "book_line", "model_mu",
            "line_vs_mu", "p_over", "best_side", "model_edge", "ev_best"]
    print(top[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
