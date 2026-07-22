"""Read-only service layer: thin wrappers over the existing data/model code.

Every function here calls the SAME functions the Streamlit app uses — the
``player_charts`` fetchers, ``edge_scanner`` scoring, and ``DatabaseManager``
consensus queries — and returns JSON-serializable native Python. No writes, no
duplicated model logic. Where the frontend needs an aggregate the data layer
doesn't already expose (KPI counts, freshest-scrape timestamp, player search),
a tiny read-only SQL helper is added HERE rather than in ``nba_model``.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model import edge_scanner as es
from nba_model.visualization import player_charts as pc

# Stats the chart layer can actually series/fit (mirrors player_charts).
CHARTABLE_STATS = (
    "points", "assists", "rebounds", "pra", "ra",
    "three_pointers_made", "field_goals_made", "minutes",
)


# ---------------------------------------------------------------------------
# Small coercion helpers — SQLite/pandas/numpy -> native, NaN -> None
# ---------------------------------------------------------------------------

def _num(x) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _int(x) -> Optional[int]:
    f = _num(x)
    return int(round(f)) if f is not None else None


def _str(x) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, float) and not math.isfinite(x):
        return None
    s = str(x).strip()
    return s or None


# ---------------------------------------------------------------------------
# Meta / slate-level aggregates (read-only SQL)
# ---------------------------------------------------------------------------

def _scalar(db: DatabaseManager, sql: str, params: tuple = ()):
    row = db.conn.execute(sql, params).fetchone()
    return row[0] if row else None


def _distinct_books(db: DatabaseManager) -> list[str]:
    """Books that have EVER produced a parsed line (DFS cards + real odds)."""
    books: set[str] = set()
    for sql in (
        "SELECT DISTINCT lower(book) FROM web_prop_cards WHERE book IS NOT NULL",
        "SELECT DISTINCT lower(book) FROM betting_lines WHERE book IS NOT NULL",
    ):
        books.update(r[0] for r in db.conn.execute(sql).fetchall() if r[0])
    return sorted(books)


def _freshest_scrape(db: DatabaseManager) -> Optional[str]:
    """Newest scrape timestamp across every live-line source."""
    candidates = [
        _scalar(db, "SELECT MAX(observed_at_utc) FROM web_prop_cards"),
        _scalar(db, "SELECT MAX(observed_at_utc) FROM web_team_lines"),
        _scalar(db, "SELECT MAX(scraped_at) FROM betting_lines"),
    ]
    vals = [str(c) for c in candidates if c]
    return max(vals) if vals else None


def slate_kpis(db_path: str) -> dict:
    with DatabaseManager(db_path=db_path) as db:
        games = _int(_scalar(db, "SELECT COUNT(DISTINCT game_id) FROM games")) or 0
        players = _int(_scalar(
            db, "SELECT COUNT(DISTINCT player_id) FROM game_logs")) or 0
        books = len(_distinct_books(db))
        freshest = _freshest_scrape(db)
        last_game = _str(_scalar(db, "SELECT MAX(game_date) FROM games"))
        prop_recent = _int(_scalar(
            db,
            "SELECT COUNT(*) FROM web_prop_cards "
            "WHERE observed_at_utc >= datetime('now', '-48 hours')",
        )) or 0
    return {
        "games_in_db": games,
        "players_tracked": players,
        "books_producing": books,
        "freshest_scrape_utc": freshest,
        "last_game_date": last_game,
        "prop_lines_recent": prop_recent,
        "edges_positive_ev": None,
    }


def health(db_path: str, exists: bool) -> dict:
    table_counts: dict[str, int] = {}
    last_game = None
    freshest = None
    if exists:
        with DatabaseManager(db_path=db_path) as db:
            for table in ("games", "game_logs", "web_prop_cards",
                          "betting_lines", "predictions"):
                table_counts[table] = _int(
                    _scalar(db, f"SELECT COUNT(*) FROM {table}")) or 0
            last_game = _str(_scalar(db, "SELECT MAX(game_date) FROM games"))
            freshest = _freshest_scrape(db)
    return {
        "last_game_date": last_game,
        "freshest_scrape_utc": freshest,
        "table_counts": table_counts,
    }


def meta(db_path: str) -> dict:
    with DatabaseManager(db_path=db_path) as db:
        books = _distinct_books(db)
    return {
        "stats": list(CHARTABLE_STATS),
        "teams": pc.list_team_codes(db_path),
        "seasons": pc.list_seasons(db_path),
        "books": books,
    }


# ---------------------------------------------------------------------------
# Recent games strip
# ---------------------------------------------------------------------------

def recent_games(
    db_path: str,
    n: int = 12,
    season: Optional[str] = None,
    season_type: Optional[str] = None,
    team: Optional[str] = None,
) -> dict:
    df = pc.fetch_recent_games(
        db_path, n=n, season=season, season_type=season_type, team_abbrev=team,
    )
    rows: list[dict] = []
    if df is not None and not df.empty:
        for r in df.to_dict("records"):
            rows.append({
                "game_id": _str(r.get("game_id")) or "",
                "game_date": _str(r.get("game_date")),
                "season": _str(r.get("season")),
                "season_type": _str(r.get("season_type")),
                "away_abbrev": _str(r.get("away_abbrev")),
                "away_name": _str(r.get("away_name")),
                "away_pts": _num(r.get("away_pts")),
                "home_abbrev": _str(r.get("home_abbrev")),
                "home_name": _str(r.get("home_name")),
                "home_pts": _num(r.get("home_pts")),
                "matchup": _str(r.get("matchup")),
                "winner": _str(r.get("winner")),
            })
    return {"rows": rows, "count": len(rows)}


# ---------------------------------------------------------------------------
# Edge scanner (dashboard "top edges" + full Edge Scanner view)
# ---------------------------------------------------------------------------

def _edge_rows(df: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    if df is None or df.empty:
        return out
    for r in df.to_dict("records"):
        out.append({
            "book": _str(r.get("book")) or "",
            "player_name": _str(r.get("player_name")) or "",
            "stat_type": _str(r.get("stat_type")) or "",
            "book_line": _num(r.get("book_line")),
            "model_mu": _num(r.get("model_mu")),
            "model_sigma": _num(r.get("model_sigma")),
            "line_vs_mu": _num(r.get("line_vs_mu")),
            "p_over": _num(r.get("p_over")),
            "p_under": _num(r.get("p_under")),
            "best_side": _str(r.get("best_side")),
            "model_edge": _num(r.get("model_edge")),
            "ev_best": _num(r.get("ev_best")),
            "consensus_mean": _num(r.get("consensus_mean")),
            "pct_from_consensus": _num(r.get("pct_from_consensus")),
            "observed_hours_ago": _num(r.get("observed_hours_ago")),
            "observed_at_utc": _str(r.get("observed_at_utc")),
            "n_games_used": _int(r.get("n_games_used")),
            "distribution": _str(r.get("distribution")),
            "model_mode": _str(r.get("model_mode")),
        })
    return out


def scan_edges(
    db_path: str,
    *,
    books: Optional[Sequence[str]] = None,
    stats: Optional[Sequence[str]] = None,
    since_hours: float = 48.0,
    n_games: int = 25,
    model_mode: str = "chart_mean",
    rolling_window: int = 10,
    min_edge: Optional[float] = None,
    min_p_over: Optional[float] = None,
    only_positive_ev: bool = False,
    limit: int = 100,
) -> dict:
    lines = es.fetch_latest_prop_lines(
        db_path, books=books, stat_types=stats, since_hours=since_hours,
    )
    n_lines = 0 if lines is None or lines.empty else int(len(lines))

    scored = es.score_prop_edges(
        lines, db_path=db_path, n_games=n_games,
        model_mode=model_mode, rolling_window=rolling_window,
    )
    n_scored = 0 if scored is None or scored.empty else int(len(scored))

    top = es.top_edges(
        scored, min_p_over=min_p_over, min_edge=min_edge,
        only_positive_ev=only_positive_ev, limit=limit,
    )
    rows = _edge_rows(top)

    with DatabaseManager(db_path=db_path) as db:
        books_available = _distinct_books(db)
    return {
        "rows": rows,
        "n_lines": n_lines,
        "n_scored": n_scored,
        "n_returned": len(rows),
        "model_mode": model_mode,
        "books_available": books_available,
        "stats_available": list(CHARTABLE_STATS),
    }


# ---------------------------------------------------------------------------
# Player search (server-side over game_logs / players via the shared fetcher)
# ---------------------------------------------------------------------------

def search_players(
    db_path: str,
    q: str = "",
    team: Optional[str] = None,
    only_with_lines: bool = False,
    limit: int = 30,
) -> dict:
    df = pc.list_players_with_data(db_path, team=team or None)
    rows: list[dict] = []
    if df is not None and not df.empty:
        query = (q or "").strip().lower()
        if query:
            df = df[df["player_name"].str.lower().str.contains(query, na=False)]
        if only_with_lines and "n_books" in df.columns:
            df = df[df["n_books"] > 0]
        df = df.head(int(max(1, limit)))
        for r in df.to_dict("records"):
            rows.append({
                "player_id": _int(r.get("player_id")) or 0,
                "player_name": _str(r.get("player_name")) or "",
                "team": _str(r.get("team")),
                "n_books": _int(r.get("n_books")) or 0,
            })
    return {"rows": rows, "count": len(rows)}


# ---------------------------------------------------------------------------
# Player detail (flagship view) — series, distribution, per-book table
# ---------------------------------------------------------------------------

def _series(data: pc.PlayerChartData, rolling_window: int) -> list[dict]:
    games = data.games
    values = data.values
    if games is None or games.empty or values.size == 0:
        return []
    roll = (
        pd.Series(values)
        .rolling(window=max(1, rolling_window), min_periods=1)
        .mean()
        .to_numpy()
    )
    records = games.to_dict("records")
    out: list[dict] = []
    for i, val in enumerate(values):
        row = records[i] if i < len(records) else {}
        out.append({
            "game_date": _str(row.get("game_date")),
            "value": float(val),
            "rolling_mean": _num(roll[i]),
            "opponent": _str(pc._extract_opponent(row.get("matchup"))),
            "home_away": _str(row.get("home_away")),
            "result": _str(row.get("result")),
        })
    return out


def _histogram(values: np.ndarray) -> list[dict]:
    if values.size == 0:
        return []
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        hi = lo + 1.0
    nbins = int(min(max(8, round(math.sqrt(values.size))), 20))
    counts, edges = np.histogram(values, bins=nbins, range=(lo, hi))
    return [
        {"x0": float(edges[i]), "x1": float(edges[i + 1]), "count": int(counts[i])}
        for i in range(len(counts))
    ]


def _fitted_normal(values: np.ndarray, mu: float, sigma: float) -> list[dict]:
    """Normal density scaled to the histogram's expected counts."""
    if values.size < 2 or sigma <= 0:
        return []
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return []
    nbins = int(min(max(8, round(math.sqrt(values.size))), 20))
    bin_width = (hi - lo) / nbins
    xs = np.linspace(lo, hi, 60)
    ys = norm.pdf(xs, loc=mu, scale=sigma) * values.size * bin_width
    return [{"x": float(x), "y": float(y)} for x, y in zip(xs, ys)]


def _book_lines(data: pc.PlayerChartData) -> tuple[list[dict], int]:
    df = data.book_lines
    rows: list[dict] = []
    positive_ev = 0
    if df is None or df.empty:
        return rows, positive_ev
    for r in df.to_dict("records"):
        line = _num(r.get("line_value"))
        if line is None:
            continue
        over_odds = _int(r.get("over_odds"))
        under_odds = _int(r.get("under_odds"))
        p_over = pc.fitted_prob_over(data, line)
        p_under = (1.0 - p_over) if p_over is not None else None
        best_side = None
        model_edge = None
        breakeven = None
        if p_over is not None:
            best_side = "over" if p_over >= p_under else "under"
            best_odds = over_odds if best_side == "over" else under_odds
            breakeven = pc._american_odds_breakeven(
                best_odds if best_odds is not None else es.DEFAULT_AMERICAN_ODDS)
            if breakeven is not None:
                model_edge = max(p_over, p_under) - breakeven
        ev_over = pc.expected_value(
            p_over, over_odds if over_odds is not None else es.DEFAULT_AMERICAN_ODDS,
        ) if p_over is not None else None
        ev_under = pc.expected_value(
            p_under, under_odds if under_odds is not None else es.DEFAULT_AMERICAN_ODDS,
        ) if p_under is not None else None
        hit_rate = (
            float(np.mean(data.values > line)) if data.values.size else None
        )
        if (ev_over is not None and ev_over > 0) or (
                ev_under is not None and ev_under > 0):
            positive_ev += 1
        rows.append({
            "book": _str(r.get("book")) or "",
            "line": line,
            "over_odds": over_odds,
            "under_odds": under_odds,
            "p_over": _num(p_over),
            "p_under": _num(p_under),
            "best_side": best_side,
            "model_edge": _num(model_edge),
            "ev_over": _num(ev_over),
            "ev_under": _num(ev_under),
            "hit_rate": _num(hit_rate),
            "breakeven": _num(breakeven),
            "is_dfs": over_odds is None and under_odds is None,
        })
    rows.sort(key=lambda x: (x["model_edge"] is None, -(x["model_edge"] or 0)))
    return rows, positive_ev


def resolve_player_name(db_path: str, player_id: int) -> Optional[str]:
    """Canonical player name for an id (active-players ref, then players)."""
    with DatabaseManager(db_path=db_path) as db:
        name = _scalar(
            db,
            "SELECT player_name FROM nba_active_players_ref WHERE player_id = ?",
            (int(player_id),),
        )
        if not name:
            name = _scalar(
                db, "SELECT name FROM players WHERE player_id = ?",
                (int(player_id),),
            )
    return _str(name)


def player_detail(
    db_path: str,
    player_id: int,
    player_name: str,
    stat_type: str,
    n_games: int = 25,
    rolling_window: int = 5,
) -> dict:
    data = pc.fetch_player_chart_data(
        db_path, player_id, player_name, stat_type, n_games=n_games,
    )
    values = data.values
    mu = data.mu
    sigma = data.sigma
    book_rows, positive_ev = _book_lines(data)
    staleness = pc.book_lines_staleness_summary(data)
    return {
        "player_id": int(player_id),
        "player_name": data.player_name,
        "stat_type": data.stat_type,
        "n_games": int(values.size),
        "rolling_window": int(rolling_window),
        "kpis": {
            "n_games": int(values.size),
            "mu": _num(mu),
            "sigma": _num(sigma),
            "market_consensus_line": _num(data.market_consensus_line),
            "n_books": len(book_rows),
            "positive_ev_sides": positive_ev,
        },
        "series": _series(data, rolling_window),
        "histogram": _histogram(values),
        "fitted": _fitted_normal(values, mu, sigma),
        "distribution": "normal",
        "book_lines": book_rows,
        "notes": list(data.notes or []),
        "last_line_scraped_utc": (
            staleness.get("latest_iso") if staleness else None
        ),
    }
