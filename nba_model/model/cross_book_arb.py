"""Cross-book layer (WS8) — line shopping, middles, and TRUE two-way arbitrage.

This sits one level above the Book Edge Scanner. The scanner ranks *model edge*
(the model's P(side) vs a book's implied price). This module compares **books
against each other** for the same prop and surfaces three distinct things — and
it is careful to keep them distinct, because conflating them is how people lose
money:

    line shopping  — best over line (the lowest) / best under line (the highest)
                     across the books quoting a prop. Always available.
    middle         — over at the low line + under at the high line. If the result
                     lands in the gap, BOTH legs win; otherwise one wins and one
                     loses. A *candidate*, never guaranteed.
    TRUE ARB       — over at book A + under at book B where the RAW implied
                     probabilities sum to < 1.0 after removing no dead-zone
                     (over_line <= under_line). This is locked profit regardless
                     of outcome. It requires REAL posted American odds on BOTH
                     legs.

Why DFS books can never be flagged as arb here: PrizePicks / Underdog / Pick6 /
ParlayPlay usually post NO American odds, so the scanner assumes -110 on each
side. A -110 / -110 pair implies 0.5238 + 0.5238 = 1.0476 > 1 — the house hold
is baked in and no locked profit exists. So with DFS-only data we emit
``line_gap`` / ``middle_candidate`` signals only; we NEVER synthesise an ``arb``
flag from assumed -110 defaults. ``detect_two_way_arb`` only ever sees rows that
carry real ``over_odds`` / ``under_odds`` (from ``betting_lines``).

Public API:
    find_cross_book_opportunities(scored_df, *, min_books=2, ...)
        -> per (player, stat) line_gap / middle-candidate table. Input is the
           ``edge_scanner.score_prop_edges`` output (one row per player/stat/book).
    fetch_two_way_lines(db_path, books=..., stat_types=..., since_hours=...)
        -> latest deduped betting_lines rows (WITH real odds) joined to players.
    detect_two_way_arb(lines_with_odds_df)
        -> only rows where a genuine cross-book two-way arb exists (raw implied
           sum < 1.0, executable direction). Never fabricates arb from -110.

CLI (see ``main``):
    python -m nba_model.model.cross_book_arb --books underdog prizepicks --min-gap 0.5
    python -m nba_model.model.cross_book_arb --arb-only          # true-arb scan
"""
from __future__ import annotations

import argparse
import math
from typing import Optional, Sequence

import pandas as pd
from scipy.stats import norm

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model import edge_scanner as es
from nba_model.model.odds import american_to_implied_prob
from nba_model.visualization import player_charts as pc

DEFAULT_DB_PATH = "data/database/nba_data.db"
DEFAULT_SINCE_HOURS = 48.0
# A gap this wide (in stat units) is treated as a realistic *middle* window —
# there is at least a full point between the low over line and the high under
# line, so a plausible integer outcome can land in the middle and win both legs.
# Below it we call the opportunity plain line shopping (``line_gap``).
DEFAULT_MIDDLE_GAP = 1.0

CROSS_BOOK_COLUMNS = [
    "player_name", "stat_type", "n_books",
    "line_min", "line_max", "line_gap",
    "best_over_book", "best_under_book",
    "consensus_mean",
    "p_over_at_line_min", "p_over_at_line_max",
    "middle_size", "opportunity_type",
    "model_mu", "model_sigma",
]

TWO_WAY_LINE_COLUMNS = [
    "player_id", "player_name", "game_date", "stat_type",
    "book", "line_value", "over_odds", "under_odds",
]

ARB_COLUMNS = [
    "player_name", "stat_type", "game_date",
    "over_book", "over_line", "over_odds",
    "under_book", "under_line", "under_odds",
    "implied_over", "implied_under", "combined_implied",
    "devig_over", "devig_under",
    "guaranteed_margin", "legs",
]

OPP_LINE_GAP = "line_gap"
OPP_MIDDLE = "middle_candidate"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _p_over_normal(mu, sigma, line) -> Optional[float]:
    """P(value > line) under N(mu, sigma). Mirrors ``player_charts.fitted_prob_over``
    and ``edge_scanner._normal_p_over`` (kept local — no private import)."""
    if mu is None or sigma is None or line is None:
        return None
    if _is_nan(mu) or _is_nan(sigma) or _is_nan(line):
        return None
    mu, sigma, line = float(mu), float(sigma), float(line)
    if sigma <= 0:
        return 1.0 if mu > line else (0.5 if mu == line else 0.0)
    return float(norm.sf(line, loc=mu, scale=sigma))


def _safe_implied_prob(odds) -> Optional[float]:
    """American odds -> implied probability, tolerant of NaN / bad values."""
    if odds is None or _is_nan(odds):
        return None
    try:
        return float(american_to_implied_prob(int(round(float(odds)))))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 1. Cross-book line shopping + middle candidates (works on DFS -110 data)
# ---------------------------------------------------------------------------

def find_cross_book_opportunities(
    scored_df: pd.DataFrame,
    *,
    min_books: int = 2,
    middle_gap_threshold: float = DEFAULT_MIDDLE_GAP,
) -> pd.DataFrame:
    """Per ``(player, stat)`` with ``>= min_books`` books, surface the line-shop
    spread and whether the gap is wide enough to be a middle candidate.

    Input is the ``edge_scanner.score_prop_edges`` output (one row per
    player/stat/book, already carrying ``model_mu`` / ``model_sigma`` fitted per
    player+stat). We do NOT refetch game logs — ``p_over_at_line_min`` /
    ``p_over_at_line_max`` are recomputed from the row's own μ/σ via ``scipy.norm``.

    ``best_over_book`` is the book at ``line_min`` (lowest line = easiest to clear
    an over); ``best_under_book`` is the book at ``line_max``. Sorted by
    ``line_gap`` descending. Single-book props are excluded. Every row is at least
    a ``line_gap`` opportunity; a gap ``>= middle_gap_threshold`` is a
    ``middle_candidate`` (never a guaranteed arb — see module docstring).
    """
    if scored_df is None or scored_df.empty:
        return pd.DataFrame(columns=CROSS_BOOK_COLUMNS)

    df = scored_df.copy()
    if "book_line" not in df.columns or "book" not in df.columns:
        return pd.DataFrame(columns=CROSS_BOOK_COLUMNS)

    df["book_line"] = pd.to_numeric(df["book_line"], errors="coerce")
    df = df.dropna(subset=["book_line"])
    if df.empty:
        return pd.DataFrame(columns=CROSS_BOOK_COLUMNS)

    rows: list[dict] = []
    for (player_name, stat_type), group in df.groupby(
        ["player_name", "stat_type"], sort=False
    ):
        if group["book"].nunique() < max(2, int(min_books)):
            continue

        # Deterministic ordering so ties on the min/max line resolve to a stable
        # book (sort by line then book name).
        g = group.sort_values(["book_line", "book"]).reset_index(drop=True)
        line_min = float(g["book_line"].iloc[0])
        line_max = float(g["book_line"].iloc[-1])
        line_gap = line_max - line_min
        best_over_book = str(g["book"].iloc[0])      # lowest line -> best for OVER
        best_under_book = str(g["book"].iloc[-1])    # highest line -> best for UNDER

        consensus_mean = float(g["book_line"].mean())

        mu = g["model_mu"].iloc[0] if "model_mu" in g.columns else None
        sigma = g["model_sigma"].iloc[0] if "model_sigma" in g.columns else None
        p_min = _p_over_normal(mu, sigma, line_min)
        p_max = _p_over_normal(mu, sigma, line_max)

        opportunity_type = (
            OPP_MIDDLE if line_gap >= float(middle_gap_threshold) else OPP_LINE_GAP
        )

        rows.append({
            "player_name": player_name,
            "stat_type": stat_type,
            "n_books": int(group["book"].nunique()),
            "line_min": round(line_min, 3),
            "line_max": round(line_max, 3),
            "line_gap": round(line_gap, 3),
            "best_over_book": best_over_book,
            "best_under_book": best_under_book,
            "consensus_mean": round(consensus_mean, 3),
            "p_over_at_line_min": round(p_min, 4) if p_min is not None else None,
            "p_over_at_line_max": round(p_max, 4) if p_max is not None else None,
            "middle_size": round(line_gap, 3),
            "opportunity_type": opportunity_type,
            "model_mu": round(float(mu), 3) if mu is not None and not _is_nan(mu) else None,
            "model_sigma": (
                round(float(sigma), 3) if sigma is not None and not _is_nan(sigma) else None
            ),
        })

    if not rows:
        return pd.DataFrame(columns=CROSS_BOOK_COLUMNS)
    out = pd.DataFrame(rows, columns=CROSS_BOOK_COLUMNS)
    return out.sort_values(
        ["line_gap", "player_name", "stat_type"], ascending=[False, True, True]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Fetch real two-way odds from betting_lines (mirrors the scanner's dedup)
# ---------------------------------------------------------------------------

def fetch_two_way_lines(
    db_path: str,
    books: Optional[Sequence[str]] = None,
    stat_types: Optional[Sequence[str]] = None,
    since_hours: Optional[float] = None,
    game_date: Optional[str] = None,
) -> pd.DataFrame:
    """Latest deduped ``betting_lines`` row per ``(player, stat, book, game_date)``
    joined to ``players`` for the display name.

    Mirrors ``fetch_latest_prop_lines`` (newest ``scraped_at`` wins). Unlike the
    DFS ``web_prop_cards`` board, ``betting_lines`` carries real ``over_odds`` /
    ``under_odds`` — this is the ONLY input from which a true arb may be flagged.
    An empty ``books`` selection returns a correctly-shaped empty frame.
    """
    if books is not None and len(books) == 0:
        return pd.DataFrame(columns=TWO_WAY_LINE_COLUMNS)

    clauses = ["bl.line_value IS NOT NULL"]
    params: list = []
    if since_hours and since_hours > 0:
        clauses.append("bl.scraped_at >= datetime('now', ?)")
        params.append(f"-{float(since_hours)} hours")
    if books:
        placeholders = ",".join("?" * len(books))
        clauses.append(f"lower(bl.book) IN ({placeholders})")
        params.extend([str(b).lower() for b in books])
    if game_date:
        clauses.append("bl.game_date = ?")
        params.append(str(game_date))
    where_sql = " AND ".join(clauses)

    query = f"""
        WITH latest AS (
            SELECT bl.player_id,
                   COALESCE(p.name, '') AS player_name,
                   bl.game_date,
                   lower(bl.stat_type) AS stat_type,
                   bl.book,
                   bl.line_value,
                   bl.over_odds,
                   bl.under_odds,
                   ROW_NUMBER() OVER (
                       PARTITION BY bl.player_id, lower(bl.stat_type),
                                    lower(bl.book), bl.game_date
                       ORDER BY bl.scraped_at DESC, bl.line_id DESC
                   ) AS rn
            FROM betting_lines bl
            LEFT JOIN players p ON p.player_id = bl.player_id
            WHERE {where_sql}
        )
        SELECT player_id, player_name, game_date, stat_type, book,
               line_value, over_odds, under_odds
        FROM latest WHERE rn = 1
        ORDER BY player_name ASC, stat_type ASC, game_date ASC, book ASC
    """
    with DatabaseManager(db_path=db_path) as db:
        df = pd.read_sql_query(query, db.conn, params=tuple(params))

    if df.empty:
        return pd.DataFrame(columns=TWO_WAY_LINE_COLUMNS)

    df["stat_type"] = df["stat_type"].map(pc._canonical_stat_type)
    if stat_types:
        wanted = {pc._canonical_stat_type(s) for s in stat_types}
        df = df[df["stat_type"].isin(wanted)].reset_index(drop=True)
        if df.empty:
            return pd.DataFrame(columns=TWO_WAY_LINE_COLUMNS)
    return df


# ---------------------------------------------------------------------------
# 3. TRUE two-way arbitrage detection (real odds required on both legs)
# ---------------------------------------------------------------------------

def detect_two_way_arb(lines_with_odds_df: pd.DataFrame) -> pd.DataFrame:
    """Flag genuine cross-book two-way arbitrage.

    Input is ``betting_lines``-shaped (``fetch_two_way_lines`` output): one row
    per player/stat/game_date/book carrying REAL American ``over_odds`` /
    ``under_odds``. For each ``(player, stat, game_date)`` and each ordered pair
    of distinct books ``(A, B)`` we consider taking the OVER at A and the UNDER
    at B and compute::

        implied_over  = implied_prob(over_odds_A)
        implied_under = implied_prob(under_odds_B)
        combined_implied = implied_over + implied_under            # RAW sum

    We flag an arb when ``combined_implied < 1.0`` **and** the direction is
    executable — ``over_line_A <= under_line_B`` — so there is no dead zone in
    which both legs lose. The flag is on the **RAW** implied sum because that is
    the executable condition (real prices, real stake split). ``guaranteed_margin
    = 1 - combined_implied`` is the locked edge. We also report the two-way
    *proportional devig* fair probabilities (``devig_over`` / ``devig_under``,
    each ``implied / combined_implied``) for reference only — the devig is NOT
    used for the flag.

    A leg with a missing / NaN price is skipped (never flagged). Both prices must
    be real, so a -110 / -110 pair (0.5238 + 0.5238 = 1.0476) is never flagged,
    and assumed-default DFS lines never reach this function as real odds.
    Returns only the flagged arb rows, sorted by ``guaranteed_margin`` desc.
    """
    if lines_with_odds_df is None or lines_with_odds_df.empty:
        return pd.DataFrame(columns=ARB_COLUMNS)

    df = lines_with_odds_df.copy()
    for col in ("player_name", "stat_type", "book", "line_value"):
        if col not in df.columns:
            return pd.DataFrame(columns=ARB_COLUMNS)
    if "game_date" not in df.columns:
        df["game_date"] = None
    for col in ("over_odds", "under_odds"):
        if col not in df.columns:
            df[col] = None

    df["stat_canon"] = df["stat_type"].map(pc._canonical_stat_type)
    df["line_value"] = pd.to_numeric(df["line_value"], errors="coerce")

    # Group per player using ``player_id`` when present — ``fetch_two_way_lines``
    # sets ``player_name = COALESCE(p.name, '')``, so two DIFFERENT players with
    # unmatched ``players`` rows would both collapse to '' and produce false
    # cross-player "arbs". player_id is the canonical key; fall back to name for
    # frames (e.g. hand-built inputs) that carry no id.
    if "player_id" in df.columns:
        df["_pkey"] = df["player_id"].where(df["player_id"].notna(), df["player_name"])
    else:
        df["_pkey"] = df["player_name"]

    group_keys = ["_pkey", "stat_canon", "game_date"]
    rows: list[dict] = []
    for _, group in df.groupby(group_keys, dropna=False, sort=False):
        books = group.reset_index(drop=True)
        if books["book"].nunique() < 2:
            continue
        recs = books.to_dict("records")
        for a in recs:
            implied_over = _safe_implied_prob(a.get("over_odds"))
            over_line = a.get("line_value")
            if implied_over is None or over_line is None or _is_nan(over_line):
                continue
            for b in recs:
                if str(a.get("book")) == str(b.get("book")):
                    continue
                implied_under = _safe_implied_prob(b.get("under_odds"))
                under_line = b.get("line_value")
                if implied_under is None or under_line is None or _is_nan(under_line):
                    continue
                # Executable direction only: no dead zone where both legs lose.
                if float(over_line) > float(under_line):
                    continue
                combined = implied_over + implied_under
                if combined >= 1.0:
                    continue
                margin = 1.0 - combined
                legs = (
                    f"OVER {float(over_line):g} @ {a.get('book')} "
                    f"({_fmt_odds(a.get('over_odds'))}) + "
                    f"UNDER {float(under_line):g} @ {b.get('book')} "
                    f"({_fmt_odds(b.get('under_odds'))})"
                )
                rows.append({
                    "player_name": a.get("player_name"),
                    "stat_type": a.get("stat_canon"),
                    "game_date": a.get("game_date"),
                    "over_book": a.get("book"),
                    "over_line": float(over_line),
                    "over_odds": int(round(float(a.get("over_odds")))),
                    "under_book": b.get("book"),
                    "under_line": float(under_line),
                    "under_odds": int(round(float(b.get("under_odds")))),
                    "implied_over": round(implied_over, 4),
                    "implied_under": round(implied_under, 4),
                    "combined_implied": round(combined, 4),
                    "devig_over": round(implied_over / combined, 4),
                    "devig_under": round(implied_under / combined, 4),
                    "guaranteed_margin": round(margin, 4),
                    "legs": legs,
                })

    if not rows:
        return pd.DataFrame(columns=ARB_COLUMNS)
    out = pd.DataFrame(rows, columns=ARB_COLUMNS)
    return out.sort_values(
        ["guaranteed_margin", "player_name", "stat_type"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _fmt_odds(odds) -> str:
    try:
        o = int(round(float(odds)))
    except (TypeError, ValueError):
        return "?"
    return f"+{o}" if o > 0 else str(o)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cross-book layer: line-gap / middle candidates from the DFS board, "
            "or TRUE two-way arbitrage from real posted odds (--arb-only)."
        ),
    )
    p.add_argument("--db", default=DEFAULT_DB_PATH, help="path to the SQLite DB")
    p.add_argument("--books", nargs="*", default=None,
                   help="book names to include (default: all)")
    p.add_argument("--stats", nargs="*", default=None,
                   help="stat types to include (default: all)")
    p.add_argument("--n-games", type=int, default=es.DEFAULT_N_GAMES)
    p.add_argument("--since-hours", type=float, default=DEFAULT_SINCE_HOURS)
    p.add_argument("--model-mode", default="chart_mean",
                   help="edge_scanner model mode (chart_mean / rolling / full)")
    p.add_argument("--rolling-window", type=int, default=10)
    p.add_argument("--min-gap", type=float, default=0.5,
                   help="minimum line_gap (stat units) to show")
    p.add_argument("--arb-only", action="store_true",
                   help="scan betting_lines for TRUE two-way arbitrage instead")
    p.add_argument("--limit", type=int, default=25)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.arb_only:
        odds_lines = fetch_two_way_lines(
            args.db, books=args.books, stat_types=args.stats,
            since_hours=args.since_hours,
        )
        if odds_lines.empty:
            print(
                "No posted two-way odds in betting_lines for that selection. "
                "True arb detection needs real sportsbook odds on both legs "
                "(DFS -110 defaults can never be arb)."
            )
            return 0
        arbs = detect_two_way_arb(odds_lines)
        if arbs.empty:
            print("No true two-way arbitrage found (no book pair with raw "
                  "implied sum < 1.0 in an executable direction).")
            return 0
        cols = ["player_name", "stat_type", "game_date", "combined_implied",
                "guaranteed_margin", "legs"]
        print(arbs[cols].head(int(args.limit)).to_string(index=False))
        return 0

    lines = es.fetch_latest_prop_lines(
        args.db, books=args.books, stat_types=args.stats,
        since_hours=args.since_hours,
    )
    if lines.empty:
        print("No prop lines found for that book/stat/lookback selection.")
        return 0
    scored = es.score_prop_edges(
        lines, db_path=args.db, n_games=args.n_games,
        model_mode=args.model_mode, rolling_window=args.rolling_window,
    )
    cross = find_cross_book_opportunities(scored)
    if not cross.empty and args.min_gap is not None:
        cross = cross[cross["line_gap"] >= float(args.min_gap)].reset_index(drop=True)
    if cross.empty:
        print(f"No cross-book opportunities with line_gap >= {args.min_gap}.")
        return 0
    cols = ["player_name", "stat_type", "n_books", "line_min", "line_max",
            "line_gap", "best_over_book", "best_under_book",
            "opportunity_type"]
    print(cross[cols].head(int(args.limit)).to_string(index=False))
    print(
        "\nNote: DFS -110 rows are line shopping / middle candidates, "
        "NOT guaranteed arbitrage. Use --arb-only for true two-way arb "
        "(requires real posted odds on both legs)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
