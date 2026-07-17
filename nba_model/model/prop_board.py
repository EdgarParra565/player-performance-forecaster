"""Prop board generator for all player props in a game.

Given a game date (and optional team filters), this script:
  - Reads available player props from the betting_lines table.
  - Builds per-player rolling stat summaries from recent game logs.
  - Uses production default distributions by stat type to compute:
      - projected mean (mu) and standard deviation (sigma)
      - probability of the OVER for each line
      - EV for over/under based on American odds.

Run examples (shell-style):

  python -m nba_model.model.prop_board \\
    --game-date 2025-03-01 \\
    --home-team LAL --away-team BOS \\
    --stat-types points assists rebounds pra
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd

from nba_model.data.data_loader import DataLoader
from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model.feature_engineering import add_rolling_stats
from nba_model.model.odds import american_to_implied_prob, expected_value
from nba_model.model.probability import prob_over_distribution
from nba_model.model.simulation import blend_team_prior, get_default_distribution


DEFAULT_DB_PATH = "data/database/nba_data.db"
DEFAULT_STAT_TYPES = ["points", "assists", "rebounds", "pra"]
DEFAULT_ROLLING_WINDOW = 10
DEFAULT_N_GAMES = 120
DEFAULT_TEAM_PRIOR_ALPHA = 0.3

# Stats the rolling-history projection can model (``add_rolling_stats`` only
# emits rolling means for these). For anything else ``_project_stat_from_history``
# falls back to a degenerate mu=0, so callers that need a trustworthy full-model
# projection (Edge Scanner "full" mode) should treat other stats as unfittable.
PROJECTABLE_STATS = ("points", "assists", "rebounds", "pra")


@dataclass
class BoardLine:
    game_date: str
    team: Optional[str]
    player_name: str
    stat_type: str
    line_value: float
    over_odds: Optional[int]
    under_odds: Optional[int]
    mu: float
    sigma: float
    distribution: str
    prob_over: float
    implied_over_prob: Optional[float]
    implied_under_prob: Optional[float]
    ev_over: Optional[float]
    ev_under: Optional[float]


def _normalize_stat_type(value: str) -> str:
    return str(value or "").strip().lower()


def _fetch_betting_lines_for_game(
    db_path: str,
    game_date: str,
    stat_types: Sequence[str],
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
    books: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Fetch betting lines joined with player metadata for a single date."""
    stat_types = sorted({_normalize_stat_type(s) for s in stat_types if s})
    if not stat_types:
        raise ValueError("At least one stat_type must be provided")

    with DatabaseManager(db_path=db_path) as db:
        # NOTE: sqlite3 cannot expand a tuple bound to a single named/qmark
        # placeholder, so multi-value IN clauses are built with one ``?`` per
        # value and fed via a positional params list (same pattern as
        # edge_scanner.fetch_latest_prop_lines).
        stat_placeholders = ",".join("?" * len(stat_types))
        clauses: list[str] = [
            "bl.game_date = ?",
            f"lower(bl.stat_type) IN ({stat_placeholders})",
        ]
        params: list = [str(game_date)[:10], *stat_types]

        if home_team and away_team:
            clauses.append("p.team IN (?, ?)")
            params.extend([home_team, away_team])
        elif home_team or away_team:
            clauses.append("p.team = ?")
            params.append(home_team or away_team)

        if books:
            books_norm = [str(b).strip() for b in books if str(b).strip()]
            if books_norm:
                book_placeholders = ",".join("?" * len(books_norm))
                clauses.append(f"bl.book IN ({book_placeholders})")
                params.extend(books_norm)

        where_sql = " AND ".join(clauses)
        sql = f"""
            SELECT
                bl.game_date,
                bl.player_id,
                p.name as player_name,
                p.team,
                bl.book,
                bl.stat_type,
                bl.line_value,
                bl.over_odds,
                bl.under_odds
            FROM betting_lines bl
            JOIN players p
              ON p.player_id = bl.player_id
            WHERE {where_sql}
            ORDER BY p.team, p.name, bl.stat_type, bl.book, bl.line_value
        """

        df = pd.read_sql_query(sql, db.conn, params=params)

    if df.empty:
        return df

    df["stat_type"] = df["stat_type"].astype(str).str.strip().str.lower()
    return df


def _build_player_history(
    player_name: str,
    n_games: int = DEFAULT_N_GAMES,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    db_path: str = DEFAULT_DB_PATH,
) -> pd.Series:
    """Return latest rolling stat row for a player (DataLoader-backed path)."""
    loader = DataLoader(db_path=db_path)
    df = loader.load_player_data(player_name=player_name, n_games=n_games)
    latest = build_history_from_games(df, rolling_window=rolling_window)
    if latest is None:
        raise ValueError(f"Insufficient data for {player_name}; no rolling window rows available.")
    return latest


def build_history_from_games(
    games: pd.DataFrame,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> Optional[pd.Series]:
    """Latest rolling-stat row from a raw ``game_logs`` frame, or ``None``.

    This is the tail of :func:`_build_player_history` factored out so the
    DataLoader-backed CLI/hourly path and the DB-direct Edge Scanner "full"
    path compute μ/σ from *identical* code. ``add_rolling_stats`` re-sorts by
    date internally, so the input order (DESC from ``get_player_games`` or ASC)
    yields the same newest-``window`` projection. Returns ``None`` when there
    aren't enough games to fill the rolling window (mirrors the ``ValueError``
    the CLI path raises)."""
    if games is None or getattr(games, "empty", True):
        return None
    enriched = add_rolling_stats(games, window=rolling_window)
    latest = enriched.dropna(subset=["rolling_mean_minutes"])
    if latest.empty:
        return None
    return latest.iloc[-1]


def project_stat_moments(
    latest: pd.Series,
    stat_type: str,
    *,
    prior_inputs: Optional[dict] = None,
    team_prior_alpha: float = DEFAULT_TEAM_PRIOR_ALPHA,
) -> dict:
    """Line-independent ``{mu, sigma, distribution}`` for one player+stat.

    Single source of truth for the full-model projection: rolling μ/σ →
    optional ``blend_team_prior`` (pace + implied team total) → per-stat default
    distribution. Shared by the prop board, the hourly recompute, and Edge
    Scanner "full" mode so the three paths can't drift. ``prior_inputs`` is the
    dict returned by ``db.get_team_prior_inputs`` /
    ``db.get_team_prior_inputs_map`` (or ``None`` to skip the blend)."""
    stat_key = _normalize_stat_type(stat_type)
    mu, sigma = _project_stat_from_history(latest, stat_type=stat_key)
    if prior_inputs:
        mu, sigma = blend_team_prior(
            mu, sigma,
            pace_factor=prior_inputs.get("pace_factor"),
            implied_team_total=prior_inputs.get("implied_team_total"),
            team_recent_avg_total=prior_inputs.get("team_recent_avg_total"),
            alpha=float(team_prior_alpha),
        )
    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "distribution": get_default_distribution(stat_key),
    }


def project_prop_line(
    latest: pd.Series,
    stat_type: str,
    line_value: float,
    *,
    prior_inputs: Optional[dict] = None,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    team_prior_alpha: float = DEFAULT_TEAM_PRIOR_ALPHA,
) -> dict:
    """``project_stat_moments`` plus the over-probability at ``line_value``.

    The distribution's ``sample_size`` is ``rolling_window`` (matches the prop
    board's original ``prob_over_distribution`` call)."""
    moments = project_stat_moments(
        latest, stat_type,
        prior_inputs=prior_inputs, team_prior_alpha=team_prior_alpha,
    )
    prob_over = prob_over_distribution(
        line=float(line_value),
        mu=moments["mu"],
        sigma=moments["sigma"],
        distribution=moments["distribution"],
        sample_size=int(rolling_window),
    )
    return {**moments, "prob_over": float(prob_over)}


def _project_stat_from_history(latest: pd.Series, stat_type: str) -> tuple[float, float]:
    """Compute (mu, sigma) for a stat from rolling history."""
    stat_key = _normalize_stat_type(stat_type)

    if stat_key == "points":
        mu = float(latest.get("rolling_mean_points", latest.get("points", 0.0)))
        sigma = float(latest.get("rolling_std_points", 0.0))
        return mu, max(sigma, 1e-3)

    col_mean = f"rolling_mean_{stat_key}"
    col_std = f"rolling_std_{stat_key}"
    mu = float(latest.get(col_mean, latest.get(stat_key, 0.0)))
    sigma = float(latest.get(col_std, 0.0))
    return mu, max(sigma, 1e-3)


def _build_board_lines(
    rows: pd.DataFrame,
    player_histories: dict[str, pd.Series],
    rolling_window: int,
    team_priors: Optional[dict] = None,
    team_prior_alpha: float = 0.3,
) -> List[BoardLine]:
    """Build per-line projections.

    ``team_priors`` (optional) maps an upper-cased team abbrev to the
    ``blend_team_prior`` inputs for that team in this game; when a row's team
    matches, its (mu, sigma) is nudged toward the market's pace + team total.
    """
    team_priors = team_priors or {}
    lines: List[BoardLine] = []

    for _, row in rows.iterrows():
        player_name = str(row["player_name"])
        stat_type = _normalize_stat_type(row["stat_type"])
        latest = player_histories.get(player_name)
        if latest is None:
            continue

        team_abbrev = str(row.get("team") or "").upper()
        prior_inputs = team_priors.get(team_abbrev)
        line_value = float(row["line_value"])

        projection = project_prop_line(
            latest, stat_type, line_value,
            prior_inputs=prior_inputs,
            rolling_window=rolling_window,
            team_prior_alpha=team_prior_alpha,
        )
        mu = projection["mu"]
        sigma = projection["sigma"]
        distribution = projection["distribution"]
        prob_over = projection["prob_over"]

        over_odds = int(row["over_odds"]) if pd.notna(row["over_odds"]) else None
        under_odds = int(row["under_odds"]) if pd.notna(row["under_odds"]) else None

        implied_over_prob = american_to_implied_prob(over_odds) if over_odds is not None else None
        implied_under_prob = american_to_implied_prob(under_odds) if under_odds is not None else None

        ev_over = expected_value(prob_over, over_odds) if over_odds is not None else None
        prob_under = max(0.0, 1.0 - prob_over)
        ev_under = expected_value(prob_under, under_odds) if under_odds is not None else None

        lines.append(
            BoardLine(
                game_date=str(row["game_date"])[:10],
                team=str(row.get("team") or "") or None,
                player_name=player_name,
                stat_type=stat_type,
                line_value=line_value,
                over_odds=over_odds,
                under_odds=under_odds,
                mu=mu,
                sigma=sigma,
                distribution=distribution,
                prob_over=prob_over,
                implied_over_prob=implied_over_prob,
                implied_under_prob=implied_under_prob,
                ev_over=ev_over,
                ev_under=ev_under,
            )
        )
    return lines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a prop board for all player props in a game.")
    parser.add_argument("--game-date", required=True, help="Game date (YYYY-MM-DD).")
    parser.add_argument("--home-team", default=None, help="Home team abbreviation (e.g., LAL).")
    parser.add_argument("--away-team", default=None, help="Away team abbreviation (e.g., BOS).")
    parser.add_argument(
        "--stat-types",
        nargs="*",
        default=DEFAULT_STAT_TYPES,
        help="Stat types to include (default: points assists rebounds pra).",
    )
    parser.add_argument(
        "--books",
        nargs="*",
        default=None,
        help="Optional sportsbook filters; if omitted, all books are included.",
    )
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--n-games", type=int, default=DEFAULT_N_GAMES)
    parser.add_argument(
        "--output",
        default="-",
        help="Output path for CSV (default: '-' for stdout).",
    )
    return parser


def main():
    args = _build_parser().parse_args()

    rows = _fetch_betting_lines_for_game(
        db_path=args.db_path,
        game_date=args.game_date,
        stat_types=args.stat_types,
        home_team=args.home_team,
        away_team=args.away_team,
        books=args.books,
    )

    if rows.empty:
        print("No betting_lines rows found for the requested filters.")
        return

    # Build per-player histories once and reuse across stat types/lines.
    player_histories: dict[str, pd.Series] = {}
    for player_name in sorted(rows["player_name"].unique()):
        try:
            player_histories[player_name] = _build_player_history(
                player_name=player_name,
                n_games=args.n_games,
                rolling_window=args.rolling_window,
                db_path=args.db_path,
            )
        except Exception as exc:  # pragma: no cover - defensive skip for bad histories
            print(f"Skipping {player_name}: {exc}")

    # Resolve cross-book team priors for both sides of the matchup (when the
    # teams are known) so projections share the market's pace/total signal.
    team_priors: dict = {}
    if args.home_team and args.away_team:
        with DatabaseManager(db_path=args.db_path) as db:
            team_priors[args.home_team.upper()] = db.get_team_prior_inputs(
                args.home_team, args.away_team)
            team_priors[args.away_team.upper()] = db.get_team_prior_inputs(
                args.away_team, args.home_team)
        team_priors = {k: v for k, v in team_priors.items() if v}

    board_lines = _build_board_lines(
        rows=rows,
        player_histories=player_histories,
        rolling_window=args.rolling_window,
        team_priors=team_priors,
    )

    if not board_lines:
        print("No board lines generated (likely due to insufficient data for all players).")
        return

    df_out = pd.DataFrame([vars(line) for line in board_lines])
    df_out = df_out.sort_values(
        ["team", "player_name", "stat_type", "line_value"],
        kind="mergesort",
    )

    if args.output == "-" or not args.output:
        print(df_out.to_csv(index=False))
    else:
        df_out.to_csv(args.output, index=False)
        print(f"Wrote prop board to {args.output}")


if __name__ == "__main__":
    main()

