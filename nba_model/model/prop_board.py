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
from nba_model.model.simulation import get_default_distribution


DEFAULT_DB_PATH = "data/database/nba_data.db"
DEFAULT_STAT_TYPES = ["points", "assists", "rebounds", "pra"]
DEFAULT_ROLLING_WINDOW = 10
DEFAULT_N_GAMES = 120


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
        clauses: list[str] = [
            "bl.game_date = :game_date",
            "lower(bl.stat_type) IN (:stat_types)",
        ]
        params: dict = {
            "game_date": str(game_date)[:10],
            "stat_types": tuple(stat_types),
        }

        if home_team and away_team:
            clauses.append("p.team IN (:home_team, :away_team)")
            params["home_team"] = home_team
            params["away_team"] = away_team
        elif home_team or away_team:
            team = home_team or away_team
            clauses.append("p.team = :team")
            params["team"] = team

        if books:
            books_norm = [str(b).strip() for b in books if str(b).strip()]
            if books_norm:
                clauses.append("bl.book IN (:books)")
                params["books"] = tuple(books_norm)

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
    """Return latest rolling stat row for a player."""
    loader = DataLoader(db_path=db_path)
    df = loader.load_player_data(player_name=player_name, n_games=n_games)
    df = add_rolling_stats(df, window=rolling_window)

    latest = df.dropna(subset=["rolling_mean_minutes"])
    if latest.empty:
        raise ValueError(f"Insufficient data for {player_name}; no rolling window rows available.")
    return latest.iloc[-1]


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
) -> List[BoardLine]:
    lines: List[BoardLine] = []

    for _, row in rows.iterrows():
        player_name = str(row["player_name"])
        stat_type = _normalize_stat_type(row["stat_type"])
        latest = player_histories.get(player_name)
        if latest is None:
            continue

        mu, sigma = _project_stat_from_history(latest, stat_type=stat_type)
        line_value = float(row["line_value"])
        distribution = get_default_distribution(stat_type)

        prob_over = prob_over_distribution(
            line=line_value,
            mu=mu,
            sigma=sigma,
            distribution=distribution,
            sample_size=int(rolling_window),
        )

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

    board_lines = _build_board_lines(
        rows=rows,
        player_histories=player_histories,
        rolling_window=args.rolling_window,
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

