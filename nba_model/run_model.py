import argparse

from nba_model.data.data_loader import DataLoader
from nba_model.model.correlation_calibration import calibrate_correlations, covariance_matrix
from nba_model.model.defense_adjustment import adjust_mu_for_defense
from nba_model.model.feature_engineering import add_rolling_stats
from nba_model.model.minutes_projection import project_minutes
from nba_model.model.odds import american_to_implied_prob, expected_value
from nba_model.model.parlay_ev import calculate_parlay_ev
from nba_model.model.parlay_simulation import simulate_multi_leg_sgp
from nba_model.model.simulation import monte_carlo_over


DEFAULT_PLAYER_NAME = "LeBron James"
DEFAULT_ROLLING_WINDOW = 10
DEFAULT_POINTS_LINE = 27.5
DEFAULT_AMERICAN_ODDS = -110
DEFAULT_OPP_DEF_RATING = 112.5
DEFAULT_VEGAS_SPREAD = 11.5


def run_single_prop(
    player_name: str,
    line: float,
    rolling_window: int,
    american_odds: int,
    opp_def_rating: float,
    vegas_spread: float,
    n_games: int = 75,
    show_plot: bool = False,
):
    """Run single-leg points model flow using unified feature columns."""
    loader = DataLoader()
    df = loader.load_player_data(player_name, n_games=n_games)
    df = add_rolling_stats(df, rolling_window)

    required_cols = [
        "rolling_mean_points_per_minute",
        "rolling_std_points",
        "rolling_mean_minutes",
    ]
    latest = df.dropna(subset=required_cols)
    if latest.empty:
        raise ValueError("Insufficient data after rolling window; increase n_games or reduce window.")
    latest = latest.iloc[-1]

    ppm = latest["rolling_mean_points_per_minute"]
    sigma = latest["rolling_std_points"]
    avg_minutes = latest["rolling_mean_minutes"]

    proj_minutes = project_minutes(avg_minutes, abs(vegas_spread))
    mu = adjust_mu_for_defense(ppm * proj_minutes, opp_def_rating)

    p_over = monte_carlo_over(mu, sigma, line)
    implied = american_to_implied_prob(american_odds)
    ev = expected_value(p_over, american_odds)

    print(f"\nSingle Prop Model: {player_name} points")
    print(f"Projected minutes: {proj_minutes:.1f}")
    print(f"Expected points (mu): {mu:.2f}")
    print(f"Sigma: {sigma:.2f}")
    print(f"Model P(OVER): {p_over:.2%}")
    print(f"Book Implied P: {implied:.2%}")
    print(f"EV: {ev:.3f}")

    if show_plot:
        from nba_model.visualization.distribution_plot import plot_distribution

        plot_distribution(mu, sigma, line)

    return {
        "player": player_name,
        "line": line,
        "mu": float(mu),
        "sigma": float(sigma),
        "prob_over": float(p_over),
        "implied_prob": float(implied),
        "ev": float(ev),
    }


def run_parlay_demo(
    player_name: str,
    lines: tuple[float, float, float],
    american_odds: int,
    n_games: int = 100,
    n_sims: int = 20000,
):
    """Run multi-leg SGP simulation with correlation derived from player history."""
    loader = DataLoader()
    df_player = loader.load_player_data(player_name, n_games=n_games)
    df_player = add_rolling_stats(df_player, window=DEFAULT_ROLLING_WINDOW)

    stats_cols = ["points", "assists", "rebounds"]
    missing_stats = [col for col in stats_cols if col not in df_player.columns]
    if missing_stats:
        raise KeyError(f"Missing columns for parlay simulation: {missing_stats}")

    corr_matrix = calibrate_correlations(df_player, stats_cols)
    stds = {col: float(df_player[col].dropna().std()) for col in stats_cols}
    means = [float(df_player[col].dropna().tail(DEFAULT_ROLLING_WINDOW).mean()) for col in stats_cols]
    cov_matrix = covariance_matrix(corr_matrix, stds)

    prob = simulate_multi_leg_sgp(means, cov_matrix, list(lines), n=n_sims)
    ev = calculate_parlay_ev(prob, american_odds)

    print(f"\nParlay Demo: {player_name} (PTS/AST/REB over)")
    print(f"Lines: {lines}")
    print(f"Means: {[round(v, 2) for v in means]}")
    print(f"SGP probability: {prob:.2%}")
    print(f"Parlay EV: {ev:.3f}")

    return {
        "player": player_name,
        "lines": lines,
        "means": means,
        "probability": float(prob),
        "ev": float(ev),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NBA player props model demos.")
    parser.add_argument("--mode", choices=["single", "parlay", "both"], default="single")
    parser.add_argument("--player", default=DEFAULT_PLAYER_NAME)
    parser.add_argument("--line", type=float, default=DEFAULT_POINTS_LINE)
    parser.add_argument("--odds", type=int, default=DEFAULT_AMERICAN_ODDS)
    parser.add_argument("--window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--opp-def-rating", type=float, default=DEFAULT_OPP_DEF_RATING)
    parser.add_argument("--spread", type=float, default=DEFAULT_VEGAS_SPREAD)
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--parlay-lines", nargs=3, type=float, default=[27.5, 7.5, 8.5])
    return parser


def main():
    args = _build_parser().parse_args()

    if args.mode in {"single", "both"}:
        run_single_prop(
            player_name=args.player,
            line=args.line,
            rolling_window=args.window,
            american_odds=args.odds,
            opp_def_rating=args.opp_def_rating,
            vegas_spread=args.spread,
            n_games=args.n_games,
            show_plot=args.plot,
        )

    if args.mode in {"parlay", "both"}:
        run_parlay_demo(
            player_name=args.player,
            lines=tuple(args.parlay_lines),
            american_odds=args.odds,
            n_games=args.n_games,
        )


if __name__ == "__main__":
    main()
