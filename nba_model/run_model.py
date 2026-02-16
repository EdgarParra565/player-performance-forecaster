import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Allow direct execution via ".../nba_model/run_model.py" by ensuring the
# project root (parent of nba_model/) is importable.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from nba_model.data.data_loader import DataLoader
from nba_model.model.correlation_calibration import calibrate_correlations, covariance_matrix
from nba_model.model.defense_adjustment import adjust_mu_for_defense
from nba_model.model.feature_engineering import add_rolling_stats
from nba_model.model.minutes_projection import project_minutes
from nba_model.model.odds import american_to_implied_prob, expected_value
from nba_model.model.parlay_ev import calculate_parlay_ev
from nba_model.model.parlay_simulation import simulate_multi_leg_sgp
from nba_model.model.simulation import (
    SUPPORTED_DISTRIBUTIONS,
    monte_carlo_over,
    normalize_distribution_name,
)


"""
python3 -m nba_model.run_model \
  --mode parlay \
  --sportsbook prizepicks \
  --player "LeBron James" \
  --parlay-stats pts ast reb \
  --parlay-lines 27.5 7.5 8.5 \
  --parlay-odds 3.0 \
  --parlay-odds-format multiplier

python3 -m nba_model.run_model \
  --mode parlay \
  --sportsbook underdog \
  --player "LeBron James" \
  --parlay-stats points assists rebounds \
  --parlay-lines 27.5 7.5 8.5 \
  --parlay-odds 2.85 \
  --parlay-odds-format decimal

python3 -m nba_model.simple_ui
"""

DEFAULT_PLAYER_NAME = "LeBron James"
DEFAULT_ROLLING_WINDOW = 10
DEFAULT_POINTS_LINE = 27.5
DEFAULT_AMERICAN_ODDS = -110
DEFAULT_OPP_DEF_RATING = 112.5
DEFAULT_VEGAS_SPREAD = 11.5
DEFAULT_LEAGUE_AVG_DEF_RATING = 113.0
DEFAULT_DEFENSE_SENSITIVITY = 0.4
DEFAULT_BLOWOUT_THRESHOLD = 10.0
DEFAULT_BLOWOUT_PENALTY = 0.12
DEFAULT_SINGLE_PROP_DISTRIBUTION = "normal"
SINGLE_PROP_DISTRIBUTION_CHOICES = list(SUPPORTED_DISTRIBUTIONS)
DEFAULT_PARLAY_STATS = ["points", "assists", "rebounds"]
DEFAULT_PARLAY_LINES = [27.5, 7.5, 8.5]

_PARLAY_STAT_ALIASES = {
    "pts": "points",
    "point": "points",
    "points": "points",
    "ast": "assists",
    "assist": "assists",
    "assists": "assists",
    "reb": "rebounds",
    "rebound": "rebounds",
    "rebounds": "rebounds",
}


def _normalize_parlay_stats(stats: list[str]) -> list[str]:
    """Normalize stat aliases for parlay legs (e.g., pts -> points)."""
    normalized = []
    unknown = []
    for stat in stats:
        key = str(stat).strip().lower()
        mapped = _PARLAY_STAT_ALIASES.get(key)
        if mapped:
            normalized.append(mapped)
        else:
            unknown.append(stat)
    if unknown:
        raise ValueError(
            f"Unsupported parlay stat(s): {unknown}. "
            f"Supported stats: {sorted(set(_PARLAY_STAT_ALIASES.values()))}"
        )
    return normalized


def _decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds (or payout multiplier) to American odds."""
    if decimal_odds <= 1:
        raise ValueError("Decimal/multiplier odds must be > 1.0")
    if decimal_odds >= 2:
        return int(round((decimal_odds - 1) * 100))
    return int(round(-100 / (decimal_odds - 1)))


def _resolve_parlay_american_odds(
    parlay_odds: Optional[float],
    parlay_odds_format: str,
    fallback_american_odds: int,
) -> int:
    """Resolve parlay odds input into a single American odds value."""
    if parlay_odds is None:
        return int(fallback_american_odds)

    odds_format = parlay_odds_format.lower()
    if odds_format == "american":
        return int(round(parlay_odds))
    if odds_format in {"decimal", "multiplier"}:
        return _decimal_to_american(float(parlay_odds))
    raise ValueError(f"Unsupported parlay_odds_format: {parlay_odds_format}")


def _clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp numeric value to closed interval [min_value, max_value]."""
    return max(min_value, min(float(value), max_value))


def _apply_correlation_severity(corr_matrix, correlation_severity: float):
    """
    Scale off-diagonal correlation strength while preserving diagonal 1.0.

    correlation_severity:
      - 0.0 => no correlation (identity matrix)
      - 1.0 => baseline empirical correlations
      - >1.0 => amplified correlation impact (clipped for stability)
    """
    severity = max(0.0, float(correlation_severity))
    values = corr_matrix.values.astype(float)
    n = values.shape[0]
    identity = np.eye(n)
    off_diag = values - identity
    scaled = identity + off_diag * severity
    scaled = np.clip(scaled, -0.99, 0.99)
    np.fill_diagonal(scaled, 1.0)
    return corr_matrix.__class__(scaled, index=corr_matrix.index, columns=corr_matrix.columns)


def _ensure_psd_covariance(cov_matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Ensure covariance matrix is positive semidefinite for simulation stability.
    """
    sym = (cov_matrix + cov_matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def run_single_prop(
    player_name: str,
    line: float,
    rolling_window: int,
    american_odds: int,
    opp_def_rating: float,
    vegas_spread: float,
    league_avg_def_rating: float = DEFAULT_LEAGUE_AVG_DEF_RATING,
    defense_sensitivity: float = DEFAULT_DEFENSE_SENSITIVITY,
    blowout_threshold: float = DEFAULT_BLOWOUT_THRESHOLD,
    blowout_penalty: float = DEFAULT_BLOWOUT_PENALTY,
    n_games: int = 75,
    show_plot: bool = False,
    distribution: str = DEFAULT_SINGLE_PROP_DISTRIBUTION,
    defense_severity: float = 1.0,
    minutes_penalty_severity: float = 1.0,
    sigma_severity: float = 1.0,
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

    base_blowout_penalty = _clamp(float(blowout_penalty), 0.0, 0.95)
    effective_blowout_penalty = _clamp(
        base_blowout_penalty * max(0.0, float(minutes_penalty_severity)),
        0.0,
        0.95,
    )
    effective_defense_sensitivity = max(
        0.0,
        float(defense_sensitivity) * max(0.0, float(defense_severity)),
    )
    adjusted_sigma = max(0.01, float(sigma) * max(0.0, sigma_severity))

    proj_minutes = project_minutes(
        avg_minutes,
        abs(vegas_spread),
        blowout_threshold=float(blowout_threshold),
        blowout_penalty=effective_blowout_penalty,
    )
    mu = adjust_mu_for_defense(
        ppm * proj_minutes,
        opp_def_rating,
        league_avg_def_rating=float(league_avg_def_rating),
        sensitivity=effective_defense_sensitivity,
    )

    selected_distribution = normalize_distribution_name(distribution)
    p_over = monte_carlo_over(
        mu,
        adjusted_sigma,
        line,
        distribution=selected_distribution,
        sample_size=int(rolling_window),
    )
    implied = american_to_implied_prob(american_odds)
    ev = expected_value(p_over, american_odds)

    print(f"\nSingle Prop Model: {player_name} points")
    print(f"Distribution: {selected_distribution}")
    print(f"Projected minutes: {proj_minutes:.1f}")
    print(f"Expected points (mu): {mu:.2f}")
    print(f"Defense sensitivity: {defense_sensitivity:.3f} -> effective {effective_defense_sensitivity:.3f}")
    print(f"Blowout penalty: {base_blowout_penalty:.3f} -> effective {effective_blowout_penalty:.3f}")
    print(f"Sigma: {sigma:.2f} -> adjusted {adjusted_sigma:.2f}")
    print(f"Model P(OVER): {p_over:.2%}")
    print(f"Book Implied P: {implied:.2%}")
    print(f"EV: {ev:.3f}")

    if show_plot:
        from nba_model.visualization.distribution_plot import plot_distribution

        plot_distribution(mu, adjusted_sigma, line)

    return {
        "player": player_name,
        "line": line,
        "mu": float(mu),
        "sigma": float(sigma),
        "sigma_adjusted": float(adjusted_sigma),
        "prob_over": float(p_over),
        "implied_prob": float(implied),
        "ev": float(ev),
        "distribution": selected_distribution,
        "league_avg_def_rating": float(league_avg_def_rating),
        "defense_sensitivity": float(defense_sensitivity),
        "defense_sensitivity_effective": float(effective_defense_sensitivity),
        "blowout_threshold": float(blowout_threshold),
        "blowout_penalty": float(base_blowout_penalty),
        "blowout_penalty_effective": float(effective_blowout_penalty),
        "defense_severity": float(defense_severity),
        "minutes_penalty_severity": float(minutes_penalty_severity),
        "sigma_severity": float(sigma_severity),
    }


def run_parlay_demo(
    player_name: str,
    stats_cols: list[str],
    lines: list[float],
    american_odds: int,
    sportsbook: str = "custom",
    n_games: int = 100,
    n_sims: int = 20000,
    correlation_severity: float = 1.0,
    volatility_severity: float = 1.0,
):
    """Run multi-leg SGP simulation with correlation derived from player history."""
    if len(stats_cols) != len(lines):
        raise ValueError("parlay stats and lines must be the same length")
    if len(stats_cols) < 2:
        raise ValueError("Provide at least 2 legs for a multi-leg parlay")

    loader = DataLoader()
    df_player = loader.load_player_data(player_name, n_games=n_games)
    df_player = add_rolling_stats(df_player, window=DEFAULT_ROLLING_WINDOW)

    missing_stats = [col for col in stats_cols if col not in df_player.columns]
    if missing_stats:
        raise KeyError(f"Missing columns for parlay simulation: {missing_stats}")

    corr_matrix = calibrate_correlations(df_player, stats_cols)
    corr_matrix = _apply_correlation_severity(corr_matrix, correlation_severity=correlation_severity)
    volatility_scale = max(0.01, float(volatility_severity))
    stds = {col: float(df_player[col].dropna().std()) * volatility_scale for col in stats_cols}
    means = [float(df_player[col].dropna().tail(DEFAULT_ROLLING_WINDOW).mean()) for col in stats_cols]
    cov_matrix = covariance_matrix(corr_matrix, stds)
    cov_matrix = _ensure_psd_covariance(cov_matrix)

    prob = simulate_multi_leg_sgp(means, cov_matrix, lines, n=n_sims)
    ev = calculate_parlay_ev(prob, american_odds)
    implied = american_to_implied_prob(american_odds)
    leg_desc = ", ".join(f"{stat} > {line}" for stat, line in zip(stats_cols, lines))

    print(f"\nParlay Demo: {player_name} ({sportsbook})")
    print(f"Legs: {leg_desc}")
    print(f"Parlay odds: {american_odds:+d} (American)")
    print(f"Book Implied P: {implied:.2%}")
    print(f"Correlation severity: {correlation_severity:.2f}")
    print(f"Volatility severity: {volatility_severity:.2f}")
    print(f"Means: {[round(v, 2) for v in means]}")
    print(f"SGP probability: {prob:.2%}")
    print(f"Parlay EV: {ev:.3f}")

    return {
        "player": player_name,
        "sportsbook": sportsbook,
        "stats": stats_cols,
        "lines": lines,
        "means": means,
        "american_odds": int(american_odds),
        "implied_prob": float(implied),
        "probability": float(prob),
        "ev": float(ev),
        "correlation_severity": float(correlation_severity),
        "volatility_severity": float(volatility_severity),
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
    parser.add_argument("--league-avg-def-rating", type=float, default=DEFAULT_LEAGUE_AVG_DEF_RATING)
    parser.add_argument("--defense-sensitivity", type=float, default=DEFAULT_DEFENSE_SENSITIVITY)
    parser.add_argument("--blowout-threshold", type=float, default=DEFAULT_BLOWOUT_THRESHOLD)
    parser.add_argument("--blowout-penalty", type=float, default=DEFAULT_BLOWOUT_PENALTY)
    parser.add_argument(
        "--distribution",
        choices=SINGLE_PROP_DISTRIBUTION_CHOICES,
        default=DEFAULT_SINGLE_PROP_DISTRIBUTION,
        help="Distribution family for single-prop simulation.",
    )
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--defense-severity", type=float, default=1.0)
    parser.add_argument("--minutes-penalty-severity", type=float, default=1.0)
    parser.add_argument("--sigma-severity", type=float, default=1.0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--sportsbook", choices=["custom", "prizepicks", "underdog"], default="custom")
    parser.add_argument("--parlay-stats", nargs="+", default=DEFAULT_PARLAY_STATS)
    parser.add_argument("--parlay-lines", nargs="+", type=float, default=DEFAULT_PARLAY_LINES)
    parser.add_argument(
        "--parlay-odds",
        type=float,
        default=None,
        help="Parlay odds from sportsbook. Use with --parlay-odds-format.",
    )
    parser.add_argument(
        "--parlay-odds-format",
        choices=["american", "decimal", "multiplier"],
        default="american",
        help="Format of --parlay-odds (PrizePicks/Underdog often use multiplier style).",
    )
    parser.add_argument("--correlation-severity", type=float, default=1.0)
    parser.add_argument("--volatility-severity", type=float, default=1.0)
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
            league_avg_def_rating=args.league_avg_def_rating,
            defense_sensitivity=args.defense_sensitivity,
            blowout_threshold=args.blowout_threshold,
            blowout_penalty=args.blowout_penalty,
            n_games=args.n_games,
            show_plot=args.plot,
            distribution=args.distribution,
            defense_severity=args.defense_severity,
            minutes_penalty_severity=args.minutes_penalty_severity,
            sigma_severity=args.sigma_severity,
        )

    if args.mode in {"parlay", "both"}:
        stats_cols = _normalize_parlay_stats(args.parlay_stats)
        lines = [float(v) for v in args.parlay_lines]
        if len(stats_cols) != len(lines):
            raise ValueError(
                f"--parlay-stats length ({len(stats_cols)}) must match "
                f"--parlay-lines length ({len(lines)})"
            )
        parlay_american_odds = _resolve_parlay_american_odds(
            parlay_odds=args.parlay_odds,
            parlay_odds_format=args.parlay_odds_format,
            fallback_american_odds=args.odds,
        )
        run_parlay_demo(
            player_name=args.player,
            stats_cols=stats_cols,
            lines=lines,
            american_odds=parlay_american_odds,
            sportsbook=args.sportsbook,
            n_games=args.n_games,
            correlation_severity=args.correlation_severity,
            volatility_severity=args.volatility_severity,
        )


if __name__ == "__main__":
    main()
