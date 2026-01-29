from nba_model.data.data_loader import load_player_logs
from nba_model.model.feature_engineering import add_rolling_stats
from nba_model.model.simulation import monte_carlo_over
from nba_model.model.defense_adjustment import adjust_mu_for_defense
from nba_model.model.minutes_projection import project_minutes
from nba_model.model.odds import american_to_implied_prob, expected_value
from nba_model.visualization.distribution_plot import plot_distribution

# === CONFIG ===
PLAYER_ID = 2544
SEASON = "2023-24"
POINTS_LINE = 27.5
ROLLING_WINDOW = 10

# odds free to change
AMERICAN_ODDS = -110
# defensive rating free to change
OPP_DEF_RATING = 112.5
# spread line free to change
VEGAS_SPREAD = 11.5


def main():
    df = load_player_logs(PLAYER_ID, SEASON)
    df = add_rolling_stats(df, ROLLING_WINDOW)

    latest = df.dropna().iloc[-1]

    # Base rates
    ppm = latest["ppm_mean"]
    sigma = latest["pts_std"]
    avg_minutes = latest["min_mean"]

    # Minutes projection
    proj_minutes = project_minutes(avg_minutes, abs(VEGAS_SPREAD))

    # Expected points
    mu = ppm * proj_minutes
    mu = adjust_mu_for_defense(mu, OPP_DEF_RATING)

    # Probability + EV
    p_over = monte_carlo_over(mu, sigma, POINTS_LINE)
    implied = american_to_implied_prob(AMERICAN_ODDS)
    ev = expected_value(p_over, AMERICAN_ODDS)

    print(f"Projected minutes: {proj_minutes:.1f}")
    print(f"Expected points (μ): {mu:.2f}")
    print(f"σ: {sigma:.2f}")
    print(f"Model P(OVER): {p_over:.2%}")
    print(f"Book Implied P: {implied:.2%}")
    print(f"EV: {ev:.3f}")

    plot_distribution(mu, sigma, POINTS_LINE)


from nba_model.model.correlation_calibration import calibrate_correlations, covariance_matrix
from nba_model.model.parlay_simulation import simulate_multi_leg_sgp
from nba_model.model.parlay_ev import calculate_parlay_ev
from nba_model.model.odds_ingestion import fetch_odds

# Example workflow
def run_parlay_simulation():
    # Fetch odds (placeholder)
    odds_data = fetch_odds("YOUR_API_KEY_HERE")

    # Prepare stats for calibration
    stats_cols = ["PTS", "AST", "REB"]
    stds = {"PTS": 8.2, "AST": 4.5, "REB": 5.0}  # example
    df_player = load_player_logs(PLAYER_ID)
    corr_matrix = calibrate_correlations(df_player, stats_cols)
    cov_matrix = covariance_matrix(corr_matrix, stds)

    # Set lines for each leg
    lines = [27.5, 7.5, 8.5]  # points, assists, rebounds

    # Multi-leg probability
    means = [27.0, 7.2, 8.0]
    prob = simulate_multi_leg_sgp(means, cov_matrix, lines)
    print(f"Multi-leg SGP probability: {prob:.2%}")

    # Example parlay EV calculation
    example_odds = -110
    ev = calculate_parlay_ev(prob, example_odds)
    print(f"Expected value: {ev:.3f}")

if __name__ == "__main__":
    main()
