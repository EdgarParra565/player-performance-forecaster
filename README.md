# ProbabilityAlgo
NBA Player Props Betting Model

This is a probabilistic sports betting analysis system that models NBA player performance 
to identify +EV (positive expected value) betting opportunities in the player props market.
Core Objective
The system aims to:

Build statistical models of player performance using historical NBA data
Generate probability distributions for player stats (points, assists, rebounds, minutes)
Compare model-generated probabilities against sportsbook odds
Identify betting opportunities where the model's probability suggests better value than 
the market price

How It Works
1. Data Collection (data_loader.py)

Pulls historical game logs from the NBA API for any player
Retrieves granular game-by-game statistics (points, minutes, shooting %, etc.)
Organizes data chronologically for time-series analysis

2. Feature Engineering (feature_engineering.py)

Calculates rolling statistics (10-game windows by default)
Computes:

Rolling mean points (recent scoring average)
Rolling standard deviation (volatility/consistency)
Points per minute (efficiency metric)
Rolling minutes played



This creates a dynamic baseline that adapts to recent form rather than using 
season-long averages.
3. Contextual Adjustments
Defense Adjustment (defense_adjustment.py):

Modifies expected points based on opponent's defensive rating
Better defenses → lower expected scoring
Uses sensitivity parameter to control how much defense matters

Minutes Projection (minutes_projection.py):

Adjusts expected playing time based on Vegas spread
Blowout games (spread ≥10 points) → reduced minutes for stars
Applies ~12% penalty in expected blowouts

4. Probability Generation
The model assumes player performance follows a normal distribution with:

μ (mu): Expected value = (points per minute) × (projected minutes) × (defense adjustment)
σ (sigma): Standard deviation from rolling window

Two methods calculate P(over):
Closed-form (prob.py):
Uses scipy.stats.norm.cdf for exact probability
Monte Carlo Simulation (simulation.py):
Runs 10,000+ simulations drawing from N(μ, σ²)
Counts how often simulated performance exceeds the line
5. Same-Game Parlays (SGP)
Correlation Modeling (correlation_calibration.py):

Calculates historical correlations between stats (e.g., points & assists)
Converts correlation matrix to covariance matrix
Critical because parlays compound correlation risk

Multi-Leg Simulation (parlay_simulation.py):

Uses multivariate normal distribution to simulate correlated outcomes
Example: Points and minutes are positively correlated (~0.6)
Simulates all legs together to get true joint probability
Much more accurate than multiplying independent probabilities

6. Expected Value (EV) Calculation (odds.py)
Converts American odds to implied probability:

Odds of -110 → ~52.4% implied probability
Compares this to model probability

EV Formula:
EV = (model_prob × payout) - ((1 - model_prob) × stake)
Positive EV = model believes bet has long-term profit potential
7. Tracking & Validation (line_tracking.py)

Logs model predictions vs opening/closing lines
Enables backtesting to validate model accuracy
Tracks line movement (sharp money indicators)

Example Workflow
Scenario: Betting on LeBron James over 27.5 points at -110 odds

Load LeBron's last 50 games
Calculate 10-game rolling average: 28.2 PPG, σ = 6.5
Tonight's opponent has 112.5 defensive rating (vs league avg 113.0)
Vegas spread is -11.5 (Lakers favored heavily)
Adjust minutes: 35 min → 30.8 min (blowout risk)
Calculate μ: (0.82 PPM × 30.8 min) × defense factor = 25.7 points
Run simulation: Model gives 38% chance of over
Book implies 52.4% (via -110 odds)
Result: Model says UNDER has value, skip this bet

When model shows 58% and book shows 52% → Positive EV opportunity on OVER
Advanced Features
Odds Ingestion (odds_ingestion.py):

Connects to odds APIs (The Odds API, etc.)
Pulls real-time lines across multiple sportsbooks
Enables line shopping for best prices

Parlay Filtering (parlay_ev.py):

Screens hundreds of potential parlays
Only surfaces those with EV > 5% threshold
Accounts for increased variance in multi-leg bets

Visualization (distribution_plot.py):

Plots probability density function
Shows where betting line sits relative to expected performance
Visual intuition for edge identification

Key Assumptions & Limitations
Assumptions:

Player performance follows normal distribution (reasonable for counting stats)
Recent games (10-game window) predict future better than season average
Correlations remain stable within games

Limitations:

Doesn't account for: injuries, rest days, back-to-backs, home/away splits
Defense ratings are team-level (not matchup-specific)
Model is only as good as the rolling window choice
Requires discipline to bet only when EV threshold is met

Use Case
This is a data-driven decision support tool for sports bettors. Instead of betting 
on gut feel or narratives, it:

Quantifies uncertainty through probability distributions
Identifies market inefficiencies systematically
Enables +EV betting strategies over large sample sizes

The edge comes from better modeling (accounting for defense, minutes, recency) than 
recreational bettors, and faster than sportsbooks can adjust lines.