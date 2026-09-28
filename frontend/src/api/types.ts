// Mirrors api/schemas.py. Kept intentionally close to the pydantic shapes so
// the network boundary is easy to reason about.

export interface Health {
  status: string;
  version: string;
  db_path: string | null; // only with API_EXPOSE_DB_PATH=1
  access_code_required: boolean;
  db_exists: boolean;
  last_game_date: string | null;
  freshest_scrape_utc: string | null;
  table_counts: Record<string, number>;
}

export interface SlateKpis {
  games_in_db: number;
  players_tracked: number;
  books_producing: number;
  freshest_scrape_utc: string | null;
  last_game_date: string | null;
  prop_lines_recent: number;
  edges_positive_ev: number | null;
}

export interface RecentGame {
  game_id: string;
  game_date: string | null;
  season: string | null;
  season_type: string | null;
  away_abbrev: string | null;
  away_name: string | null;
  away_pts: number | null;
  home_abbrev: string | null;
  home_name: string | null;
  home_pts: number | null;
  matchup: string | null;
  winner: string | null;
}

export interface RecentGamesResponse {
  rows: RecentGame[];
  count: number;
}

export interface EdgeRow {
  book: string;
  player_name: string;
  stat_type: string;
  book_line: number | null;
  model_mu: number | null;
  model_sigma: number | null;
  line_vs_mu: number | null;
  p_over: number | null;
  p_under: number | null;
  best_side: string | null;
  model_edge: number | null;
  ev_best: number | null;
  consensus_mean: number | null;
  pct_from_consensus: number | null;
  observed_hours_ago: number | null;
  observed_at_utc: string | null;
  n_games_used: number | null;
  distribution: string | null;
  model_mode: string | null;
}

export interface EdgeScanResponse {
  rows: EdgeRow[];
  n_lines: number;
  n_scored: number;
  n_returned: number;
  model_mode: string;
  books_available: string[];
  stats_available: string[];
}

export interface PlayerSearchRow {
  player_id: number;
  player_name: string;
  team: string | null;
  n_books: number;
}

export interface PlayerSearchResponse {
  rows: PlayerSearchRow[];
  count: number;
}

export interface SeriesPoint {
  game_date: string | null;
  value: number;
  rolling_mean: number | null;
  opponent: string | null;
  home_away: string | null;
  result: string | null;
}

export interface HistogramBin {
  x0: number;
  x1: number;
  count: number;
}

export interface FittedPoint {
  x: number;
  y: number;
}

export interface BookLineRow {
  book: string;
  line: number | null;
  over_odds: number | null;
  under_odds: number | null;
  p_over: number | null;
  p_under: number | null;
  best_side: string | null;
  model_edge: number | null;
  ev_over: number | null;
  ev_under: number | null;
  hit_rate: number | null;
  breakeven: number | null;
  is_dfs: boolean;
}

export interface PlayerDetailKpis {
  n_games: number;
  mu: number | null;
  sigma: number | null;
  market_consensus_line: number | null;
  n_books: number;
  positive_ev_sides: number;
}

export interface PlayerDetail {
  player_id: number;
  player_name: string;
  stat_type: string;
  n_games: number;
  rolling_window: number;
  kpis: PlayerDetailKpis;
  series: SeriesPoint[];
  histogram: HistogramBin[];
  fitted: FittedPoint[];
  distribution: string;
  book_lines: BookLineRow[];
  notes: string[];
  last_line_scraped_utc: string | null;
}

export interface Meta {
  stats: string[];
  teams: string[];
  seasons: string[];
  books: string[];
}

export interface CrossBookRow {
  player_name: string;
  stat_type: string;
  n_books: number;
  line_min: number | null;
  line_max: number | null;
  line_gap: number | null;
  best_over_book: string | null;
  best_under_book: string | null;
  consensus_mean: number | null;
  p_over_at_line_min: number | null;
  p_over_at_line_max: number | null;
  middle_size: number | null;
  opportunity_type: string | null;
  model_mu: number | null;
  model_sigma: number | null;
}

export interface ArbRow {
  player_name: string;
  stat_type: string;
  game_date: string | null;
  over_book: string;
  over_line: number | null;
  over_odds: number | null;
  under_book: string;
  under_line: number | null;
  under_odds: number | null;
  implied_over: number | null;
  implied_under: number | null;
  combined_implied: number | null;
  devig_over: number | null;
  devig_under: number | null;
  guaranteed_margin: number | null;
  legs: string | null;
}

export interface CrossBookKpis {
  pairs: number;
  max_gap: number;
  over_threshold: number;
  arb_count: number;
  freshest_hours: number | null;
}

export interface CrossBookResponse {
  rows: CrossBookRow[];
  arbs: ArbRow[];
  kpis: CrossBookKpis;
  n_lines: number;
  n_scored: number;
  min_gap: number;
  model_mode: string;
  books_available: string[];
  stats_available: string[];
}

export interface LineMovementPoint {
  ts: string;
  line: number;
  over_odds: number | null;
  under_odds: number | null;
}

export interface LineMovementSeries {
  book: string;
  points: LineMovementPoint[];
  open_line: number | null;
  close_line: number | null;
  line_delta: number | null;
}

export interface LineMovementResponse {
  player_id: number;
  stat_type: string;
  series: LineMovementSeries[];
  timestamps: string[];
  n_books: number;
  n_snapshots: number;
  last_snapshot_utc: string | null;
}

export interface TeamSeriesPoint {
  game_date: string | null;
  value: number;
  opponent: string | null;
  home_away: string | null;
  result: string | null;
}

export interface TeamChartKpis {
  n_games: number;
  mu: number | null;
  sigma: number | null;
  market_consensus_line: number | null;
  derived_reference_line: number | null;
}

export interface TeamChartResponse {
  team: string;
  stat_type: string;
  n_games: number;
  series: TeamSeriesPoint[];
  kpis: TeamChartKpis;
  market_consensus_line: number | null;
  derived_reference_line: number | null;
  derived_reference_label: string | null;
  notes: string[];
}

// --- Parlay builder (POST /api/parlay/price) --------------------------------

export type LegSide = "over" | "under";

export interface ParlayLegInput {
  player_id: number;
  player_name: string; // display only; the API resolves the canonical name
  stat: string;
  line: number;
  side: LegSide;
  odds: number;
}

export interface ParlayLegPriced {
  player_id: number;
  player_name: string;
  stat: string;
  line: number;
  side: string;
  odds: number;
  n_games: number;
  mu: number;
  sigma: number;
  p_over: number;
  p_hit: number;
  implied_prob: number;
  ev: number;
}

export interface ParlayResponse {
  legs: ParlayLegPriced[];
  n_games: number;
  n_sims: number;
  n_joint_games: number;
  correlation_fallback: boolean;
  joint_prob: number;
  joint_prob_se: number;
  independent_prob: number;
  correlation_lift: number | null;
  combined_decimal: number;
  combined_american: number | null;
  offered_american: number | null;
  offered_is_custom: boolean;
  implied_prob: number | null;
  fair_american: number | null;
  ev_joint: number | null;
  ev_independent: number | null;
  correlation: { labels: string[]; matrix: number[][] };
  disclaimer: string;
}

// --- Paper trades (bet_log) + calibration ------------------------------------

export interface PaperTradeRow {
  log_id: number;
  created_at_utc: string | null;
  game_date: string | null;
  player_id: number | null;
  player_name: string;
  stat_type: string;
  book: string | null;
  line: number | null;
  side: string;
  model_prob: number | null;
  implied_prob: number | null;
  edge: number | null;
  model_mode: string | null;
  stake_units: number | null;
  status: string;
  settled_at_utc: string | null;
  actual_value: number | null;
  clv_delta: number | null;
  est_profit_units: number | null;
}

export interface PaperTradeSummary {
  total: number;
  pending: number;
  won: number;
  lost: number;
  push: number;
  void: number;
  win_rate: number | null;
  n_clv: number;
  mean_clv: number | null;
  positive_clv_rate: number | null;
  est_units: number | null;
}

export interface PaperTradesResponse {
  status_filter: string;
  rows: PaperTradeRow[];
  summary: PaperTradeSummary;
}

export interface ReliabilityBucket {
  stat_type: string;
  bucket: number;
  bucket_low: number;
  bucket_high: number;
  n: number;
  mean_pred: number;
  realized_rate: number;
  calibration_gap: number;
}

export interface BrierRow {
  stat_type: string;
  n: number;
  brier_score: number;
  mean_pred: number;
  realized_rate: number;
}

export interface CalibrationResponse {
  source: string;
  stat: string;
  n_buckets: number;
  n_settled: number;
  stats_available: string[];
  reliability: ReliabilityBucket[];
  brier: BrierRow | null;
  brier_by_stat: BrierRow[];
}
