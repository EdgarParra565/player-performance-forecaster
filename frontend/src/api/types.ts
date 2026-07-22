// Mirrors api/schemas.py. Kept intentionally close to the pydantic shapes so
// the network boundary is easy to reason about.

export interface Health {
  status: string;
  version: string;
  db_path: string;
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
