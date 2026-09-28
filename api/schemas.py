"""Pydantic response models for the read-only API.

These describe the JSON shapes the frontend consumes. They are intentionally
permissive on optional/nullable fields — most live-line surfaces are empty
during the NBA offseason, and the UI renders deliberate empty-states rather
than assuming data is present.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    db_path: Optional[str] = None  # only with API_EXPOSE_DB_PATH=1
    # True when FLAGSHIP_ACCESS_CODE is set: the UI prompts for the code.
    access_code_required: bool = False
    # Snapshot delivery status: never | not-configured | updated | unchanged
    # | empty | error (api/db_sync.py). Error detail stays in server logs.
    db_sync: str = "never"
    db_exists: bool
    last_game_date: Optional[str] = None
    freshest_scrape_utc: Optional[str] = None
    table_counts: dict[str, int] = {}


class SlateKpis(BaseModel):
    games_in_db: int
    players_tracked: int
    books_producing: int
    freshest_scrape_utc: Optional[str] = None
    last_game_date: Optional[str] = None
    prop_lines_recent: int
    edges_positive_ev: Optional[int] = None


class RecentGame(BaseModel):
    game_id: str
    game_date: Optional[str] = None
    season: Optional[str] = None
    season_type: Optional[str] = None
    away_abbrev: Optional[str] = None
    away_name: Optional[str] = None
    away_pts: Optional[float] = None
    home_abbrev: Optional[str] = None
    home_name: Optional[str] = None
    home_pts: Optional[float] = None
    matchup: Optional[str] = None
    winner: Optional[str] = None


class RecentGamesResponse(BaseModel):
    rows: list[RecentGame]
    count: int


class EdgeRow(BaseModel):
    # ``model_*`` fields would otherwise collide with pydantic's protected
    # namespace; these mirror the edge-scanner column names exactly.
    model_config = ConfigDict(protected_namespaces=())

    book: str
    player_name: str
    stat_type: str
    book_line: Optional[float] = None
    model_mu: Optional[float] = None
    model_sigma: Optional[float] = None
    line_vs_mu: Optional[float] = None
    p_over: Optional[float] = None
    p_under: Optional[float] = None
    best_side: Optional[str] = None
    model_edge: Optional[float] = None
    ev_best: Optional[float] = None
    consensus_mean: Optional[float] = None
    pct_from_consensus: Optional[float] = None
    observed_hours_ago: Optional[float] = None
    observed_at_utc: Optional[str] = None
    n_games_used: Optional[int] = None
    distribution: Optional[str] = None
    model_mode: Optional[str] = None


class EdgeScanResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    rows: list[EdgeRow]
    n_lines: int
    n_scored: int
    n_returned: int
    model_mode: str
    books_available: list[str]
    stats_available: list[str]


class PlayerSearchRow(BaseModel):
    player_id: int
    player_name: str
    team: Optional[str] = None
    n_books: int = 0


class PlayerSearchResponse(BaseModel):
    rows: list[PlayerSearchRow]
    count: int


class SeriesPoint(BaseModel):
    game_date: Optional[str] = None
    value: float
    rolling_mean: Optional[float] = None
    opponent: Optional[str] = None
    home_away: Optional[str] = None
    result: Optional[str] = None


class HistogramBin(BaseModel):
    x0: float
    x1: float
    count: int


class FittedPoint(BaseModel):
    x: float
    y: float


class BookLineRow(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    book: str
    line: Optional[float] = None
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None
    p_over: Optional[float] = None
    p_under: Optional[float] = None
    best_side: Optional[str] = None
    model_edge: Optional[float] = None
    ev_over: Optional[float] = None
    ev_under: Optional[float] = None
    hit_rate: Optional[float] = None
    breakeven: Optional[float] = None
    is_dfs: bool = False


class PlayerDetailKpis(BaseModel):
    n_games: int
    mu: Optional[float] = None
    sigma: Optional[float] = None
    market_consensus_line: Optional[float] = None
    n_books: int = 0
    positive_ev_sides: int = 0


class PlayerDetailResponse(BaseModel):
    player_id: int
    player_name: str
    stat_type: str
    n_games: int
    rolling_window: int
    kpis: PlayerDetailKpis
    series: list[SeriesPoint]
    histogram: list[HistogramBin]
    fitted: list[FittedPoint]
    distribution: str
    book_lines: list[BookLineRow]
    notes: list[str] = []
    last_line_scraped_utc: Optional[str] = None


class MetaResponse(BaseModel):
    stats: list[str]
    teams: list[str]
    seasons: list[str]
    books: list[str]


# --- Cross-book layer -----------------------------------------------------

class CrossBookRow(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    player_name: str
    stat_type: str
    n_books: int
    line_min: Optional[float] = None
    line_max: Optional[float] = None
    line_gap: Optional[float] = None
    best_over_book: Optional[str] = None
    best_under_book: Optional[str] = None
    consensus_mean: Optional[float] = None
    p_over_at_line_min: Optional[float] = None
    p_over_at_line_max: Optional[float] = None
    middle_size: Optional[float] = None
    opportunity_type: Optional[str] = None
    model_mu: Optional[float] = None
    model_sigma: Optional[float] = None


class ArbRow(BaseModel):
    player_name: str
    stat_type: str
    game_date: Optional[str] = None
    over_book: str
    over_line: Optional[float] = None
    over_odds: Optional[int] = None
    under_book: str
    under_line: Optional[float] = None
    under_odds: Optional[int] = None
    implied_over: Optional[float] = None
    implied_under: Optional[float] = None
    combined_implied: Optional[float] = None
    devig_over: Optional[float] = None
    devig_under: Optional[float] = None
    guaranteed_margin: Optional[float] = None
    legs: Optional[str] = None


class CrossBookKpis(BaseModel):
    pairs: int
    max_gap: float
    over_threshold: int
    arb_count: int
    freshest_hours: Optional[float] = None


class CrossBookResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    rows: list[CrossBookRow]
    arbs: list[ArbRow]
    kpis: CrossBookKpis
    n_lines: int
    n_scored: int
    min_gap: float
    model_mode: str
    books_available: list[str]
    stats_available: list[str]


# --- Line movement --------------------------------------------------------

class LineMovementPoint(BaseModel):
    ts: str
    line: float
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None


class LineMovementSeries(BaseModel):
    book: str
    points: list[LineMovementPoint]
    open_line: Optional[float] = None
    close_line: Optional[float] = None
    line_delta: Optional[float] = None


class LineMovementResponse(BaseModel):
    player_id: int
    stat_type: str
    series: list[LineMovementSeries]
    timestamps: list[str]
    n_books: int
    n_snapshots: int
    last_snapshot_utc: Optional[str] = None


# --- Team charts ----------------------------------------------------------

class TeamSeriesPoint(BaseModel):
    game_date: Optional[str] = None
    value: float
    opponent: Optional[str] = None
    home_away: Optional[str] = None
    result: Optional[str] = None


class TeamChartKpis(BaseModel):
    n_games: int
    mu: Optional[float] = None
    sigma: Optional[float] = None
    market_consensus_line: Optional[float] = None
    derived_reference_line: Optional[float] = None


class TeamChartResponse(BaseModel):
    team: str
    stat_type: str
    n_games: int
    series: list[TeamSeriesPoint]
    kpis: TeamChartKpis
    market_consensus_line: Optional[float] = None
    derived_reference_line: Optional[float] = None
    derived_reference_label: Optional[str] = None
    notes: list[str] = []


# ---------------------------------------------------------------------------
# Parlay builder (compute-only POST)
# ---------------------------------------------------------------------------

class ParlayLegIn(BaseModel):
    player_id: int = Field(..., ge=1, le=2**31 - 1)
    stat: str = Field(..., min_length=1, max_length=32)
    line: float
    side: Literal["over", "under"]
    odds: Optional[int] = None  # American; default -110


class ParlayRequest(BaseModel):
    # Hard parse bound; the 2..6 business rule is input_validation's.
    legs: list[ParlayLegIn] = Field(..., max_length=12)
    n_games: int = Field(25, ge=3, le=200)
    n_sims: Optional[int] = None
    parlay_odds: Optional[int] = None  # book's SGP price if known


class ParlayLegOut(BaseModel):
    player_id: int
    player_name: str
    stat: str
    line: float
    side: str
    odds: int
    n_games: int
    mu: float
    sigma: float
    p_over: float
    p_hit: float
    implied_prob: float
    ev: float


class CorrelationMatrix(BaseModel):
    labels: list[str]
    matrix: list[list[float]]


class ParlayResponse(BaseModel):
    legs: list[ParlayLegOut]
    n_games: int
    n_sims: int
    n_joint_games: int
    correlation_fallback: bool
    joint_prob: float
    joint_prob_se: float
    independent_prob: float
    correlation_lift: Optional[float] = None
    combined_decimal: float
    combined_american: Optional[int] = None
    offered_american: Optional[int] = None
    offered_is_custom: bool
    implied_prob: Optional[float] = None
    fair_american: Optional[int] = None
    ev_joint: Optional[float] = None
    ev_independent: Optional[float] = None
    correlation: CorrelationMatrix
    disclaimer: str


# ---------------------------------------------------------------------------
# Paper trades (bet_log) + calibration (read-only)
# ---------------------------------------------------------------------------

class PaperTradeRow(BaseModel):
    log_id: int
    created_at_utc: Optional[str] = None
    game_date: Optional[str] = None
    player_id: Optional[int] = None
    player_name: str
    stat_type: str
    book: Optional[str] = None
    line: Optional[float] = None
    side: str
    model_prob: Optional[float] = None
    implied_prob: Optional[float] = None
    edge: Optional[float] = None
    model_mode: Optional[str] = None
    stake_units: Optional[float] = None
    status: str
    settled_at_utc: Optional[str] = None
    actual_value: Optional[float] = None
    clv_delta: Optional[float] = None
    est_profit_units: Optional[float] = None


class PaperTradeSummary(BaseModel):
    total: int
    pending: int
    won: int
    lost: int
    push: int
    void: int
    win_rate: Optional[float] = None
    n_clv: int
    mean_clv: Optional[float] = None
    positive_clv_rate: Optional[float] = None
    est_units: Optional[float] = None


class PaperTradesResponse(BaseModel):
    status_filter: str
    rows: list[PaperTradeRow]
    summary: PaperTradeSummary


class ReliabilityBucket(BaseModel):
    stat_type: str
    bucket: int
    bucket_low: float
    bucket_high: float
    n: int
    mean_pred: float
    realized_rate: float
    calibration_gap: float


class BrierRow(BaseModel):
    stat_type: str
    n: int
    brier_score: float
    mean_pred: float
    realized_rate: float


class CalibrationResponse(BaseModel):
    source: str
    stat: str
    n_buckets: int
    n_settled: int
    stats_available: list[str]
    reliability: list[ReliabilityBucket]
    brier: Optional[BrierRow] = None
    brier_by_stat: list[BrierRow]
