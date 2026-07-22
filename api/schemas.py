"""Pydantic response models for the read-only API.

These describe the JSON shapes the frontend consumes. They are intentionally
permissive on optional/nullable fields — most live-line surfaces are empty
during the NBA offseason, and the UI renders deliberate empty-states rather
than assuming data is present.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    status: str
    version: str
    db_path: str
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
