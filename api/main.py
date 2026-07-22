"""FastAPI app — read-only flagship-UI backend.

Boot locally with:
    uvicorn api.main:app --reload --port 8000

Every endpoint is a GET, validates its inputs through
``nba_model.web.input_validation`` (the same validators the Streamlit dispatch
uses), and returns a pydantic-typed JSON body. Nothing writes to the DB.
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from nba_model.web import input_validation as iv
from nba_model.model import edge_scanner as es

from . import __version__, config, schemas, services

app = FastAPI(
    title="NBA Props — Flagship UI API",
    version=__version__,
    description="Read-only wrapper over the NBA player-props data layer.",
    # This is a local-only tool; keep the interactive docs available.
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# The frontend is served by Vite (same-origin via its dev proxy in practice),
# but allow the common local dev origins directly too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _cache_headers(request, call_next):
    """Short, safe cache window on successful GETs (hourly-refreshed DB)."""
    response = await call_next(request)
    if request.method == "GET" and response.status_code == 200:
        if request.url.path == "/api/health":
            response.headers["Cache-Control"] = "no-store"
        elif request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = (
                f"public, max-age={config.CACHE_MAX_AGE_SECONDS}"
            )
    return response


def _require_db() -> str:
    db_path = config.get_db_path()
    if not config.db_exists(db_path):
        raise HTTPException(
            status_code=503,
            detail=f"database not found at {db_path}",
        )
    return db_path


def _validated_stat(stat: str) -> str:
    try:
        return iv.validate_stat_type(stat)
    except iv.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validated_team(team: Optional[str]) -> Optional[str]:
    if not team:
        return None
    try:
        return iv.validate_team_code(team)
    except iv.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validated_season(season: Optional[str]) -> Optional[str]:
    if not season:
        return None
    try:
        return iv.validate_season(season)
    except iv.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validated_n_games(n: int) -> int:
    try:
        return iv.validate_n_games(n)
    except iv.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validated_rolling(n: int) -> int:
    try:
        return iv.validate_rolling_window(n)
    except iv.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validated_since(h: float) -> float:
    try:
        return iv.validate_since_hours(h)
    except iv.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=schemas.HealthResponse)
def health() -> schemas.HealthResponse:
    db_path = config.get_db_path()
    exists = config.db_exists(db_path)
    extra = services.health(db_path, exists)
    return schemas.HealthResponse(
        status="ok" if exists else "degraded",
        version=__version__,
        db_path=db_path,
        db_exists=exists,
        **extra,
    )


@app.get("/api/meta", response_model=schemas.MetaResponse)
def meta() -> schemas.MetaResponse:
    db_path = _require_db()
    return schemas.MetaResponse(**services.meta(db_path))


@app.get("/api/slate/kpis", response_model=schemas.SlateKpis)
def slate_kpis() -> schemas.SlateKpis:
    db_path = _require_db()
    return schemas.SlateKpis(**services.slate_kpis(db_path))


@app.get("/api/slate/recent-games", response_model=schemas.RecentGamesResponse)
def recent_games(
    n: int = Query(12, ge=1, le=100),
    season: Optional[str] = None,
    season_type: Optional[str] = None,
    team: Optional[str] = None,
) -> schemas.RecentGamesResponse:
    db_path = _require_db()
    return schemas.RecentGamesResponse(**services.recent_games(
        db_path, n=n, season=_validated_season(season),
        season_type=season_type, team=_validated_team(team),
    ))


@app.get("/api/slate/edges", response_model=schemas.EdgeScanResponse)
def slate_edges(
    books: Optional[list[str]] = Query(None),
    stats: Optional[list[str]] = Query(None),
    since_hours: float = Query(48.0),
    n_games: int = Query(25, ge=1),
    model_mode: str = Query("chart_mean"),
    rolling_window: int = Query(10, ge=1),
    min_edge: Optional[float] = None,
    min_p_over: Optional[float] = None,
    only_positive_ev: bool = False,
    limit: int = Query(100, ge=1, le=500),
) -> schemas.EdgeScanResponse:
    db_path = _require_db()
    if model_mode not in es.MODEL_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"model_mode must be one of {list(es.MODEL_MODES)}",
        )
    clean_stats = [_validated_stat(s) for s in stats] if stats else None
    return schemas.EdgeScanResponse(**services.scan_edges(
        db_path,
        books=books,
        stats=clean_stats,
        since_hours=_validated_since(since_hours),
        n_games=_validated_n_games(n_games),
        model_mode=model_mode,
        rolling_window=_validated_rolling(rolling_window),
        min_edge=min_edge,
        min_p_over=min_p_over,
        only_positive_ev=only_positive_ev,
        limit=limit,
    ))


@app.get("/api/players/search", response_model=schemas.PlayerSearchResponse)
def players_search(
    q: str = "",
    team: Optional[str] = None,
    only_with_lines: bool = False,
    limit: int = Query(30, ge=1, le=100),
) -> schemas.PlayerSearchResponse:
    db_path = _require_db()
    return schemas.PlayerSearchResponse(**services.search_players(
        db_path, q=q, team=_validated_team(team),
        only_with_lines=only_with_lines, limit=limit,
    ))


@app.get("/api/players/{player_id}", response_model=schemas.PlayerDetailResponse)
def player_detail(
    player_id: int,
    stat: str = Query("points"),
    name: Optional[str] = None,
    n_games: int = Query(25, ge=1),
    rolling_window: int = Query(5, ge=1),
) -> schemas.PlayerDetailResponse:
    db_path = _require_db()
    canonical_stat = _validated_stat(stat)
    player_name = (name or "").strip() or services.resolve_player_name(
        db_path, player_id)
    if not player_name:
        raise HTTPException(status_code=404, detail="player not found")
    return schemas.PlayerDetailResponse(**services.player_detail(
        db_path, player_id, player_name, canonical_stat,
        n_games=_validated_n_games(n_games),
        rolling_window=_validated_rolling(rolling_window),
    ))
