"""FastAPI app — read-only flagship-UI backend.

Boot locally with:
    uvicorn api.main:app --reload --port 8000

Every endpoint is a GET, validates its inputs through
``nba_model.web.input_validation`` (the same validators the Streamlit dispatch
uses), and returns a pydantic-typed JSON body. Nothing writes to the DB.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import asynccontextmanager
import unicodedata
from pathlib import Path as FsPath
from typing import Optional

from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from nba_model.web import input_validation as iv
from nba_model.model import edge_scanner as es

from . import __version__, config, db_sync, guards, paper_trades, parlay, schemas, services

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Interactive docs load Swagger UI from a CDN, which the production CSP
# blocks; the flagship image turns them off (API_DOCS=0).
_DOCS = _env_flag("API_DOCS", default=True)

@asynccontextmanager
async def _lifespan(_app):
    """Start the DB snapshot poller when object storage is configured
    (api/db_sync.py; DEPLOYMENT.md §16). Boot-time download happens in the
    container entrypoint before uvicorn starts."""
    try:
        interval = float(os.environ.get("DB_SYNC_INTERVAL_SECONDS", "0") or 0)
    except ValueError:
        interval = 0.0
    db_sync.start_background_sync(interval)
    yield


app = FastAPI(
    lifespan=_lifespan,
    title="NBA Props — Flagship UI API",
    version=__version__,
    description="Read-only wrapper over the NBA player-props data layer.",
    docs_url="/api/docs" if _DOCS else None,
    redoc_url=None,
    openapi_url="/api/openapi.json" if _DOCS else None,
)

# Production: restrict Host headers (DNS-rebinding defence). CSV of hosts.
_trusted = [h.strip() for h in os.environ.get("API_TRUSTED_HOSTS", "").split(",") if h.strip()]
if _trusted:
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_trusted)

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
async def _request_guards(request, call_next):
    """Per-IP rate limit, then the optional shared access code.

    Registered first => innermost of the http middlewares, so 401/429
    responses still get the security + cache headers added by the outer ones.
    See api/guards.py for budgets and the X-Forwarded-For contract.
    """
    g = guards.GUARDS
    path = request.url.path
    if request.method == "OPTIONS" or not g.is_api(path):
        return await call_next(request)
    ip_header = os.environ.get("API_CLIENT_IP_HEADER", "").strip()
    client = g.client_key(
        request.headers.get(ip_header) if ip_header else None,
        request.client.host if request.client else None,
    )
    t = guards.now()

    if g.enabled:
        tier = g.tier_for(path)
        if tier is not None:
            wait = tier.hit(client, t)
            if wait is not None:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "rate limit exceeded; slow down"},
                    headers={"Retry-After": guards.retry_after(wait)},
                )

    if g.access_code() and path not in guards.EXEMPT_PATHS:
        if g.enabled:
            wait = g.auth_fail.blocked(client, t)
            if wait is not None:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "too many wrong access codes"},
                    headers={"Retry-After": guards.retry_after(wait)},
                )
        if not g.code_ok(request.headers.get(guards.ACCESS_HEADER)):
            if g.enabled and request.headers.get(guards.ACCESS_HEADER):
                g.auth_fail.record(client, t)  # only WRONG codes count
            return JSONResponse(
                status_code=401,
                content={"detail": "access code required"},
                headers={"WWW-Authenticate": 'AccessCode realm="flagship"'},
            )
    return await call_next(request)


# Served on every response. The SPA needs inline style attributes (React /
# ECharts) but no inline or third-party scripts, fonts or connections.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}


@app.middleware("http")
async def _security_headers(request, call_next):
    response = await call_next(request)
    for key, value in SECURITY_HEADERS.items():
        # Swagger UI (dev only) needs its CDN; don't break it.
        if key == "Content-Security-Policy" and request.url.path in {"/api/docs", "/api/openapi.json"}:
            continue
        response.headers.setdefault(key, value)
    return response


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


@app.exception_handler(sqlite3.Error)
async def _sqlite_error(_request: Request, exc: sqlite3.Error) -> JSONResponse:
    """A locked / corrupt / schema-drifted DB is a 503, never a raw 500.

    The detail is generic on purpose: sqlite messages can carry SQL text.
    """
    return JSONResponse(status_code=503, content={"detail": "database unavailable"})


# Tables every endpoint depends on. Checked read-only BEFORE the data layer
# opens the file: ``DatabaseManager`` runs schema.sql + migrations on open,
# so pointing it at an empty/foreign file would silently write a fresh schema
# into it (the API must stay read-only).
REQUIRED_TABLES = config.REQUIRED_TABLES
_preflight_ok: dict[str, float] = {}


def _db_ready(db_path: str) -> bool:
    try:
        mtime = FsPath(db_path).stat().st_mtime
    except OSError:
        return False
    if _preflight_ok.get(db_path) == mtime:
        return True
    uri = f"{FsPath(db_path).resolve().as_uri()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=5) as conn:
            names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
    except sqlite3.Error:
        return False
    if not REQUIRED_TABLES.issubset(names):
        return False
    _preflight_ok[db_path] = mtime
    return True


def _require_db() -> str:
    db_path = config.get_db_path()
    if not config.db_exists(db_path) or not _db_ready(db_path):
        # No filesystem path in the body: it's public-facing.
        raise HTTPException(status_code=503, detail="database unavailable")
    return db_path


def _expose_db_path() -> bool:
    return os.environ.get("API_EXPOSE_DB_PATH", "").strip().lower() in {
        "1", "true", "yes", "on"}


def _name_key(name: str) -> str:
    """Accent/case-insensitive name key ("Nikola Jokić" == "nikola jokic")."""
    folded = unicodedata.normalize("NFKD", name or "")
    return "".join(c for c in folded if not unicodedata.combining(c)).casefold().strip()


MAX_BOOK_FILTERS = 25
MAX_BOOK_NAME_LEN = 40
SEASON_TYPES = frozenset({"Regular Season", "Playoffs", "PlayIn", "Play-In", "Pre Season"})
PLAYER_ID = Path(..., ge=1, le=2**31 - 1)


def _validated_books(books: Optional[list[str]]) -> Optional[list[str]]:
    if not books:
        return None
    if len(books) > MAX_BOOK_FILTERS:
        raise HTTPException(status_code=400,
                            detail=f"at most {MAX_BOOK_FILTERS} books")
    clean = []
    for b in books:
        text = str(b).strip()
        if not text or len(text) > MAX_BOOK_NAME_LEN:
            raise HTTPException(status_code=400, detail="invalid book name")
        clean.append(text)
    return clean


def _validated_season_type(season_type: Optional[str]) -> Optional[str]:
    if not season_type:
        return None
    if season_type not in SEASON_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"season_type must be one of {sorted(SEASON_TYPES)}")
    return season_type


def _chartable_stat(stat: str) -> str:
    """Stats the chart layer can series/fit (not steals/blocks/turnovers)."""
    try:
        return iv.validate_stat_type(stat, allowed=services.CHARTABLE_STATS)
    except iv.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _resolved_player(db_path: str, player_id: int, name: Optional[str]) -> str:
    """Canonical name for ``player_id``; 404 if unknown.

    ``name`` is a legacy hint only: it is accepted when it matches the DB name
    (accent/case-insensitive) and otherwise ignored — book lines are matched
    by name, so trusting it mixed one player's logs with another's lines.
    """
    resolved = services.resolve_player_name(db_path, player_id)
    if not resolved:
        raise HTTPException(status_code=404, detail="player not found")
    if name and _name_key(name) != _name_key(resolved):
        raise HTTPException(status_code=400,
                            detail="name does not match player_id")
    return resolved


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


def _validated_lookback(h: float) -> float:
    # Line-movement replay only: 1-year cap (not the 30-day since_hours cap) so
    # offseason snapshots months old still replay. See iv.validate_lookback_hours.
    try:
        return iv.validate_lookback_hours(h)
    except iv.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=schemas.HealthResponse)
def health() -> schemas.HealthResponse:
    db_path = config.get_db_path()
    exists = config.db_exists(db_path) and _db_ready(db_path)
    try:
        extra = services.health(db_path, exists)
        status = "ok" if exists else "degraded"
    except sqlite3.Error:
        extra = {"last_game_date": None, "freshest_scrape_utc": None,
                 "table_counts": {}}
        status = "degraded"
    return schemas.HealthResponse(
        status=status,
        version=__version__,
        # Absolute server paths are not for public eyes; opt in for local
        # debugging with API_EXPOSE_DB_PATH=1.
        db_path=db_path if _expose_db_path() else None,
        db_exists=exists,
        access_code_required=bool(guards.GUARDS.access_code()),
        db_sync=db_sync.STATE.last_status,
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
        season_type=_validated_season_type(season_type),
        team=_validated_team(team),
    ))


@app.get("/api/slate/edges", response_model=schemas.EdgeScanResponse)
def slate_edges(
    books: Optional[list[str]] = Query(None),
    stats: Optional[list[str]] = Query(None),
    since_hours: float = Query(48.0),
    n_games: int = Query(25, ge=3),
    model_mode: str = Query("chart_mean"),
    rolling_window: int = Query(10, ge=1),
    # ge/le also reject NaN / inf (comparisons with NaN are false).
    min_edge: Optional[float] = Query(None, ge=0.0, le=1.0),
    min_p_over: Optional[float] = Query(None, ge=0.0, le=1.0),
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
        books=_validated_books(books),
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
    q: str = Query("", max_length=64),
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
    player_id: int = PLAYER_ID,
    stat: str = Query("points"),
    name: Optional[str] = Query(None, max_length=80),
    n_games: int = Query(25, ge=3),
    rolling_window: int = Query(5, ge=1),
) -> schemas.PlayerDetailResponse:
    db_path = _require_db()
    canonical_stat = _chartable_stat(stat)
    player_name = _resolved_player(db_path, player_id, name)
    return schemas.PlayerDetailResponse(**services.player_detail(
        db_path, player_id, player_name, canonical_stat,
        n_games=_validated_n_games(n_games),
        rolling_window=_validated_rolling(rolling_window),
    ))


@app.get("/api/players/{player_id}/line-movement",
         response_model=schemas.LineMovementResponse)
def line_movement(
    player_id: int = PLAYER_ID,
    stat: str = Query("points"),
    lookback_hours: float = Query(168.0),
) -> schemas.LineMovementResponse:
    db_path = _require_db()
    canonical_stat = _validated_stat(stat)
    _resolved_player(db_path, player_id, None)  # 404 like player detail
    return schemas.LineMovementResponse(**services.line_movement(
        db_path, player_id, canonical_stat,
        lookback_hours=_validated_lookback(lookback_hours),
    ))


@app.get("/api/cross-book", response_model=schemas.CrossBookResponse)
def cross_book(
    books: Optional[list[str]] = Query(None),
    stats: Optional[list[str]] = Query(None),
    since_hours: float = Query(48.0),
    n_games: int = Query(25, ge=3),
    model_mode: str = Query("chart_mean"),
    rolling_window: int = Query(10, ge=1),
    min_gap: float = Query(0.5, ge=0.0, le=100.0),
    min_books: int = Query(2, ge=2, le=20),
) -> schemas.CrossBookResponse:
    db_path = _require_db()
    if model_mode not in es.MODEL_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"model_mode must be one of {list(es.MODEL_MODES)}",
        )
    clean_stats = [_validated_stat(s) for s in stats] if stats else None
    return schemas.CrossBookResponse(**services.cross_book(
        db_path,
        books=_validated_books(books),
        stats=clean_stats,
        since_hours=_validated_since(since_hours),
        n_games=_validated_n_games(n_games),
        model_mode=model_mode,
        rolling_window=_validated_rolling(rolling_window),
        min_gap=min_gap,
        min_books=min_books,
    ))


@app.get("/api/teams/{team}/chart", response_model=schemas.TeamChartResponse)
def team_chart(
    team: str = Path(..., max_length=4),
    stat: str = Query("points"),
    n_games: int = Query(25, ge=3),
) -> schemas.TeamChartResponse:
    db_path = _require_db()
    try:
        canonical_team = iv.validate_team_code(team)
    except iv.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    canonical_stat = _chartable_stat(stat)
    return schemas.TeamChartResponse(**services.team_chart(
        db_path, canonical_team, canonical_stat,
        n_games=_validated_n_games(n_games),
    ))


# ---------------------------------------------------------------------------
# Parlay builder — compute-only (POST because legs are structured; nothing
# is persisted). Heavy rate-limit tier (Monte Carlo).
# ---------------------------------------------------------------------------

PARLAY_DISCLAIMER = (
    "Model estimate — not validated for real money. WS10 calibration gates "
    "are unpassed; correlation comes from shared game history and falls back "
    "to independence when that history is thin."
)
PARLAY_MAX_SIMS = 50_000  # public-deploy CPU bound (iv caps at 200k)


def _bad(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


@app.post("/api/parlay/price", response_model=schemas.ParlayResponse)
def parlay_price(req: schemas.ParlayRequest) -> schemas.ParlayResponse:
    db_path = _require_db()
    try:
        iv.validate_parlay_legs_count(len(req.legs))
        n_sims = iv.validate_n_sims(req.n_sims, hard_cap=PARLAY_MAX_SIMS)
        n_games = iv.validate_n_games(req.n_games, min_value=3)
        parlay_odds = iv.validate_american_odds(req.parlay_odds)
    except iv.ValidationError as exc:
        raise _bad(str(exc)) from exc

    legs: list[dict] = []
    seen: set[tuple] = set()
    for i, leg in enumerate(req.legs, start=1):
        stat = _chartable_stat(leg.stat)
        try:
            line = iv.validate_line(stat, leg.line)
            odds = iv.validate_american_odds(leg.odds)
        except iv.ValidationError as exc:
            raise _bad(f"leg {i}: {exc}") from exc
        key = (leg.player_id, stat, line, leg.side)
        if key in seen:
            raise _bad(f"leg {i} duplicates an earlier leg")
        seen.add(key)
        legs.append({
            "player_id": leg.player_id,
            "player_name": _resolved_player(db_path, leg.player_id, None),
            "stat": stat,
            "line": line,
            "side": leg.side,
            "odds": odds if odds is not None else parlay.DEFAULT_LEG_ODDS,
        })

    try:
        priced = parlay.price_parlay(
            db_path, legs, n_games=n_games, n_sims=n_sims, parlay_odds=parlay_odds,
        )
    except parlay.LegDataError as exc:
        raise _bad(str(exc)) from exc
    return schemas.ParlayResponse(**priced, disclaimer=PARLAY_DISCLAIMER)


# ---------------------------------------------------------------------------
# Paper trades (bet_log) + calibration — read-only
# ---------------------------------------------------------------------------

@app.get("/api/paper-trades", response_model=schemas.PaperTradesResponse)
def paper_trades_list(
    status: str = Query("all", max_length=16),
    limit: int = Query(200, ge=1, le=500),
) -> schemas.PaperTradesResponse:
    db_path = _require_db()
    if status not in paper_trades.STATUS_FILTERS:
        raise _bad(f"status must be one of {list(paper_trades.STATUS_FILTERS)}")
    return schemas.PaperTradesResponse(
        **paper_trades.list_paper_trades(db_path, status=status, limit=limit))


@app.get("/api/paper-trades/calibration", response_model=schemas.CalibrationResponse)
def paper_trades_calibration(
    source: str = Query("bet_log", max_length=16),
    stat: Optional[str] = Query(None, max_length=32),
    n_buckets: int = Query(10, ge=2, le=20),
) -> schemas.CalibrationResponse:
    db_path = _require_db()
    if source not in paper_trades.CALIBRATION_SOURCES:
        raise _bad(f"source must be one of {list(paper_trades.CALIBRATION_SOURCES)}")
    canonical = _validated_stat(stat) if stat and stat != "all" else None
    return schemas.CalibrationResponse(**paper_trades.calibration(
        db_path, source=source, stat=canonical, n_buckets=n_buckets))


# ---------------------------------------------------------------------------
# Same-origin production serving of the built SPA (frontend/dist)
# ---------------------------------------------------------------------------
# Opt-in: set FLAGSHIP_STATIC_DIR=/app/frontend/dist (the flagship Docker
# image does). One uvicorn process then serves the UI and /api/* from a single
# origin — no CORS, no dev proxy. Registered LAST so API routes win.

_STATIC_DIR = os.environ.get("FLAGSHIP_STATIC_DIR", "").strip()


def _mount_spa(static_dir: str) -> None:
    from fastapi.responses import FileResponse

    root = FsPath(static_dir).resolve()
    index = root / "index.html"
    if not index.is_file():
        raise RuntimeError(f"FLAGSHIP_STATIC_DIR has no index.html: {root}")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="not found")
        candidate = (root / full_path).resolve()
        # Path traversal guard: only files inside the build dir.
        if full_path and candidate.is_file() and root in candidate.parents:
            # Vite emits content-hashed names under assets/ -> cache forever.
            cache = ("public, max-age=31536000, immutable"
                     if full_path.startswith("assets/") else "public, max-age=300")
            return FileResponse(candidate, headers={"Cache-Control": cache})
        # Client-side route (/player/2544, /edges, ...) -> the app shell.
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


if _STATIC_DIR:
    _mount_spa(_STATIC_DIR)
