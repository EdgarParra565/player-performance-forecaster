"""Request-scoped READ-ONLY database access for the flagship API.

Every request opens exactly one ``DatabaseManager(path, read_only=True)``
(SQLite ``mode=ro``: no DDL, no migrations, never creates the file) on the
mounted file itself, and every data-layer read during that request reuses it.

Why the module rebinding: the read paths the API wraps (player_charts,
edge_scanner, cross_book_arb, calibration_report — owned by other agents)
each do ``with DatabaseManager(db_path=db_path) as db:`` internally, i.e. a
WRITER open (schema.sql + migrations) 3–6 times per request. ``install()``
rebinds the ``DatabaseManager`` name in those modules (in this process only)
to ``RequestScopedDatabaseManager``. Inside ``read_only_session()`` it hands
back the request's shared read-only connection, with ``close`` / ``__exit__``
as no-ops (the session closes it once). Outside a session it is the real
class, unchanged — so the ETL, tests and anything else behave as before.

A write attempted through the shared connection fails loudly with
``sqlite3.OperationalError`` ("attempt to write a readonly database"); the
API maps that to a 503 rather than ever writing the host file.
"""
from __future__ import annotations

import importlib
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from nba_model.data.database import db_manager as _dbm

from . import config

logger = logging.getLogger("api.db")

_RealDatabaseManager = _dbm.DatabaseManager
DatabaseNotReadyError = _dbm.DatabaseNotReadyError

# Modules whose ``DatabaseManager`` name is rebound by ``install()``.
DATA_MODULES = (
    "api.services",
    "api.paper_trades",
    "nba_model.visualization.player_charts",
    "nba_model.model.edge_scanner",
    "nba_model.model.cross_book_arb",
    "nba_model.evaluation.calibration_report",
)


class DbNotReady(Exception):
    """The mounted file can't be served. ``state`` is one of:
    ``not_mounted`` (missing or 0-byte — e.g. ``docker run`` without -v),
    ``unmigrated`` (a real NBA DB missing migrated columns / newer tables),
    ``invalid`` (not SQLite, or not an NBA DB at all)."""

    def __init__(self, state: str, reason: str):
        super().__init__(f"{state}: {reason}")
        self.state = state
        self.reason = reason  # may contain the path: log it, never return it


def classify(exc: BaseException) -> str:
    """Map a ``DatabaseNotReadyError`` message to a serving state."""
    msg = str(exc)
    if "database file not found" in msg or "database file is empty" in msg:
        return "not_mounted"
    if "incomplete" in msg:
        # Missing CORE tables = not an NBA database (foreign file); anything
        # else (migrated columns, newer tables) = a real DB that predates the
        # current schema and needs one writer open to migrate.
        missing_tables = ""
        if "missing tables:" in msg:
            missing_tables = msg.split("missing tables:", 1)[1].split(";", 1)[0].split(")", 1)[0]
        names = {t.strip() for t in missing_tables.split(",") if t.strip()}
        return "invalid" if names & set(config.REQUIRED_TABLES) else "unmigrated"
    return "invalid"


@dataclass
class _Session:
    path: Path
    dm: object


_current: ContextVar[Optional[_Session]] = ContextVar("api_db_session", default=None)


class _Borrowed:
    """What data-layer code receives from ``DatabaseManager(...)`` inside a
    session: the shared read-only manager, minus the ability to close it."""

    __slots__ = ("_dm",)

    def __init__(self, dm):
        object.__setattr__(self, "_dm", dm)

    def __getattr__(self, name):
        return getattr(self._dm, name)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def close(self):  # the session owns the connection
        return None


class RequestScopedDatabaseManager(_RealDatabaseManager):
    """Drop-in for ``DatabaseManager`` in ``DATA_MODULES`` (see module doc)."""

    def __new__(cls, db_path="data/database/nba_data.db", read_only=False, *args, **kwargs):
        session = _current.get()
        if session is not None:
            if Path(db_path).resolve() == session.path:
                return _Borrowed(session.dm)
            # A different file inside a request: still never write.
            return _RealDatabaseManager(db_path, read_only=True)
        # Not in a request: the real class, exactly as the caller asked.
        return _RealDatabaseManager(db_path, *args, read_only=read_only, **kwargs)


def install() -> None:
    """Rebind ``DatabaseManager`` in the API's data modules (idempotent)."""
    for name in DATA_MODULES:
        mod = importlib.import_module(name)
        if getattr(mod, "DatabaseManager", None) is not RequestScopedDatabaseManager:
            mod.DatabaseManager = RequestScopedDatabaseManager


@contextmanager
def read_only_session(db_path: str) -> Iterator[str]:
    """Open ONE read-only connection to ``db_path`` for this request.

    Raises ``DbNotReady`` (classified) instead of creating or migrating
    anything. Nested use on the same path reuses the outer connection.
    """
    existing = _current.get()
    resolved = Path(db_path).resolve()
    if existing is not None and existing.path == resolved:
        yield db_path
        return
    try:
        dm = _RealDatabaseManager(db_path, read_only=True)
    except DatabaseNotReadyError as exc:
        state = classify(exc)
        logger.warning("database not ready (%s): %s", state, exc)
        raise DbNotReady(state, str(exc)) from exc
    token = _current.set(_Session(resolved, dm))
    try:
        yield db_path
    finally:
        _current.reset(token)
        dm.close()


def current_path() -> Optional[Path]:
    """The file the active session is serving (tests / diagnostics)."""
    s = _current.get()
    return s.path if s else None
