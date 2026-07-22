"""Runtime configuration for the read-only API service.

The single piece of real config is the path to the SQLite database. It
resolves, in order:
  1. the ``NBA_DB_PATH`` environment variable (absolute or relative), then
  2. the canonical project DB at ``<repo>/data/database/nba_data.db``.

Everything else in this service is stateless.
"""
from __future__ import annotations

import os
from pathlib import Path

# ``api/config.py`` -> ``api/`` -> repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "database" / "nba_data.db"

# Cache the read-heavy endpoints briefly. The underlying DB is refreshed by an
# hourly ETL, so a short client/proxy cache is safe and keeps repeated slate
# renders cheap without ever serving stale-by-more-than-a-poll data.
CACHE_MAX_AGE_SECONDS = 60


def get_db_path() -> str:
    """Absolute path to the SQLite DB this service reads from."""
    override = os.environ.get("NBA_DB_PATH", "").strip()
    if override:
        return str(Path(override).expanduser())
    return str(DEFAULT_DB_PATH)


def db_exists(db_path: str | None = None) -> bool:
    return Path(db_path or get_db_path()).is_file()
