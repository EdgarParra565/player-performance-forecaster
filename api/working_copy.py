"""Serve a disposable working copy when the source DB is read-only.

Why: ``DatabaseManager.__init__`` (nba_model/data — not ours) runs schema
migrations on EVERY open (e.g. ``ALTER TABLE web_team_lines ADD COLUMN
game_date``). On a read-only bind mount (``./data/database:/data:ro``) those
writes fail with "attempt to write a readonly database", taking down every
endpoint. Mounting read-write would let the API write into the ETL's live
file, which is worse.

So when the source file is not writable, the API serves
``$API_DB_CACHE_DIR/nba_data.db`` (default ``/tmp/nba-flagship``): a copy made
with SQLite's online ``.backup`` from a read-only connection, refreshed
whenever the source's (mtime, size) changes, swapped in with ``os.replace``.
The data layer's open-time migrations then land in the copy, never the host
file. Proper fix: a read-only open mode in db_manager (CROSS-BOUNDARY handoff,
review_findings_web.md).

``API_DB_WORKING_COPY``: ``auto`` (default: copy only if the source isn't
writable) | ``always`` | ``never``.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
from pathlib import Path
from threading import Lock

logger = logging.getLogger("api.working_copy")

_LOCK = Lock()
_state: dict[str, tuple[float, int]] = {}  # source path -> (mtime, size) copied


def _mode() -> str:
    m = os.environ.get("API_DB_WORKING_COPY", "auto").strip().lower()
    return m if m in {"auto", "always", "never"} else "auto"


def _cache_dir() -> Path:
    return Path(os.environ.get("API_DB_CACHE_DIR", "/tmp/nba-flagship"))


def source_is_writable(source: str) -> bool:
    p = Path(source)
    # access(2) reports EROFS for read-only mounts as well as permission bits.
    return os.access(p, os.W_OK) and os.access(p.parent, os.W_OK)


def needs_copy(source: str) -> bool:
    mode = _mode()
    if mode == "never":
        return False
    if mode == "always":
        return True
    return not source_is_writable(source)


def copy_path(source: str) -> str:
    return str(_cache_dir() / Path(source).name)


def serving_path(source: str) -> str:
    """Path the data layer should open for ``source`` (refreshing the copy)."""
    if not needs_copy(source):
        return source
    try:
        st = Path(source).stat()
    except OSError:
        return source  # caller's preflight reports not-mounted
    sig = (st.st_mtime, st.st_size)
    dest = copy_path(source)
    if _state.get(source) == sig and Path(dest).is_file():
        return dest
    with _LOCK:
        if _state.get(source) == sig and Path(dest).is_file():
            return dest
        refresh(source, dest)
        _state[source] = sig
    return dest


def refresh(source: str, dest: str) -> None:
    """Consistent copy via online backup from a READ-ONLY connection."""
    d = Path(dest)
    d.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{d.name}.", dir=str(d.parent))
    os.close(fd)
    try:
        src = sqlite3.connect(f"{Path(source).resolve().as_uri()}?mode=ro", uri=True, timeout=10)
        try:
            out = sqlite3.connect(tmp)
            try:
                src.backup(out)
            finally:
                out.close()
        finally:
            src.close()
        os.replace(tmp, d)
        logger.info("working copy refreshed from read-only source")
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def reset() -> None:
    """Forget copy signatures (tests)."""
    with _LOCK:
        _state.clear()
