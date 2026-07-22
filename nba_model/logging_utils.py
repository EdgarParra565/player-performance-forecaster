"""Shared structured logging for the NBA data / evaluation pipelines.

Standard-library ``logging`` only — no new dependencies. One place configures
the root logger; modules grab a child logger with :func:`get_logger`::

    from nba_model.logging_utils import configure_logging, get_logger

    logger = get_logger(__name__)

    def main():
        configure_logging(file_prefix="daily_etl")   # once, at process start
        logger.info("step complete",
                    extra={"step": "game_logs", "rows": 1234, "duration_ms": 87})

Output split:
  * **Console** — human-readable (``ts [LEVEL] name - message``).
  * **File** (optional, when ``file_prefix`` is given) — **JSON-lines**: one
    JSON object per record with a UTC ISO-8601 ``ts``, ``level``, ``logger``,
    ``message``, plus any fields passed through ``extra=``. This is what makes
    the hourly scheduler's output machine-parseable for later alerting.

Level is env-configurable via ``LOG_LEVEL`` (default ``INFO``).

SECURITY (docs/SECURITY.md §A09 — hard contract): never pass payload bodies,
email addresses, customer ids, API keys, cookies, or the *contents* of
auth-state files to the logger — not as a positional/format arg and not as an
``extra=`` field. This module intentionally performs **no** redaction; keeping
secrets out of log calls is the caller's responsibility.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_LOG_DIR = "nba_model/data/logs"
_CONSOLE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"

# Attributes present on a bare LogRecord are the "standard" ones; anything
# else a caller attached (via ``extra=``) is a structured field we serialize.
_STD_ATTRS = frozenset(vars(logging.makeLogRecord({})).keys()) | {
    "message", "asctime", "taskName",
}

# Marks the JSON-lines file handler this module installs, so re-configuration
# can find (and report) the active log file without touching foreign handlers.
_JSONL_TAG = "_nba_jsonl_handler"


class JsonLinesFormatter(logging.Formatter):
    """Render a ``LogRecord`` as one line of JSON (message + any extras)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": _iso_utc(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STD_ATTRS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # default=str keeps a stray non-JSON value (e.g. a Path) from crashing
        # the log call — logging must never take down the pipeline.
        return json.dumps(payload, default=str)


def _iso_utc(epoch_seconds: float) -> str:
    """UTC ISO-8601 with a trailing ``Z`` (e.g. ``2026-07-21T14:03:11.482Z``)."""
    return (
        datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _resolve_level(level) -> int:
    """Coerce a level (int, name, or ``None``→``$LOG_LEVEL``) to a logging int."""
    if level is None:
        level = os.environ.get("LOG_LEVEL") or "INFO"
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(str(level).strip().upper())
    return resolved if isinstance(resolved, int) else logging.INFO


def _existing_jsonl_path(root: logging.Logger) -> Optional[str]:
    for handler in root.handlers:
        if getattr(handler, _JSONL_TAG, False):
            return getattr(handler, "baseFilename", None)
    return None


def configure_logging(
    *,
    file_prefix: Optional[str] = None,
    log_dir: str = DEFAULT_LOG_DIR,
    level=None,
    console: bool = True,
    force: bool = False,
) -> Optional[str]:
    """Configure root logging once: human console + optional JSON-lines file.

    ``file_prefix`` (when given) creates ``<log_dir>/<file_prefix>_<UTC ts>.log``
    and returns its path; otherwise returns ``None``. ``level`` defaults to
    ``$LOG_LEVEL`` (else ``INFO``).

    Idempotent by default: if the root logger already has handlers and
    ``force`` is ``False``, this does nothing and returns the path of an
    already-installed JSON-lines file (or ``None``) — so tests / callers that
    pre-configured logging aren't disturbed. Pass ``force=True`` to tear down
    existing handlers and reinstall (used by the hourly runner, which owns its
    process).
    """
    root = logging.getLogger()
    resolved_level = _resolve_level(level)

    if root.handlers and not force:
        return _existing_jsonl_path(root)

    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass

    root.setLevel(resolved_level)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
        console_handler.setLevel(resolved_level)
        root.addHandler(console_handler)

    file_path: Optional[str] = None
    if file_prefix:
        out_dir = Path(log_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        file_path = str(out_dir / f"{file_prefix}_{ts}.log")
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(JsonLinesFormatter())
        file_handler.setLevel(resolved_level)
        setattr(file_handler, _JSONL_TAG, True)
        root.addHandler(file_handler)

    return file_path


def get_logger(name: str) -> logging.Logger:
    """Return the module logger for ``name`` (a plain ``logging.Logger``)."""
    return logging.getLogger(name)
