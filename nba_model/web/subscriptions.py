"""Pluggable subscription store (SQLite or Postgres).

Schema is intentionally minimal — one row per email — because every Stripe
event we care about (checkout completed / subscription updated / cancelled)
maps to a single update on this table. The Stripe webhook handler writes;
the Streamlit app reads.

Backend selection:

    SUBSCRIPTIONS_DB_URL = postgres://… or postgresql://…   -> Postgres
    (anything else, or unset)                               -> SQLite

The SQLite path is preserved for local dev (matches the previous default
file ``data/database/subscriptions.db``). The Postgres path is for cloud
deploys (Streamlit Cloud / Render) where the disk is ephemeral and the
existing SQLite file would vanish on every redeploy.

The public API — ``tier_for`` / ``upsert`` / ``record_stripe_event`` /
``lookup`` — is unchanged so ``webhook_app.py`` and ``auth.py`` continue
to work without edits. Security invariants (email validation, no PII in
logs, busy_timeout / connect timeout, fail-closed on garbage) hold across
both backends.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_SUBSCRIPTIONS_DB = "data/database/subscriptions.db"

# RFC 5321 + practical: rejects whitespace, multiple @, control chars, etc.
# We use this as a defense-in-depth check on top of trusting the OIDC provider.
# Local-part: alnum + . _ % + - ; domain: alnum + . - ; TLD: 2+ alpha.
_EMAIL_RE = re.compile(
    r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
)
_MAX_EMAIL_LEN = 254  # RFC 5321 limit


def _validate_email(email: str) -> str:
    """Reject obviously malformed emails before storing them as PKs.

    OIDC providers already enforce email validity, but we run this check too
    so a bug or compromised secret store can't poison the table with garbage
    keys (e.g. with embedded null bytes, quotes, or control characters).

    SECURITY: also rejects IDN domains encoded as punycode (`xn--...`).
    Punycode is the on-the-wire form of internationalized domains and is a
    common homograph-attack vector (e.g. `xn--pple-43d.com` renders as
    `аpple.com` with a Cyrillic 'а'). Since this app's user base is a US-
    focused paid sports product, no legitimate user needs an IDN, so the
    UX cost of rejecting them is effectively zero.
    """
    e = (email or "").strip().lower()
    if not e or len(e) > _MAX_EMAIL_LEN:
        raise ValueError(f"invalid email: length={len(e)}")
    if not _EMAIL_RE.match(e):
        raise ValueError("invalid email: format")
    # Reject punycode-encoded domain labels (xn--...). Match per-label so
    # `domain.xn--foo.com` is rejected too.
    domain = e.rsplit("@", 1)[-1]
    for label in domain.split("."):
        if label.startswith("xn--"):
            raise ValueError("invalid email: punycode/IDN not allowed")
    return e


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------

def _is_postgres_url(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.scheme in {"postgres", "postgresql", "postgresql+psycopg"}


def _selected_backend(db_path: Optional[str]) -> str:
    """Return 'postgres' or 'sqlite' based on env + caller override."""
    if db_path:
        # Explicit caller override always wins (used by the test suite).
        return "postgres" if _is_postgres_url(db_path) else "sqlite"
    url = os.environ.get("SUBSCRIPTIONS_DB_URL", "")
    return "postgres" if _is_postgres_url(url) else "sqlite"


def _resolved_target(db_path: Optional[str]) -> str:
    """Return the resolved DSN / file path for the active backend."""
    if db_path:
        return db_path
    url = os.environ.get("SUBSCRIPTIONS_DB_URL", "")
    if _is_postgres_url(url):
        return url
    return os.environ.get("SUBSCRIPTIONS_DB_PATH", DEFAULT_SUBSCRIPTIONS_DB)


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_subscriptions (
    email                  TEXT PRIMARY KEY,
    tier                   TEXT NOT NULL DEFAULT 'free',
    stripe_customer_id     TEXT,
    stripe_subscription_id TEXT,
    current_period_end     TEXT,
    last_event_type        TEXT,
    updated_at             TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS stripe_events (
    event_id     TEXT PRIMARY KEY,
    event_type   TEXT NOT NULL,
    received_at  TEXT NOT NULL DEFAULT (datetime('now')),
    payload      TEXT
);
"""


def _harden_db_file(path: str) -> None:
    """Best-effort `chmod 0600` so other unix users can't read the DB.

    Skipped on Windows (no posix mode) and silently ignored if the FS doesn't
    support it (e.g. some container overlays). Only runs when the DB file
    exists; new connections will create it then we re-call this.
    """
    try:
        if os.name != "posix":
            return
        if not Path(path).exists():
            return
        current = os.stat(path).st_mode & 0o777
        if current != 0o600:
            os.chmod(path, 0o600)
    except OSError:
        pass


_INIT_LOCK = threading.Lock()
_BOOTSTRAPPED_PATHS: set[str] = set()


def _bootstrap_sqlite(path: str) -> None:
    """Run the one-time PRAGMA-mode setup serialized across threads.

    SECURITY / RELIABILITY: changing `journal_mode=WAL` requires an exclusive
    lock; running it concurrently from N threads on a fresh DB is what made
    the stress suite's parallel-upsert test flake with `database is locked`.
    We do the journal-mode flip behind a process-wide `_INIT_LOCK` exactly
    once per path. The schema CREATE-IF-NOT-EXISTS is run on every connect
    (it's idempotent + cheap) so an operator dropping the table still gets
    it recreated automatically on the next read.
    """
    if path in _BOOTSTRAPPED_PATHS:
        return
    with _INIT_LOCK:
        if path in _BOOTSTRAPPED_PATHS:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        boot = sqlite3.connect(path, isolation_level=None, timeout=10.0)
        try:
            boot.execute("PRAGMA journal_mode=WAL")
            boot.execute("PRAGMA synchronous=NORMAL")
        finally:
            boot.close()
        _harden_db_file(path)
        _BOOTSTRAPPED_PATHS.add(path)


@contextmanager
def _connect_sqlite(path: str) -> Iterator[sqlite3.Connection]:
    _bootstrap_sqlite(path)
    # SECURITY: 5-second busy_timeout. Without this, two concurrent webhook
    # events for the same email (or even the Streamlit reader contending
    # with the webhook writer) raise `database is locked` immediately. Real
    # Stripe deliveries can race - the stress suite caught this. 5s lets the
    # writer block briefly instead of failing the whole request.
    conn = sqlite3.connect(path, isolation_level=None, timeout=5.0)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        try:
            conn.enable_load_extension(False)  # type: ignore[attr-defined]
        except (AttributeError, sqlite3.NotSupportedError):
            pass
        conn.executescript(_SQLITE_SCHEMA)
        yield conn
    finally:
        conn.close()


def _sqlite_tier_for(path: str, normalized_email: str) -> Optional[tuple]:
    with _connect_sqlite(path) as conn:
        return conn.execute(
            "SELECT tier, current_period_end FROM user_subscriptions "
            "WHERE email = ?",
            (normalized_email,),
        ).fetchone()


def _sqlite_upsert(
    path: str,
    *,
    email: str, tier: str, stripe_customer_id, stripe_subscription_id,
    current_period_end, last_event_type, now: str,
) -> None:
    with _connect_sqlite(path) as conn:
        conn.execute(
            """
            INSERT INTO user_subscriptions (
                email, tier, stripe_customer_id, stripe_subscription_id,
                current_period_end, last_event_type, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                tier                   = excluded.tier,
                stripe_customer_id     = COALESCE(excluded.stripe_customer_id,
                                                  user_subscriptions.stripe_customer_id),
                stripe_subscription_id = COALESCE(excluded.stripe_subscription_id,
                                                  user_subscriptions.stripe_subscription_id),
                current_period_end     = COALESCE(excluded.current_period_end,
                                                  user_subscriptions.current_period_end),
                last_event_type        = COALESCE(excluded.last_event_type,
                                                  user_subscriptions.last_event_type),
                updated_at             = excluded.updated_at
            """,
            (email, tier, stripe_customer_id, stripe_subscription_id,
             current_period_end, last_event_type, now),
        )


def _sqlite_record_event(path: str, event_id: str, event_type: str, payload: str) -> bool:
    with _connect_sqlite(path) as conn:
        try:
            conn.execute(
                "INSERT INTO stripe_events (event_id, event_type, payload) "
                "VALUES (?, ?, ?)",
                (event_id, event_type, payload),
            )
            return True
        except sqlite3.IntegrityError:
            return False


def _sqlite_lookup(path: str, normalized_email: str) -> Optional[dict]:
    with _connect_sqlite(path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM user_subscriptions WHERE email = ?",
            (normalized_email,),
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Postgres backend
# ---------------------------------------------------------------------------

_POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_subscriptions (
    email                  TEXT PRIMARY KEY,
    tier                   TEXT NOT NULL DEFAULT 'free',
    stripe_customer_id     TEXT,
    stripe_subscription_id TEXT,
    current_period_end     TEXT,
    last_event_type        TEXT,
    updated_at             TEXT NOT NULL DEFAULT (now()::text)
);

CREATE TABLE IF NOT EXISTS stripe_events (
    event_id     TEXT PRIMARY KEY,
    event_type   TEXT NOT NULL,
    received_at  TEXT NOT NULL DEFAULT (now()::text),
    payload      TEXT
);
"""

# Postgres analog to SQLite's busy_timeout/synchronous tuning:
#   - statement_timeout    => abort runaway upserts after 5s (was busy_timeout)
#   - lock_timeout         => fail fast instead of blocking on row locks
#   - idle_in_transaction_session_timeout => abandon idle transactions
# These run as session GUCs on each connect; safer than tampering with the
# instance config in a hosted Postgres.
_POSTGRES_SESSION_TUNING = (
    "SET statement_timeout = '5s'",
    "SET lock_timeout = '5s'",
    "SET idle_in_transaction_session_timeout = '30s'",
)

_PG_BOOTSTRAPPED_DSNS: set[str] = set()


def _import_psycopg():
    """Return the ``psycopg`` (v3) module, raising a clear message if missing."""
    try:
        import psycopg  # type: ignore[import]
        return psycopg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Postgres backend selected (SUBSCRIPTIONS_DB_URL points at "
            "postgres://) but the `psycopg` driver is not installed. "
            "Install it with: pip install 'psycopg[binary]>=3.1'"
        ) from exc


def _bootstrap_postgres(dsn: str) -> None:
    if dsn in _PG_BOOTSTRAPPED_DSNS:
        return
    with _INIT_LOCK:
        if dsn in _PG_BOOTSTRAPPED_DSNS:
            return
        psycopg = _import_psycopg()
        with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(_POSTGRES_SCHEMA)
        _PG_BOOTSTRAPPED_DSNS.add(dsn)


@contextmanager
def _connect_postgres(dsn: str):
    _bootstrap_postgres(dsn)
    psycopg = _import_psycopg()
    conn = psycopg.connect(dsn, autocommit=True, connect_timeout=5)
    try:
        with conn.cursor() as cur:
            for stmt in _POSTGRES_SESSION_TUNING:
                cur.execute(stmt)
        yield conn
    finally:
        conn.close()


def _pg_tier_for(dsn: str, normalized_email: str):
    with _connect_postgres(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT tier, current_period_end FROM user_subscriptions "
            "WHERE email = %s",
            (normalized_email,),
        )
        return cur.fetchone()


def _pg_upsert(
    dsn: str,
    *,
    email: str, tier: str, stripe_customer_id, stripe_subscription_id,
    current_period_end, last_event_type, now: str,
) -> None:
    with _connect_postgres(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO user_subscriptions (
                email, tier, stripe_customer_id, stripe_subscription_id,
                current_period_end, last_event_type, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (email) DO UPDATE SET
                tier                   = EXCLUDED.tier,
                stripe_customer_id     = COALESCE(EXCLUDED.stripe_customer_id,
                                                  user_subscriptions.stripe_customer_id),
                stripe_subscription_id = COALESCE(EXCLUDED.stripe_subscription_id,
                                                  user_subscriptions.stripe_subscription_id),
                current_period_end     = COALESCE(EXCLUDED.current_period_end,
                                                  user_subscriptions.current_period_end),
                last_event_type        = COALESCE(EXCLUDED.last_event_type,
                                                  user_subscriptions.last_event_type),
                updated_at             = EXCLUDED.updated_at
            """,
            (email, tier, stripe_customer_id, stripe_subscription_id,
             current_period_end, last_event_type, now),
        )


def _pg_record_event(dsn: str, event_id: str, event_type: str, payload: str) -> bool:
    psycopg = _import_psycopg()
    with _connect_postgres(dsn) as conn, conn.cursor() as cur:
        try:
            cur.execute(
                "INSERT INTO stripe_events (event_id, event_type, payload) "
                "VALUES (%s, %s, %s)",
                (event_id, event_type, payload),
            )
            return True
        except psycopg.errors.UniqueViolation:
            return False


def _pg_lookup(dsn: str, normalized_email: str) -> Optional[dict]:
    with _connect_postgres(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT email, tier, stripe_customer_id, stripe_subscription_id, "
            "current_period_end, last_event_type, updated_at "
            "FROM user_subscriptions WHERE email = %s",
            (normalized_email,),
        )
        row = cur.fetchone()
        if not row:
            return None
        cols = [d.name for d in cur.description]
    return dict(zip(cols, row))


# ---------------------------------------------------------------------------
# Public API (backend-agnostic)
# ---------------------------------------------------------------------------

def tier_for(email: str, db_path: Optional[str] = None) -> str:
    """Return 'free' or 'premium' for the given email.

    Treats invalid emails as 'free' (fail closed) so an attacker who manages
    to inject a malformed email into the chain doesn't accidentally get
    elevated permissions due to a partial match.
    """
    if not email:
        return "free"
    try:
        normalized = _validate_email(email)
    except ValueError:
        return "free"

    backend = _selected_backend(db_path)
    target = _resolved_target(db_path)
    row = (_pg_tier_for(target, normalized) if backend == "postgres"
           else _sqlite_tier_for(target, normalized))
    if not row:
        return "free"
    tier, period_end = row
    if tier != "premium":
        return "free"
    # Premium with an end date in the past is treated as expired.
    if period_end:
        try:
            cutoff = datetime.fromisoformat(str(period_end).replace("Z", "+00:00"))
            if cutoff.tzinfo is None:
                cutoff = cutoff.replace(tzinfo=timezone.utc)
            if cutoff < datetime.now(timezone.utc):
                return "free"
        except ValueError:
            pass
    return "premium"


def upsert(
    email: str,
    *,
    tier: str,
    stripe_customer_id: Optional[str] = None,
    stripe_subscription_id: Optional[str] = None,
    current_period_end: Optional[str] = None,
    last_event_type: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Insert or update the subscription record for an email."""
    if tier not in {"free", "premium"}:
        raise ValueError(f"tier must be 'free' or 'premium', got {tier!r}")
    normalized_email = _validate_email(email)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    backend = _selected_backend(db_path)
    target = _resolved_target(db_path)
    fn = _pg_upsert if backend == "postgres" else _sqlite_upsert
    fn(
        target,
        email=normalized_email,
        tier=tier,
        stripe_customer_id=stripe_customer_id,
        stripe_subscription_id=stripe_subscription_id,
        current_period_end=current_period_end,
        last_event_type=last_event_type,
        now=now,
    )


def record_stripe_event(
    event_id: str,
    event_type: str,
    payload: str,
    db_path: Optional[str] = None,
) -> bool:
    """Record a Stripe event id for idempotency. Returns False if already seen."""
    backend = _selected_backend(db_path)
    target = _resolved_target(db_path)
    if backend == "postgres":
        return _pg_record_event(target, event_id, event_type, payload)
    return _sqlite_record_event(target, event_id, event_type, payload)


def lookup(email: str, db_path: Optional[str] = None) -> Optional[dict]:
    try:
        normalized = _validate_email(email)
    except ValueError:
        return None
    backend = _selected_backend(db_path)
    target = _resolved_target(db_path)
    if backend == "postgres":
        return _pg_lookup(target, normalized)
    return _sqlite_lookup(target, normalized)


def _all_subscription_rows(db_path: Optional[str] = None) -> list:
    """Return every subscription row as a list of dicts (both backends)."""
    backend = _selected_backend(db_path)
    target = _resolved_target(db_path)
    cols = ("email", "tier", "last_event_type", "updated_at",
            "current_period_end")
    if backend == "postgres":
        with _connect_postgres(target) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT email, tier, last_event_type, updated_at, "
                "current_period_end FROM user_subscriptions")
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    with _connect_sqlite(target) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT email, tier, last_event_type, updated_at, "
            "current_period_end FROM user_subscriptions").fetchall()
    return [dict(r) for r in rows]


def _resolve_price_monthly(price_monthly: Optional[float]) -> Optional[float]:
    if price_monthly is not None:
        return float(price_monthly)
    raw = os.environ.get("STRIPE_PRICE_MONTHLY_USD")
    if raw:
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _aggregate_from_rows(rows: list, price_monthly: Optional[float] = None) -> dict:
    """Pure aggregation for the admin dashboard (testable without a DB)."""
    n_premium = sum(1 for r in rows if str(r.get("tier")) == "premium")
    n_total = len(rows)
    n_free = n_total - n_premium
    churned = sum(
        1 for r in rows
        if "deleted" in str(r.get("last_event_type") or "").lower()
    )
    mrr = round(n_premium * float(price_monthly), 2) if price_monthly else None
    return {
        "premium": n_premium,
        "free": n_free,
        "total": n_total,
        "churned": churned,
        "mrr_estimate": mrr,
        "price_monthly": float(price_monthly) if price_monthly else None,
    }


def aggregate_stats(
    db_path: Optional[str] = None,
    price_monthly: Optional[float] = None,
) -> dict:
    """Subscriber counts + MRR estimate + recent churn for the admin view.

    ``price_monthly`` falls back to the ``STRIPE_PRICE_MONTHLY_USD`` env var;
    when neither is set ``mrr_estimate`` is ``None`` (we can't infer revenue).
    """
    rows = _all_subscription_rows(db_path)
    return _aggregate_from_rows(rows, _resolve_price_monthly(price_monthly))


def selected_backend(db_path: Optional[str] = None) -> str:
    """Inspect which backend would be used for the given override (or env).

    Exposed for the deployment readiness check (`docs/DEPLOYMENT.md` shows
    a one-liner that prints this) and for tests that need to assert the
    URL-based dispatch lights up correctly without actually connecting.
    """
    return _selected_backend(db_path)
