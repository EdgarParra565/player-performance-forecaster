"""SQLite-backed subscription store.

Schema is intentionally minimal - one row per email - because every Stripe
event we care about (checkout completed / subscription updated / cancelled)
maps to a single update on this table. The Stripe webhook handler writes;
the Streamlit app reads.

We use a separate file (`data/database/subscriptions.db`) from the main
nba_data.db so the production deployment can mount the subscription DB as a
persistent volume without giving the Streamlit container write access to the
analytics DB.
"""

from __future__ import annotations

import os
import re
import sqlite3
import stat
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

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

_SCHEMA = """
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


def _db_path() -> str:
    return os.environ.get("SUBSCRIPTIONS_DB_PATH", DEFAULT_SUBSCRIPTIONS_DB)


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


_INIT_LOCK = __import__("threading").Lock()
_BOOTSTRAPPED_PATHS: set[str] = set()


def _bootstrap_db(path: str) -> None:
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
def _connect(db_path: Optional[str] = None) -> Iterator[sqlite3.Connection]:
    path = db_path or _db_path()
    _bootstrap_db(path)
    # SECURITY: 5-second busy_timeout. Without this, two concurrent webhook
    # events for the same email (or even the Streamlit reader contending
    # with the webhook writer) raise `database is locked` immediately. Real
    # Stripe deliveries can race - the stress suite caught this. 5s lets the
    # writer block briefly instead of failing the whole request.
    conn = sqlite3.connect(path, isolation_level=None, timeout=5.0)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        # Block any external SQL extensions from being loaded over our handle.
        try:
            conn.enable_load_extension(False)  # type: ignore[attr-defined]
        except (AttributeError, sqlite3.NotSupportedError):
            pass
        # `CREATE TABLE IF NOT EXISTS` is idempotent and cheap; running it on
        # every connect lets operators drop a table for emergency reset and
        # have it auto-recreated on the next read.
        conn.executescript(_SCHEMA)
        yield conn
    finally:
        conn.close()


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
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT tier, current_period_end FROM user_subscriptions "
            "WHERE email = ?",
            (normalized,),
        ).fetchone()
    if not row:
        return "free"
    tier, period_end = row
    if tier != "premium":
        return "free"
    # Premium with an end date in the past is treated as expired.
    if period_end:
        try:
            cutoff = datetime.fromisoformat(period_end.replace("Z", "+00:00"))
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
    with _connect(db_path) as conn:
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
            (normalized_email, tier, stripe_customer_id,
             stripe_subscription_id, current_period_end, last_event_type, now),
        )


def record_stripe_event(
    event_id: str,
    event_type: str,
    payload: str,
    db_path: Optional[str] = None,
) -> bool:
    """Record a Stripe event id for idempotency. Returns False if already seen."""
    with _connect(db_path) as conn:
        try:
            conn.execute(
                "INSERT INTO stripe_events (event_id, event_type, payload) "
                "VALUES (?, ?, ?)",
                (event_id, event_type, payload),
            )
            return True
        except sqlite3.IntegrityError:
            return False


def lookup(email: str, db_path: Optional[str] = None) -> Optional[dict]:
    try:
        normalized = _validate_email(email)
    except ValueError:
        return None
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM user_subscriptions WHERE email = ?",
            (normalized,),
        ).fetchone()
    return dict(row) if row else None
