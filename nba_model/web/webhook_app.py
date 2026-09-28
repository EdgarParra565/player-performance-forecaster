"""FastAPI app that handles Stripe webhooks and updates the subscriptions DB.

Run with:
    .venv/bin/python3 -m uvicorn nba_model.web.webhook_app:app \\
        --host 0.0.0.0 --port 8081

Configure Stripe to send events to: https://your-domain/stripe/webhook
We listen for: checkout.session.completed, customer.subscription.{created,
updated,deleted}, invoice.paid, invoice.payment_failed.

Required env vars (or .streamlit/secrets.toml under [stripe]):
    STRIPE_WEBHOOK_SECRET  (whsec_...)
    STRIPE_SECRET_KEY      (sk_live_... or sk_test_...)
    SUBSCRIPTIONS_DB_PATH  (optional; defaults to data/database/subscriptions.db)
    WEBHOOK_TRUSTED_HOSTS  (optional CSV; defaults to "*" -- set this in prod)
    WEBHOOK_TOLERANCE_SECONDS (optional; default 300 = Stripe's recommendation)

Security model:
    1. Body size capped at MAX_BODY_BYTES (default 256 KiB) - rejects DoS-by-payload.
    2. `stripe.Webhook.construct_event` verifies HMAC signature AND rejects
       events older than `tolerance` seconds (replay protection).
    3. `record_stripe_event` is idempotent on event_id - retries by Stripe
       (or attacker replays inside the tolerance window) are no-ops.
    4. TrustedHostMiddleware optionally restricts which Host headers are
       accepted (set WEBHOOK_TRUSTED_HOSTS).
    5. We log only event_id + event_type + email-presence flag - never
       payload bodies or emails to keep PII out of stdout/log files.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("stripe_webhook")
logger.setLevel(logging.INFO)

try:
    import stripe  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "stripe SDK not installed. Run: pip install stripe"
    ) from exc

from collections import deque
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import Response

from nba_model.web import subscriptions


# ---- Security headers + rate limiter ----------------------------------------
#
# Per OWASP + the 2026 HTTP-security-headers checklist we attach the following
# to every response:
#   - Strict-Transport-Security: force HTTPS for 2 years; include subdomains.
#   - Content-Security-Policy:   default-src 'self'; lock down everything else.
#                                The webhook serves only JSON, so this is
#                                essentially a belt-and-suspenders header to
#                                neutralise XSS if a future endpoint ever
#                                returns HTML by accident.
#   - X-Content-Type-Options:    nosniff (defeats MIME sniffing)
#   - X-Frame-Options:           DENY (no embedding)
#   - Referrer-Policy:           no-referrer (don't leak target URL)
#   - Permissions-Policy:        deny everything we'd never legitimately use
#   - Cross-Origin-*:            isolate aggressively
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach the OWASP-recommended response headers to every reply."""

    HEADERS = {
        "Strict-Transport-Security":
            "max-age=63072000; includeSubDomains; preload",
        "Content-Security-Policy":
            "default-src 'none'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'none'",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Permissions-Policy":
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
            "magnetometer=(), microphone=(), payment=(), usb=()",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
        # Servers shouldn't advertise themselves any more than necessary.
        "Server": "webhook",
    }

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        for k, v in self.HEADERS.items():
            response.headers[k] = v
        return response


class IPRateLimiter:
    """Simple sliding-window per-IP rate limiter (in-memory).

    Stripe legitimately bursts events during batch operations, so we keep the
    default headroom generous (120 reqs / 60s per IP) but tunable via env.
    For a multi-replica deploy swap this out for Redis / a shared store; the
    in-memory version is fine on Streamlit Community Cloud + a single Render
    webhook host because Stripe sends serially from a small IP range.

    Returns a 429 to the client; the Stripe dashboard will retry with backoff.
    """

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = int(max_requests)
        self.window = float(window_seconds)
        self._buckets: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str, now: float) -> bool:
        with self._lock:
            bucket = self._buckets.setdefault(key, deque())
            cutoff = now - self.window
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                return False
            bucket.append(now)
            return True

    def blocked(self, key: str, now: float) -> bool:
        """True if ``key`` is over the limit, WITHOUT recording a hit."""
        with self._lock:
            bucket = self._buckets.get(key)
            if not bucket:
                return False
            cutoff = now - self.window
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            return len(bucket) >= self.max_requests

    def gc(self, now: float) -> None:
        """Drop empty buckets so memory doesn't grow unbounded over weeks."""
        with self._lock:
            cutoff = now - self.window
            stale = [
                k for k, b in self._buckets.items()
                if not b or b[-1] < cutoff
            ]
            for k in stale:
                self._buckets.pop(k, None)


_rate_limit_max = int(os.environ.get("WEBHOOK_RATE_MAX", "120"))
_rate_limit_window = float(os.environ.get("WEBHOOK_RATE_WINDOW_SECONDS", "60"))
_rate_limiter = IPRateLimiter(_rate_limit_max, _rate_limit_window)


def _client_key(request: Request) -> str:
    """Pick the most reliable client identifier we can.

    Behind a trusted proxy (Render / Railway / Fly) we use X-Forwarded-For's
    first IP. Otherwise fall back to the connecting socket. Caller is
    responsible for setting WEBHOOK_TRUSTED_PROXY_HOPS if there's more than
    one trusted hop, so an attacker can't spoof an IP via header injection.
    """
    trusted_hops = int(os.environ.get("WEBHOOK_TRUSTED_PROXY_HOPS", "0"))
    if trusted_hops > 0:
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            ips = [p.strip() for p in xff.split(",") if p.strip()]
            if ips:
                # Pick the leftmost-after-trusted-hops IP.
                idx = max(0, len(ips) - trusted_hops)
                return ips[idx]
    return request.client.host if request.client else "unknown"

# Pin the Stripe API version so an upstream upgrade can't silently change
# event payload shapes under us. Update intentionally + re-test.
stripe.api_version = "2024-11-20.acacia"

# 256 KiB - real Stripe events are typically < 30 KiB. Anything bigger is a
# misconfigured client or a deliberate DoS attempt; reject it with 413.
MAX_BODY_BYTES = 256 * 1024

# Stripe recommends 300s for the construct_event tolerance. Configurable
# via env so a security incident can shorten it.
DEFAULT_TOLERANCE_SECONDS = 300

app = FastAPI(
    title="NBA Probability Model - Stripe webhook",
    docs_url=None,        # disable interactive docs in prod
    redoc_url=None,       # ditto
    openapi_url=None,     # don't expose schema either
)

_trusted_hosts_env = os.environ.get("WEBHOOK_TRUSTED_HOSTS", "").strip()
if _trusted_hosts_env:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[h.strip() for h in _trusted_hosts_env.split(",") if h.strip()],
    )

app.add_middleware(SecurityHeadersMiddleware)


def _webhook_secret() -> str:
    val = os.environ.get("STRIPE_WEBHOOK_SECRET")
    if not val:
        raise HTTPException(
            status_code=500,
            detail="STRIPE_WEBHOOK_SECRET is not configured",
        )
    return val


def _stripe_api_key() -> str:
    val = os.environ.get("STRIPE_SECRET_KEY")
    if not val:
        raise HTTPException(
            status_code=500,
            detail="STRIPE_SECRET_KEY is not configured",
        )
    return val


def _iso_from_unix(ts: Optional[int]) -> Optional[str]:
    if not ts:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat(
        timespec="seconds"
    )


def _safe_get(obj, key, default=None):
    """Dict-or-StripeObject-safe getter.

    SECURITY note: Stripe SDK 15.x's `StripeObject` does NOT subclass dict
    and intentionally does NOT expose `.get()`. Calling `obj.get("x")` on a
    StripeObject raises AttributeError. We learned this the hard way during
    security stress-testing. This helper papers over the difference.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return obj[key]
    except (KeyError, TypeError, AttributeError):
        return default


class EmailLookupError(RuntimeError):
    """Stripe Customer lookup failed transiently; the event must be retried."""


def _email_from_event(obj) -> Optional[str]:
    """Pull a customer email from the most likely fields on the event object.

    Returns None only when the event genuinely carries no customer. A FAILED
    ``Customer.retrieve`` (network, auth, missing STRIPE_SECRET_KEY) raises
    ``EmailLookupError`` so the caller can 503 and let Stripe retry, instead
    of acknowledging and silently dropping e.g. a cancellation.
    """
    email = _safe_get(obj, "customer_email") or _safe_get(
        _safe_get(obj, "customer_details", {}) or {}, "email",
    )
    if email:
        return str(email).strip().lower()
    customer_id = _safe_get(obj, "customer")
    if not customer_id:
        return None
    try:
        stripe.api_key = _stripe_api_key()
        cust = stripe.Customer.retrieve(customer_id)
    except Exception as exc:  # noqa: BLE001 — any failure means "retry later"
        # Type only: exception text can echo request details.
        logger.warning("webhook: customer lookup failed (%s)", type(exc).__name__)
        raise EmailLookupError(type(exc).__name__) from exc
    return str(_safe_get(cust, "email") or "").strip().lower() or None


def _event_subscription_id(obj) -> Optional[str]:
    """The ``sub_...`` id an event refers to (never an invoice / session id)."""
    sub = _safe_get(obj, "subscription")
    if isinstance(sub, str) and sub:
        return sub
    oid = _safe_get(obj, "id")
    if isinstance(oid, str) and oid.startswith("sub_"):
        return oid
    return None


def alert_payment_failed(email, *, poster=None) -> dict:
    """Best-effort admin alert when a known paying user's payment fails.

    Posts a minimal JSON to ``BILLING_ALERT_WEBHOOK_URL`` (Slack/Discord/
    generic) so the operator can reach out before the subscription lapses.
    The email is sent only to that operator-controlled endpoint — never logged
    here (the webhook's no-PII-in-logs policy still holds). Never raises.
    ``poster`` is injected for tests.
    """
    url = os.environ.get("BILLING_ALERT_WEBHOOK_URL")
    if not url or not email:
        return {"sent": False, "reason": "no_webhook_or_email"}
    try:
        if poster is None:
            import requests
            poster = requests.post
        resp = poster(
            url,
            json={"text": f"Payment failed for paying user: {email}",
                  "event": "invoice.payment_failed", "email": email},
            timeout=10,
        )
        return {"sent": True, "status_code": getattr(resp, "status_code", None)}
    except Exception as exc:  # noqa: BLE001 — alerting must never break the webhook
        return {"sent": False, "reason": f"post_failed: {type(exc).__name__}"}


# Stripe subscription statuses that entitle the customer to premium. Every
# other status (past_due, unpaid, incomplete, incomplete_expired, canceled,
# paused, or anything Stripe adds later) fails closed to free.
ENTITLED_SUBSCRIPTION_STATUSES = frozenset({"active", "trialing"})


def tier_for_event(event_type: str, obj) -> Optional[str]:
    """Map a verified Stripe event to the tier it implies, or None (no change).

    SECURITY: ``customer.subscription.updated`` also fires on transitions INTO
    past_due / unpaid / incomplete, and ``checkout.session.completed`` fires
    for sessions whose payment is still pending (async methods) — so premium
    is granted only on the object's actual status, never on the event type
    alone.
    """
    if event_type == "checkout.session.completed":
        # 'unpaid' (async payment pending) and 'no_payment_required' make no
        # change here; the subscription.created/updated event that follows
        # carries the authoritative status (incl. 'trialing').
        if str(_safe_get(obj, "payment_status") or "") == "paid":
            return "premium"
        return None
    if event_type in {
        "customer.subscription.created",
        "customer.subscription.updated",
    }:
        status = str(_safe_get(obj, "status") or "").strip().lower()
        return "premium" if status in ENTITLED_SUBSCRIPTION_STATUSES else "free"
    if event_type == "invoice.paid":
        return "premium"
    if event_type in {
        "customer.subscription.deleted",
        "invoice.payment_failed",
    }:
        return "free"
    return None


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe. Intentionally returns no app version or build info to
    avoid information disclosure to opportunistic scanners."""
    return {"status": "ok"}


def _tolerance_seconds() -> int:
    raw = os.environ.get("WEBHOOK_TOLERANCE_SECONDS")
    if not raw:
        return DEFAULT_TOLERANCE_SECONDS
    try:
        val = int(raw)
        return max(60, min(val, 600))
    except ValueError:
        return DEFAULT_TOLERANCE_SECONDS


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request) -> dict:
    # 0. Rate-limit per source IP before doing any work. Stripe retries on
    #    429 with exponential backoff so this is safe to enforce.
    #    Only REJECTED requests (bad/missing signature, bad payload, oversize,
    #    slow body) count toward the bucket: behind a proxy every request can
    #    share one key, and counting valid Stripe deliveries let an unsigned
    #    flood starve real events with 429s.
    import time
    now = time.time()
    client_key = _client_key(request)
    if _rate_limiter.blocked(client_key, now):
        logger.warning("webhook: rate-limited %s", client_key)
        raise HTTPException(
            status_code=429, detail="too many requests",
            headers={"Retry-After": str(int(_rate_limit_window))},
        )

    def _reject(status: int, detail: str) -> HTTPException:
        _rate_limiter.allow(client_key, now)
        return HTTPException(status_code=status, detail=detail)
    # Periodic GC of the in-memory bucket dict so long-lived processes don't
    # accumulate stale per-IP entries.
    if int(now) % 60 == 0:
        _rate_limiter.gc(now)

    # 1. Cap body size AND wall-clock read time. The size cap blocks the
    #    "send 1 GB body" attack; the timeout blocks the slowloris-style
    #    "trickle bytes for hours" attack that would otherwise let a few
    #    cheap connections eat all our worker slots.
    import asyncio

    async def _read_capped() -> bytes:
        chunks: list[bytes] = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_BODY_BYTES:
                # 413 = Content Too Large / Request Entity Too Large.
                raise _reject(413, "payload too large")
            chunks.append(chunk)
        return b"".join(chunks)

    try:
        payload_bytes = await asyncio.wait_for(
            _read_capped(),
            timeout=float(os.environ.get("WEBHOOK_BODY_TIMEOUT_SECONDS", "10")),
        )
    except asyncio.TimeoutError:
        logger.warning("webhook: body read timeout from %s",
                       _client_key(request))
        raise _reject(408, "body read timeout")

    sig_header = request.headers.get("stripe-signature", "")
    if not sig_header:
        # Cheaper rejection before SDK does its own work.
        raise _reject(400, "missing stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload_bytes,
            sig_header,
            _webhook_secret(),
            tolerance=_tolerance_seconds(),
        )
    except ValueError as exc:
        # Generic 400 - don't echo the raw exception (could leak header bytes).
        logger.warning("webhook: bad payload (%s)", type(exc).__name__)
        raise _reject(400, "bad payload") from exc
    except stripe.error.SignatureVerificationError as exc:  # type: ignore[attr-defined]
        logger.warning("webhook: bad signature (%s)", type(exc).__name__)
        raise _reject(400, "bad signature") from exc

    event_id = event["id"]
    event_type = event["type"]
    payload_text = payload_bytes.decode("utf-8", errors="replace")
    fresh = subscriptions.record_stripe_event(event_id, event_type, payload_text)
    if not fresh:
        # Idempotent re-delivery from Stripe (or replay attempt).
        logger.info("webhook: duplicate event %s", event_id)
        return {"received": True, "duplicate": True}

    obj = event["data"]["object"]
    try:
        return _apply_event(event_id, event_type, event, obj)
    except EmailLookupError:
        # Transient: un-record so Stripe's retry is processed, and 503 so it
        # retries at all.
        subscriptions.forget_stripe_event(event_id)
        raise HTTPException(status_code=503, detail="customer lookup failed")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — e.g. "database is locked"
        subscriptions.forget_stripe_event(event_id)
        logger.error("webhook: processing failed for event %s (%s)",
                     event_id, type(exc).__name__)
        raise HTTPException(status_code=500, detail="processing failed") from exc


def _apply_event(event_id: str, event_type: str, event, obj) -> dict:
    """Apply one verified, first-seen event to the subscriptions store."""
    target_tier = tier_for_event(event_type, obj)
    if target_tier is None:
        # Decide before any Stripe API call: most event types change nothing.
        logger.info("webhook: no tier change for event type %s", event_type)
        return {"received": True, "event_type": event_type}

    email = _email_from_event(obj)
    if not email:
        # Don't log the payload (PII / card-last-four can be in there).
        logger.warning("webhook: no email on event %s type=%s",
                       event_id, event_type)
        return {"received": True, "no_email": True}

    event_created = _safe_get(event, "created")
    try:
        event_created = int(event_created) if event_created is not None else None
    except (TypeError, ValueError):
        event_created = None
    event_sub_id = _event_subscription_id(obj)

    try:
        existing = subscriptions.lookup(email)
    except ValueError:
        existing = None
    if existing:
        # Stripe does not guarantee delivery order: never let an older event
        # overwrite state written by a newer one.
        stored_created = existing.get("last_event_created")
        if (event_created is not None and stored_created is not None
                and event_created < int(stored_created)):
            logger.info("webhook: stale event %s ignored (out of order)", event_id)
            return {"received": True, "stale": True}
        # A downgrade for a DIFFERENT subscription than the one on file (old
        # sub's period-end cancel after a re-subscribe) must not revoke the
        # current one.
        stored_sub = existing.get("stripe_subscription_id")
        if (target_tier == "free" and event_sub_id and stored_sub
                and str(stored_sub).startswith("sub_")
                and event_sub_id != stored_sub):
            logger.info("webhook: downgrade for non-current subscription on "
                        "event %s ignored", event_id)
            return {"received": True, "stale_subscription": True}

    try:
        subscriptions.upsert(
            email=email,
            tier=target_tier,
            stripe_customer_id=_safe_get(obj, "customer"),
            stripe_subscription_id=event_sub_id,
            current_period_end=_iso_from_unix(
                _safe_get(obj, "current_period_end")
            ),
            last_event_type=event_type,
            last_event_created=event_created,
        )
    except ValueError as exc:
        # Email failed our validator (e.g. someone signed up with a
        # punycode address). The event row is already persisted in
        # `stripe_events` for forensic review. Returning 200 stops
        # Stripe's retry loop on what is structurally a permanent
        # rejection.
        logger.warning(
            "webhook: rejected upsert for event %s (%s)",
            event_id, type(exc).__name__,
        )
        return {"received": True, "rejected": True}

    # Alert the operator when a paying user's payment fails (best-effort).
    if event_type == "invoice.payment_failed":
        alert_payment_failed(email)

    # Don't echo the email back in the response body. Stripe doesn't read it,
    # and a misconfigured proxy could log responses.
    return {"received": True, "event_type": event_type}
