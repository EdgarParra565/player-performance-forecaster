"""Request guards for the flagship API: per-IP rate limiting + optional
shared access code. Both are plain ASGI-level checks registered in main.py.

Client identity
---------------
The limiter keys on ``request.client.host``. Behind a reverse proxy, run
uvicorn with ``--proxy-headers --forwarded-allow-ips=<proxy IP>`` (the
flagship image does, via FORWARDED_ALLOW_IPS): uvicorn then rewrites
``client.host`` from X-Forwarded-For, trusting ONLY hops from that proxy. A
client-supplied X-Forwarded-For from anywhere else is ignored, so the key
cannot be spoofed. Without a proxy, ``client.host`` is the socket peer.

On platforms whose edge proxy sets a client-IP header it always OVERWRITES
(Fly.io: ``Fly-Client-IP``), set ``API_CLIENT_IP_HEADER`` to that header name
instead. Only do this when every request reaches the app through that proxy,
or a client could pick its own key. A missing or malformed value falls back
to ``client.host``.

Budgets (per client IP, sliding 60 s window; env-tunable)
---------------------------------------------------------
- heavy  (API_RATE_HEAVY_PER_MIN, default 20):  CPU-heavy model scans /
  Monte Carlo — /api/slate/edges, /api/cross-book, /api/parlay/*
- read   (API_RATE_READ_PER_MIN,  default 240): every other /api/* call
- auth   (API_RATE_AUTH_FAIL_PER_MIN, default 10): WRONG access codes
  (brute-force brake; counts failures only)
Static SPA files and /api/health are not limited. API_RATE_LIMIT=0 disables
the limiter (tests / trusted internal deploys).
"""
from __future__ import annotations

import hmac
import ipaddress
import math
import os
import time
from collections import deque
from threading import Lock
from typing import Optional

WINDOW_SECONDS = 60.0
ACCESS_HEADER = "x-access-code"

HEAVY_PREFIXES = ("/api/slate/edges", "/api/cross-book", "/api/parlay")
EXEMPT_PATHS = frozenset({"/api/health"})


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, "") or default))
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class SlidingWindowLimiter:
    """Thread-safe per-key sliding-window counter (in-memory, per process).

    Multi-replica deploys get a budget per replica; put a shared limiter at
    the proxy/CDN if that matters (DEPLOYMENT.md §15)."""

    def __init__(self, max_requests: int, window_seconds: float = WINDOW_SECONDS):
        self.max_requests = int(max_requests)
        self.window = float(window_seconds)
        self._buckets: dict[str, deque[float]] = {}
        self._lock = Lock()
        self._last_gc = 0.0

    def _prune(self, bucket: deque, now: float) -> None:
        cutoff = now - self.window
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

    def hit(self, key: str, now: float) -> Optional[float]:
        """Record a request. None if allowed, else seconds until a slot frees."""
        with self._lock:
            self._gc(now)
            bucket = self._buckets.setdefault(key, deque())
            self._prune(bucket, now)
            if len(bucket) >= self.max_requests:
                return max(0.0, bucket[0] + self.window - now)
            bucket.append(now)
            return None

    def blocked(self, key: str, now: float) -> Optional[float]:
        """Like ``hit`` but without recording (used for failure-only budgets)."""
        with self._lock:
            bucket = self._buckets.get(key)
            if not bucket:
                return None
            self._prune(bucket, now)
            if len(bucket) >= self.max_requests:
                return max(0.0, bucket[0] + self.window - now)
            return None

    def record(self, key: str, now: float) -> None:
        with self._lock:
            self._buckets.setdefault(key, deque()).append(now)

    def _gc(self, now: float) -> None:
        # Drop idle keys once a window so memory can't grow without bound.
        if now - self._last_gc < self.window:
            return
        self._last_gc = now
        stale = [k for k, b in self._buckets.items() if not b or b[-1] <= now - self.window]
        for k in stale:
            del self._buckets[k]

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


class Guards:
    """Holds limiter state; rebuilt by ``configure()`` (tests change env)."""

    def __init__(self) -> None:
        self.configure()

    def configure(self) -> None:
        self.enabled = _env_flag("API_RATE_LIMIT", True)
        self.heavy = SlidingWindowLimiter(_env_int("API_RATE_HEAVY_PER_MIN", 20))
        self.read = SlidingWindowLimiter(_env_int("API_RATE_READ_PER_MIN", 240))
        self.auth_fail = SlidingWindowLimiter(_env_int("API_RATE_AUTH_FAIL_PER_MIN", 10))

    @staticmethod
    def client_key(header_value: Optional[str], peer: Optional[str]) -> str:
        """Rate-limit key: the trusted proxy header when configured + valid."""
        if os.environ.get("API_CLIENT_IP_HEADER", "").strip() and header_value:
            candidate = header_value.split(",")[0].strip()
            try:
                return str(ipaddress.ip_address(candidate))
            except ValueError:
                pass
        return peer or "unknown"

    @staticmethod
    def access_code() -> str:
        return os.environ.get("FLAGSHIP_ACCESS_CODE", "").strip()

    @staticmethod
    def is_api(path: str) -> bool:
        return path == "/api" or path.startswith("/api/")

    def tier_for(self, path: str) -> Optional[SlidingWindowLimiter]:
        if not self.is_api(path) or path in EXEMPT_PATHS:
            return None
        if path.startswith(HEAVY_PREFIXES):
            return self.heavy
        return self.read

    def code_ok(self, supplied: Optional[str]) -> bool:
        expected = self.access_code()
        if not expected:
            return True
        return bool(supplied) and hmac.compare_digest(
            supplied.encode("utf-8"), expected.encode("utf-8"))


def retry_after(seconds: float) -> str:
    return str(max(1, int(math.ceil(seconds))))


def now() -> float:
    return time.monotonic()


GUARDS = Guards()
