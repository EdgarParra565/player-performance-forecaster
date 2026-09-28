"""Lightweight per-client / per-session rate limiting for the web app (WS4).

Free-tier users can hammer the slate-wide scan / book fetches; this is a basic
app-layer throttle (not a substitute for an LB / Cloudflare gate, but enough to
stop accidental tight loops). The decision core is pure so it's unit-testable
without a Streamlit runtime; ``session_rate_limit`` wires it to session_state.
"""
from __future__ import annotations

from typing import Optional


def check_rate(
    timestamps: list,
    now: float,
    max_calls: int,
    window_seconds: float,
) -> tuple:
    """Sliding-window decision.

    Returns ``(allowed, pruned_timestamps)``. ``allowed`` is True when fewer
    than ``max_calls`` calls fall within the trailing ``window_seconds``. When
    allowed, ``now`` is appended to the returned (pruned) list so the caller can
    persist it.
    """
    cutoff = now - float(window_seconds)
    recent = [t for t in timestamps if t >= cutoff]
    if len(recent) >= int(max_calls):
        return False, recent
    recent.append(now)
    return True, recent


def session_rate_limit(
    key: str,
    max_calls: int,
    window_seconds: float,
    now: Optional[float] = None,
) -> bool:
    """True if this call is within the per-session budget for ``key``.

    Stores timestamps under ``st.session_state['_rate_limits'][key]``. Degrades
    open (returns True) if a Streamlit runtime isn't available.
    """
    import time
    if now is None:
        now = time.monotonic()
    try:
        import streamlit as st
        store = st.session_state.setdefault("_rate_limits", {})
    except Exception:
        return True
    allowed, recent = check_rate(
        list(store.get(key, [])), now, max_calls, window_seconds)
    store[key] = recent
    return allowed


# ---------------------------------------------------------------------------
# Per-client (IP) limiting — a new browser session must NOT reset the budget.
# ---------------------------------------------------------------------------
import os as _os
from threading import Lock as _Lock

_CLIENT_STORE: dict = {}
_CLIENT_LOCK = _Lock()
_CLIENT_STORE_MAX_KEYS = 50_000


def client_ip_from(
    peer_ip: Optional[str],
    forwarded_for: Optional[str],
    trusted_hops: int = 0,
) -> Optional[str]:
    """Resolve the client IP. Pure + testable.

    With ``trusted_hops == 0`` (no reverse proxy) the socket peer is used and
    any client-supplied X-Forwarded-For is ignored (it's spoofable). Behind N
    trusted proxies, the N-th address from the RIGHT of X-Forwarded-For is the
    client (each trusted hop appends the address it received from) — the same
    contract as ``webhook_app._client_key``.
    """
    if trusted_hops > 0 and forwarded_for:
        ips = [p.strip() for p in str(forwarded_for).split(",") if p.strip()]
        if ips:
            return ips[max(0, len(ips) - trusted_hops)]
    return peer_ip or None


def current_client_ip() -> Optional[str]:
    """The visitor's IP from the Streamlit request context, if obtainable."""
    try:
        hops = int(_os.environ.get("STREAMLIT_TRUSTED_PROXY_HOPS", "0") or 0)
    except ValueError:
        hops = 0
    try:
        import streamlit as st
        ctx = st.context
        peer = getattr(ctx, "ip_address", None)
        headers = getattr(ctx, "headers", None) or {}
        xff = headers.get("X-Forwarded-For") if hasattr(headers, "get") else None
    except Exception:
        return None
    return client_ip_from(peer, xff, hops)


def client_rate_limit(
    client_id: str,
    key: str,
    max_calls: int,
    window_seconds: float,
    now: Optional[float] = None,
) -> bool:
    """Process-wide sliding window keyed by (client_id, key)."""
    import time
    if now is None:
        now = time.monotonic()
    with _CLIENT_LOCK:
        if len(_CLIENT_STORE) > _CLIENT_STORE_MAX_KEYS:
            cutoff = now - float(window_seconds)
            for k in [k for k, v in _CLIENT_STORE.items() if not v or v[-1] < cutoff]:
                del _CLIENT_STORE[k]
        allowed, recent = check_rate(
            list(_CLIENT_STORE.get((client_id, key), [])), now, max_calls, window_seconds)
        _CLIENT_STORE[(client_id, key)] = recent
    return allowed


def rate_limit(key: str, max_calls: int, window_seconds: float) -> bool:
    """Per-client-IP budget when the IP is obtainable, else per-session."""
    ip = current_client_ip()
    if ip:
        return client_rate_limit(ip, key, max_calls, window_seconds)
    return session_rate_limit(key, max_calls, window_seconds)
