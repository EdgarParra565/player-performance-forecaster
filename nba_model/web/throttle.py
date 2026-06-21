"""Lightweight per-session rate limiting for the web app (WS4).

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
