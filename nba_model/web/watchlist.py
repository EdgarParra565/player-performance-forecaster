"""Per-user (when billed) / per-browser (when open-launch) player watchlist.

Persistence policy:
- **Always**: in-session via `st.session_state["watchlist"]` so removing/adding
  is instant for the current visit.
- **When BILLING_ENABLED and the user is signed in**: persisted to
  `data/state/watchlists.json` keyed by email, so the same user sees their
  list on the next visit / device.
- **Open-launch (BILLING disabled) anonymous visitors**: persisted to the same
  JSON store keyed by an anonymous per-browser token (`anon:<token>`). The
  token lives in a first-party cookie the browser keeps across sessions — no
  account required, no server-side fingerprint. Pins survive a refresh / a
  return visit on the same browser. Capped at `MAX_ITEMS` like every other key.

When BILLING_ENABLED is on but the visitor is signed out, we stay session-only
(no cookie, no JSON) so we don't track people who haven't opted in.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Callable, Optional

import streamlit as st

from nba_model.web import auth as web_auth

DEFAULT_STORE = "data/state/watchlists.json"
MAX_ITEMS = 25
ANON_COOKIE = "ppf_watchlist_id"
_ANON_SESSION_KEY = "_wl_anon_token"
# Tokens we mint are uuid4 hex (≤32 lowercase hex chars). We refuse to reuse
# any cookie/session value that doesn't match — an attacker-planted cookie
# could otherwise inject markup into the document.cookie JS bridge.
_TOKEN_RE = re.compile(r"^[0-9a-f]{1,32}$")


def _store_path() -> str:
    return os.environ.get("WATCHLIST_STORE_PATH", DEFAULT_STORE)


def _load_all() -> dict:
    path = Path(_store_path())
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_all(payload: dict) -> None:
    path = Path(_store_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # File-mode 0600 so multi-tenant hosts can't read other users' lists.
        if os.name == "posix":
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
    except OSError:
        pass


def _load_for(key: str) -> list[str]:
    """Stored items for a watchlist key (email or ``anon:<token>``)."""
    return list(_load_all().get(key, []))


def _save_for(key: str, items: list[str]) -> None:
    """Persist items for a key, enforcing the MAX_ITEMS cap."""
    payload = _load_all()
    payload[key] = list(items)[:MAX_ITEMS]
    _save_all(payload)


# ---------------------------------------------------------------------------
# Anonymous per-browser token (open-launch cross-session persistence)
# ---------------------------------------------------------------------------

def _resolve_anon_token(
    session_token: Optional[str],
    cookie_token: Optional[str],
    mint: Callable[[], str] = lambda: uuid.uuid4().hex[:24],
) -> str:
    """Pick the anonymous token to use.

    Priority: this session's cached token, then the cookie the browser sent
    on a return visit, otherwise mint a fresh one. Pure so it can be tested
    without a Streamlit runtime. Only well-formed (hex) tokens are reused —
    anything else is treated as absent and a fresh token is minted.
    """
    if session_token and _TOKEN_RE.match(session_token):
        return session_token
    if cookie_token and _TOKEN_RE.match(cookie_token):
        return cookie_token
    return mint()


def _read_cookie(name: str) -> Optional[str]:
    """Best-effort read of a request cookie; None when unavailable."""
    try:
        cookies = st.context.cookies  # type: ignore[attr-defined]
        value = cookies.get(name) if cookies else None
        return str(value) if value else None
    except Exception:
        return None


def _write_cookie(name: str, value: str) -> None:
    """Best-effort persist of a first-party cookie via a tiny JS bridge.

    The component iframe is same-origin with the app, so ``document.cookie``
    sets a host cookie the next page load will send back. 1-year expiry,
    SameSite=Lax. No-op (swallowed) when components aren't available.
    """
    try:
        import streamlit.components.v1 as components
        components.html(
            f"""
            <script>
            try {{
                var d = new Date();
                d.setTime(d.getTime() + 365*24*60*60*1000);
                document.cookie = "{name}={value};expires=" + d.toUTCString()
                    + ";path=/;SameSite=Lax";
            }} catch (e) {{}}
            </script>
            """,
            height=0,
        )
    except Exception:
        pass


def _anon_token() -> Optional[str]:
    """Stable per-browser token for anonymous cross-session persistence."""
    try:
        cached = st.session_state.get(_ANON_SESSION_KEY)
    except Exception:
        return None
    if cached:
        return str(cached)
    token = _resolve_anon_token(None, _read_cookie(ANON_COOKIE))
    try:
        st.session_state[_ANON_SESSION_KEY] = token
    except Exception:
        return token
    # Persist (or refresh the expiry of) the token in the browser.
    _write_cookie(ANON_COOKIE, token)
    return token


def _user_key() -> Optional[str]:
    """Stable key for cross-session persistence; None for session-only."""
    if web_auth.BILLING_ENABLED:
        user = web_auth.current_user()
        if user.is_authenticated and user.email:
            return user.email.lower().strip()
        # Signed-out under billing: session-only, no tracking.
        return None
    # Open-launch anonymous visitor: persist per-browser via the anon token.
    token = _anon_token()
    return f"anon:{token}" if token else None


def get() -> list[str]:
    """Return the current watchlist (in priority order)."""
    session_list = st.session_state.setdefault("watchlist", None)
    if session_list is not None:
        return list(session_list)
    # First-load hydration from the persistent store (if we have a key).
    key = _user_key()
    stored = _load_for(key) if key else []
    st.session_state["watchlist"] = stored
    return list(stored)


def add(player_name: str) -> bool:
    name = str(player_name or "").strip()
    if not name:
        return False
    current = get()
    if name in current:
        return False
    current.insert(0, name)
    if len(current) > MAX_ITEMS:
        current = current[:MAX_ITEMS]
    st.session_state["watchlist"] = current
    _persist(current)
    return True


def remove(player_name: str) -> bool:
    name = str(player_name or "").strip()
    current = get()
    if name not in current:
        return False
    current.remove(name)
    st.session_state["watchlist"] = current
    _persist(current)
    return True


def clear() -> None:
    st.session_state["watchlist"] = []
    _persist([])


def _persist(items: list[str]) -> None:
    key = _user_key()
    if not key:
        return  # session-only when we have no stable key
    _save_for(key, items)
