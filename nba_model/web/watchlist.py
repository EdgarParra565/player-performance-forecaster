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
import tempfile
import threading
import time
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


# One process-wide lock serialises read-modify-write cycles between
# Streamlit sessions (threads in the same server process).
_STORE_LOCK = threading.Lock()
MAX_ITEM_LEN = 80
# Anonymous (open-launch) keys expire after this many days without a save,
# so one-off visitors don't grow the store forever. Signed-in keys never expire.
ANON_TTL_DAYS = int(os.environ.get("WATCHLIST_ANON_TTL_DAYS", "90") or 90)
_SEEN_KEY = "__seen__"


class _CorruptStore(Exception):
    """The store exists but can't be parsed — never overwrite it."""


def _read_store(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _CorruptStore(str(path)) from exc
    return data if isinstance(data, dict) else {}


def _load_all() -> dict:
    try:
        return _read_store(Path(_store_path()))
    except _CorruptStore:
        return {}


def _save_all(payload: dict) -> None:
    """Atomic write: temp file in the same dir + os.replace, so a reader
    never sees a half-written file (which used to parse as {} and then get
    saved back over EVERY user's list)."""
    path = Path(_store_path())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".watchlists.", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            # File-mode 0600 so multi-tenant hosts can't read other users' lists.
            if os.name == "posix":
                os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError:
        pass


def _load_for(key: str) -> list[str]:
    """Stored items for a watchlist key (email or ``anon:<token>``)."""
    if key == _SEEN_KEY:
        return []
    return list(_load_all().get(key, []))


def _prune_expired_anon(payload: dict, now: float) -> None:
    """Drop ``anon:*`` keys not saved within ANON_TTL_DAYS (in place).

    ``__seen__`` maps key -> last-save epoch seconds. Legacy anon keys with no
    stamp get one now (a grace period) instead of being deleted outright."""
    seen = payload.setdefault(_SEEN_KEY, {})
    cutoff = now - ANON_TTL_DAYS * 86_400
    for key in [k for k in payload if isinstance(k, str) and k.startswith("anon:")]:
        stamp = seen.get(key)
        if stamp is None:
            seen[key] = now
        elif stamp < cutoff:
            payload.pop(key, None)
            seen.pop(key, None)
    for key in [k for k in seen if k not in payload]:
        seen.pop(key, None)


def _save_for(key: str, items: list[str], now: Optional[float] = None) -> None:
    """Persist items for a key, enforcing the MAX_ITEMS / length caps."""
    if key == _SEEN_KEY:
        return
    clean = [str(i)[:MAX_ITEM_LEN] for i in items if str(i).strip()][:MAX_ITEMS]
    with _STORE_LOCK:
        try:
            payload = _read_store(Path(_store_path()))
        except _CorruptStore:
            # Keep the damaged file for recovery rather than clobbering
            # everyone's pins with a single-key payload.
            return
        payload[key] = clean
        now = time.time() if now is None else now
        payload.setdefault(_SEEN_KEY, {})[key] = now
        _prune_expired_anon(payload, now)
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


def add(player_name: str, is_known: Optional[Callable[[str], bool]] = None) -> bool:
    """Pin a player. ``is_known`` (the app passes a DB-backed check) rejects
    names that aren't real players — the pin value comes from a URL param."""
    name = str(player_name or "").strip()
    if not name or len(name) > MAX_ITEM_LEN:
        return False
    if is_known is not None and not is_known(name):
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
