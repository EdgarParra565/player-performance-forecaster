"""Per-user (when billed) / per-browser (when free) player watchlist.

Persistence policy:
- **Always**: in-session via `st.session_state["watchlist"]` so removing/adding
  is instant for the current visit.
- **When BILLING_ENABLED and the user is signed in**: also persisted to
  `data/state/watchlists.json` keyed by email, so the same user sees their
  list on the next visit / device.

Anonymous (open-launch) visitors get session-only persistence — no JSON
fingerprint, no cross-session tracking.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import streamlit as st

from nba_model.web import auth as web_auth

DEFAULT_STORE = "data/state/watchlists.json"
MAX_ITEMS = 25


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


def _user_key() -> Optional[str]:
    """Stable key for cross-session persistence; None when anonymous."""
    if not web_auth.BILLING_ENABLED:
        return None
    user = web_auth.current_user()
    if not user.is_authenticated or not user.email:
        return None
    return user.email.lower().strip()


def get() -> list[str]:
    """Return the current watchlist (in priority order)."""
    session_list = st.session_state.setdefault("watchlist", None)
    if session_list is not None:
        return list(session_list)
    # First-load hydration.
    key = _user_key()
    if key:
        stored = _load_all().get(key, [])
    else:
        stored = []
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
        return  # session-only for anonymous / open-launch visitors
    payload = _load_all()
    payload[key] = items
    _save_all(payload)
