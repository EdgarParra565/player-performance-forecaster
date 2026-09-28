"""Per-book session health for the hourly scraping loop.

The hourly runner fetches every book through the dedicated scraping Chrome
(CDP ``:9222``). Its only routine failure mode that needs a human is an
expired book login. This module turns one run's web-text fetch results into a
per-book status and decides — with a remembered state file — whether the
owner must be told:

- ``ok``            every fetched page for the book is real content
- ``login-needed``  at least one fetched page is a login wall
                    (``web_text_ingestion.detect_login_wall``, i.e. the
                    book-specific ``session_markers`` the parse path uses)
- ``unreachable``   no page for the book could be fetched this run

Debounce: an alert fires only on a NEW ``login-needed`` — once per episode.
The book's ``login_alerted`` flag is set when the alert is delivered (or when
no alert channel is configured, so the report doesn't re-announce it every
hour) and re-armed only when the book is seen ``ok`` again. ``unreachable``
ticks in between neither re-arm nor re-fire. A failed delivery leaves the flag
unset so the next tick retries.

Books listed in ``data/config/blocked_books.txt`` (geo / age / app-only blocks
the owner can't fix by logging in) report ``blocked`` and never alert.

No logins, credentials, or page interaction happen here — classification is
pure over already-fetched text.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

STATUS_OK = "ok"
STATUS_LOGIN_NEEDED = "login-needed"
STATUS_UNREACHABLE = "unreachable"
# Known non-auth block (geo / age / app-only) from data/config/blocked_books.txt:
# reported, never alerted, never prompted for login.
STATUS_BLOCKED = "blocked"

DEFAULT_BLOCKED_BOOKS_FILE = "data/config/blocked_books.txt"

# Books whose boards need a signed-in session. For every OTHER (public) book a
# wall verdict from the GENERIC nav rule ("Log in" link + too few content
# markers) means the board didn't render — a flaky capture, not an expired
# session — so it reports unreachable instead of alerting "re-login needed"
# (live 2026-09-28: a Kalshi nav-shell capture). Specific wall phrases still
# count for every book.
AUTH_REQUIRED_BOOKS = frozenset({
    "prizepicks", "underdog", "pick6", "parlayplay", "betrivers", "sleeper", "dabble",
})
_GENERIC_WALL_PREFIXES = ("generic login nav", "generic login-wall markers")

STATE_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _book_for_url(url: str) -> str:
    from nba_model.model.web_text_ingestion import classify_snapshot_session

    return classify_snapshot_session("", url)["book"]


def classify_books(
    fetch_results: Optional[list[dict]],
    unreachable_urls: Optional[list[str]] = None,
) -> dict:
    """Aggregate per-URL fetch results into ``{book: {"status", "urls"}}``.

    ``fetch_results`` are ``fetch_and_store_web_text(...)["results"]`` rows
    (``status`` fetched/failed, plus ``book`` / ``session`` on fetched rows).
    ``unreachable_urls`` covers a web-text step that raised before producing
    results. Precedence per book: login-needed > ok > unreachable.
    """
    per_book: dict[str, list[dict]] = {}
    for row in fetch_results or []:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "")
        book = row.get("book") or _book_for_url(url)
        fetch_status = row.get("status")
        http_status = row.get("http_status")
        if fetch_status == "fetched" and isinstance(http_status, int) and http_status >= 400:
            # A 404 page's nav ("Log In / Sign Up / Explore") can pass the
            # content-marker check — a dead URL is not a healthy session.
            status = STATUS_UNREACHABLE
            row = dict(row, session_reason=f"HTTP {http_status} (URL moved?)")
        elif fetch_status == "fetched" and row.get("session") == "login_wall":
            reason = str(row.get("session_reason") or "")
            if (book not in AUTH_REQUIRED_BOOKS
                    and reason.lower().startswith(_GENERIC_WALL_PREFIXES)):
                status = STATUS_UNREACHABLE
                row = dict(row, session_reason=f"board didn't render ({reason})")
            else:
                status = STATUS_LOGIN_NEEDED
        elif fetch_status == "fetched":
            status = STATUS_OK
        elif fetch_status == "failed":
            status = STATUS_UNREACHABLE
        else:  # skipped_recent etc. — says nothing about the session
            continue
        per_book.setdefault(book, []).append({
            "url": url,
            "status": status,
            "reason": row.get("session_reason") or row.get("error_message"),
        })
    for url in unreachable_urls or []:
        per_book.setdefault(_book_for_url(url), []).append(
            {"url": url, "status": STATUS_UNREACHABLE, "reason": "web_text step failed"}
        )

    books: dict = {}
    for book, entries in per_book.items():
        statuses = {e["status"] for e in entries}
        if STATUS_LOGIN_NEEDED in statuses:
            status = STATUS_LOGIN_NEEDED
        elif STATUS_OK in statuses:
            status = STATUS_OK
        else:
            status = STATUS_UNREACHABLE
        books[book] = {"status": status, "urls": entries}
    return books


def load_blocked_books(path: Optional[str]) -> dict:
    """``{book: {"kind", "note"}}`` from the blocked-books file (missing = none)."""
    out: dict = {}
    if not path:
        return out
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        out[parts[0].lower()] = {
            "kind": parts[1] if len(parts) > 1 else "blocked",
            "note": parts[2] if len(parts) > 2 else "",
        }
    return out


def apply_blocked(books: dict, blocked: dict) -> list[str]:
    """Rewrite non-ok statuses of blocked books to ``blocked`` (mutates
    ``books``). Returns blocked books that read ``ok`` — the block lifted."""
    lifted = []
    for book, info in books.items():
        if book not in blocked:
            continue
        if info["status"] == STATUS_OK:
            lifted.append(book)
            continue
        info["status"] = STATUS_BLOCKED
        info["blocked"] = dict(blocked[book])
    return sorted(lifted)


def load_state(path: str) -> dict:
    """Read the remembered per-book state; a missing/corrupt file = fresh state."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": STATE_VERSION, "books": {}}
    if not isinstance(data, dict) or not isinstance(data.get("books"), dict):
        return {"version": STATE_VERSION, "books": {}}
    return data


def save_state(path: str, state: dict) -> None:
    """Atomically write the state file (tmp + replace)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)


def evaluate_transitions(books: dict, state: dict, now_utc: Optional[str] = None) -> dict:
    """Merge this run's ``books`` into ``state`` (mutated) and report changes.

    Returns ``{"needs_alert": [book...], "recovered": [book...],
    "changed": [book...]}``. ``needs_alert`` = books login-needed whose
    current episode hasn't been alerted yet. Books absent from this run keep
    their remembered entry untouched.
    """
    now_utc = now_utc or _utc_now_iso()
    remembered = state.setdefault("books", {})
    needs_alert: list[str] = []
    recovered: list[str] = []
    changed: list[str] = []
    for book in sorted(books):
        status = books[book]["status"]
        prev = remembered.get(book) or {}
        prev_status = prev.get("status")
        entry = dict(prev)
        if status != prev_status:
            changed.append(book)
            entry["since_utc"] = now_utc
        entry["status"] = status
        entry["last_checked_utc"] = now_utc
        if status == STATUS_OK:
            if prev.get("login_alerted") or prev_status == STATUS_LOGIN_NEEDED:
                recovered.append(book)
            entry["login_alerted"] = False
        elif status == STATUS_LOGIN_NEEDED and not prev.get("login_alerted"):
            needs_alert.append(book)
        entry.setdefault("login_alerted", False)
        remembered[book] = entry
    state["version"] = STATE_VERSION
    state["updated_at_utc"] = now_utc
    return {"needs_alert": needs_alert, "recovered": recovered, "changed": changed}


def run_session_health(
    fetch_results: Optional[list[dict]],
    state_file: str,
    *,
    unreachable_urls: Optional[list[str]] = None,
    notify: Optional[Callable[[list[str], dict], dict]] = None,
    now_utc: Optional[str] = None,
    blocked: Optional[dict] = None,
) -> dict:
    """Classify, debounce, notify, persist. Returns the step summary.

    ``notify(books_needing_login, books)`` returns
    ``{"delivered": bool, "retry": bool, ...}``; ``retry=True`` means the
    delivery failed and the episode must stay un-alerted for the next tick.
    With ``notify=None`` the alert is recorded in the summary only.
    """
    now_utc = now_utc or _utc_now_iso()
    books = classify_books(fetch_results, unreachable_urls)
    blocked_but_ok = apply_blocked(books, blocked or {})
    state = load_state(state_file)
    transitions = evaluate_transitions(books, state, now_utc)

    notification = None
    if transitions["needs_alert"]:
        if notify is None:
            notification = {"delivered": False, "retry": False, "reason": "no_channel"}
        else:
            try:
                notification = notify(list(transitions["needs_alert"]), books)
            except Exception as exc:  # noqa: BLE001 — alerting is best-effort
                notification = {"delivered": False, "retry": True,
                                "reason": f"notify raised: {exc}"}
        if not notification.get("retry"):
            for book in transitions["needs_alert"]:
                state["books"][book]["login_alerted"] = True
                state["books"][book]["login_alerted_at_utc"] = now_utc

    save_state(state_file, state)
    counts = {s: 0 for s in (STATUS_OK, STATUS_LOGIN_NEEDED, STATUS_UNREACHABLE,
                             STATUS_BLOCKED)}
    for info in books.values():
        counts[info["status"]] += 1
    return {
        # Informational step: login-needed is surfaced by the debounced
        # session alert, never by failing/degrading the hourly run (which
        # would re-alert every hour through build_alert).
        "status": "success",
        "books": {b: info["status"] for b, info in sorted(books.items())},
        "details": books,
        "counts": counts,
        "new_login_needed": transitions["needs_alert"],
        "recovered": transitions["recovered"],
        "changed": transitions["changed"],
        "blocked_but_ok": blocked_but_ok,
        "notification": notification,
        "state_file": str(state_file),
    }
