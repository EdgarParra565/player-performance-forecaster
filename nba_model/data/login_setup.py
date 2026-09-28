"""One-time login pass for the scraping Chrome (the owner's only manual step).

Opens one tab per auth book in the dedicated scraping Chrome (CDP :9222),
then polls each tab's session classification every ~30s and prints a live
checklist until every book is ``ok`` (or ``--timeout``):

    .venv/bin/python3 -m nba_model.data.login_setup
    scripts/scheduler/login_setup.sh --timeout 3600

The owner types credentials into the tabs themselves. This tool NEVER fills
forms, stores credentials or touches captchas — it only navigates each tab to
the book's board/login URL once and then reads page text passively (no
navigation, no focus changes) to classify it with ``detect_login_wall`` — the
same book-specific session markers the hourly ``session_health`` step uses.

While running it holds the hourly runner's lockfile, so an hourly tick that
fires mid-login exits 75 instead of running tab hygiene (which would close
the login tabs). When every tab reads ``ok``, a verification pass re-fetches
each book in a FRESH tab (the real hourly fetch path) so a login that only
looks done in the owner's tab isn't reported as ok.

Exit codes: 0 all ok · 3 timeout / books left · 75 hourly lock busy ·
78 Chrome CDP unreachable.
"""
from __future__ import annotations

import argparse
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from nba_model.data import session_health
from nba_model.logging_utils import get_logger

logger = get_logger("nba_model.login_setup")

DEFAULT_URLS_FILE = "data/config/web_text_urls.txt"
DEFAULT_STATE_FILE = "nba_model/data/artifacts/hourly/session_health_state.json"
DEFAULT_LOCKFILE = "/tmp/nba_hourly_update.lock"

# Books whose boards need a signed-in session (DFS pick'em). Any other book
# the last hourly run marked login-needed is added on top.
# BetRivers (21+) and BetMGM (geo) are permanent blocks for this owner — see
# data/config/blocked_books.txt, which is always excluded from the prompts.
DEFAULT_AUTH_BOOKS = ("prizepicks", "underdog", "pick6", "parlayplay")

# Curated login/board URL per auth book; wins over web_text_urls.txt (whose
# board URL can be dead — Underdog's /NBA paths 404 logged-out, 2026-09-28).
LOGIN_URLS = {
    "prizepicks": "https://app.prizepicks.com/board/nba",
    "underdog": "https://underdogsports.com/pick-em/higher-lower/all",
    "pick6": "https://pick6.draftkings.com/?sport=NBA",
    "parlayplay": "https://parlayplay.io/",
}

EXIT_OK = 0
EXIT_INCOMPLETE = 3
EXIT_LOCKED = 75
EXIT_NO_CHROME = 78

TAB_CLOSED = "tab-closed"


@dataclass
class LoginTarget:
    book: str
    url: str
    status: str = session_health.STATUS_LOGIN_NEEDED
    reason: Optional[str] = None
    page: object = field(default=None, repr=False)


def _book_for_url(url: str) -> str:
    from nba_model.model.web_text_ingestion import classify_snapshot_session

    return classify_snapshot_session("", url)["book"]


def build_login_targets(
    urls: list[str],
    auth_books=DEFAULT_AUTH_BOOKS,
    state: Optional[dict] = None,
    only_books: Optional[list[str]] = None,
    blocked: Optional[dict] = None,
) -> list[LoginTarget]:
    """One target per book: auth books + books remembered as login-needed.

    Uses the curated ``LOGIN_URLS`` entry, else the book's FIRST URL in
    ``urls`` (file order). ``only_books`` restricts the set (CLI ``--books``).
    """
    first_url: dict[str, str] = {}
    for url in urls:
        first_url.setdefault(_book_for_url(url), url)
    wanted = list(dict.fromkeys(auth_books))
    for book, info in sorted(((state or {}).get("books") or {}).items()):
        if (info or {}).get("status") == session_health.STATUS_LOGIN_NEEDED:
            wanted.append(book)
    wanted = [b for b in dict.fromkeys(wanted) if b not in (blocked or {})]
    if only_books:
        keep = {b.strip().lower() for b in only_books}
        wanted = [b for b in wanted if b in keep]
    targets = []
    for book in wanted:
        url = LOGIN_URLS.get(book) or first_url.get(book)
        if url:
            targets.append(LoginTarget(book=book, url=url))
    return targets


def classify_text(target: LoginTarget, text: str) -> tuple[str, Optional[str]]:
    """Session status of a tab's text for ``target`` (board-URL markers)."""
    from nba_model.model.web_text_ingestion import detect_login_wall

    if not str(text or "").strip():
        return session_health.STATUS_LOGIN_NEEDED, "page still empty / loading"
    is_wall, reason = detect_login_wall(text, target.url)
    if is_wall:
        return session_health.STATUS_LOGIN_NEEDED, reason
    return session_health.STATUS_OK, None


_ICON = {
    session_health.STATUS_OK: "[ok]  ",
    session_health.STATUS_LOGIN_NEEDED: "[ .. ]",
    session_health.STATUS_UNREACHABLE: "[ !! ]",
    TAB_CLOSED: "[ xx ]",
}


def render_checklist(targets: list[LoginTarget], now: Optional[str] = None) -> str:
    now = now or datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    done = sum(t.status == session_health.STATUS_OK for t in targets)
    lines = [f"--- {now}  {done}/{len(targets)} books ok ---"]
    for t in targets:
        extra = f"  ({t.reason})" if t.reason and t.status != session_health.STATUS_OK else ""
        lines.append(f"{_ICON.get(t.status, '[ ?? ]')} {t.book:<11} {t.status:<13}{extra}")
    return "\n".join(lines)


def poll_until_ok(
    targets: list[LoginTarget],
    read_text: Callable[[LoginTarget], Optional[str]],
    *,
    poll_seconds: float,
    timeout_seconds: float,
    printer: Callable[[str], None] = print,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Poll every tab until all are ok or the timeout passes. Returns all-ok.

    ``read_text(target)`` returns the tab's visible text, or None when the tab
    was closed by the owner.
    """
    deadline = clock() + max(0.0, float(timeout_seconds))
    last_render = None
    while True:
        for t in targets:
            if t.status == session_health.STATUS_UNREACHABLE and t.page is None:
                continue  # navigation never succeeded; nothing to read
            try:
                text = read_text(t)
            except Exception as exc:  # noqa: BLE001 — a flaky read is not fatal
                t.status, t.reason = session_health.STATUS_UNREACHABLE, f"read failed: {exc}"
                continue
            if text is None:
                t.status, t.reason = TAB_CLOSED, "tab closed — will verify in a fresh tab"
            else:
                t.status, t.reason = classify_text(t, text)
        render = render_checklist(targets)
        body = render.split("\n", 1)[1]
        if body != last_render:
            printer(render)
            last_render = body
        if all(t.status in (session_health.STATUS_OK, TAB_CLOSED) for t in targets):
            return True
        if clock() >= deadline:
            return False
        sleeper(poll_seconds)


# --- live Chrome glue (not unit-tested; verified against the real :9222) ---

def _open_tabs(context, targets: list[LoginTarget], nav_timeout_ms: int) -> None:
    for t in targets:
        try:
            page = context.new_page()
            page.set_default_timeout(nav_timeout_ms)
            page.goto(t.url, wait_until="domcontentloaded")
            t.page = page
        except Exception as exc:  # noqa: BLE001
            t.status, t.reason = session_health.STATUS_UNREACHABLE, f"open failed: {exc}"


def _read_tab_text(target: LoginTarget) -> Optional[str]:
    from nba_model.model.web_text_ingestion import _extract_page_text

    page = target.page
    if page is None or page.is_closed():
        return None
    return _extract_page_text(page, url=target.url)


def _verify_fresh(targets: list[LoginTarget], chrome_port: int) -> None:
    """Close the login tabs, then re-fetch each book in a fresh tab through the
    hourly fetch path (navigation + content waits) and re-classify."""
    from nba_model.model.web_text_ingestion import (
        _fetch_url_text,
        close_target_tabs_via_cdp,
    )

    close_target_tabs_via_cdp([t.url for t in targets], chrome_port)
    for t in targets:
        try:
            rec = _fetch_url_text(
                url=t.url, timeout=45, retries=0, retry_delay_seconds=0,
                retry_backoff=1.0, user_agent="", max_chars=60000,
                chrome_debug_port=chrome_port,
            )
            t.status, t.reason = classify_text(t, rec.get("text_content") or "")
            if t.status == session_health.STATUS_OK:
                t.reason = None
        except Exception as exc:  # noqa: BLE001
            t.status, t.reason = session_health.STATUS_UNREACHABLE, f"verify fetch failed: {exc}"


def _hold_hourly_lock(stack: ExitStack, args) -> bool:
    """Hold the hourly lockfile for the whole login pass (waiting up to
    ``--lock-wait-seconds`` for an in-flight hourly run). True when held."""
    from nba_model.data.hourly_update import _acquire_lock

    waited = 0.0
    while True:
        cm = _acquire_lock(args.lockfile)
        if cm.__enter__():
            stack.push(cm.__exit__)
            return True
        cm.__exit__(None, None, None)
        if waited >= args.lock_wait_seconds:
            return False
        if waited == 0:
            print("An hourly run is in progress — waiting for it to finish...", flush=True)
        time.sleep(10)
        waited += 10


def run_login_setup(args) -> int:
    from nba_model.model.web_text_ingestion import (
        _import_playwright,
        check_chrome_cdp_reachable,
        close_target_tabs_via_cdp,
        load_urls_from_file,
    )

    chrome = check_chrome_cdp_reachable(args.chrome_port)
    if not chrome["ok"]:
        print(f"Chrome CDP unreachable: {chrome['error']}\n"
              "Load the com.nba.scraping-chrome LaunchAgent first "
              "(scripts/scheduler/README.md).", file=sys.stderr)
        return EXIT_NO_CHROME

    targets = build_login_targets(
        load_urls_from_file(args.urls_file),
        state=session_health.load_state(args.state_file),
        only_books=args.books,
        blocked=session_health.load_blocked_books(args.blocked_books_file),
    )
    if not targets:
        print("No auth books to set up.")
        return EXIT_OK

    with ExitStack() as stack:
        if not args.no_lock and not _hold_hourly_lock(stack, args):
            print("Hourly lock still busy; retry after the run finishes.", file=sys.stderr)
            return EXIT_LOCKED
        hygiene = close_target_tabs_via_cdp([t.url for t in targets], args.chrome_port)
        if hygiene["closed"]:
            print(f"Closed {len(hygiene['closed'])} stale tab(s) on target hosts.")
        sync_playwright = _import_playwright()
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(f"http://localhost:{args.chrome_port}")
            context = browser.contexts[0]
            _open_tabs(context, targets, nav_timeout_ms=45_000)
            print("\nOpened login tabs in the scraping Chrome window:")
            for t in targets:
                print(f"  - {t.book:<11} {t.url}")
            print("Sign in to each one yourself; this checklist updates every "
                  f"{int(args.poll_seconds)}s (Ctrl-C to stop).\n", flush=True)
            all_ok = poll_until_ok(
                targets, _read_tab_text,
                poll_seconds=args.poll_seconds, timeout_seconds=args.timeout,
                printer=lambda text: print(text, flush=True),
            )
            try:
                browser.close()  # disconnect only; the user's Chrome stays up
            except Exception:
                pass
        if all_ok and not args.no_verify:
            print("\nAll tabs look signed in — verifying each book in a fresh tab...")
            _verify_fresh(targets, args.chrome_port)
            print(render_checklist(targets))

    left = [t for t in targets if t.status != session_health.STATUS_OK]
    if left:
        print("\nStill not ok: " + ", ".join(f"{t.book} ({t.status})" for t in left))
        return EXIT_INCOMPLETE
    print("\nAll books ok. The hourly loop will pick the sessions up on its next tick.")
    return EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="One-time book login pass for the scraping Chrome.")
    ap.add_argument("--chrome-port", type=int, default=9222)
    ap.add_argument("--urls-file", default=DEFAULT_URLS_FILE)
    ap.add_argument("--state-file", default=DEFAULT_STATE_FILE)
    ap.add_argument("--blocked-books-file", default=session_health.DEFAULT_BLOCKED_BOOKS_FILE,
                    help="Books never prompted for (geo/age/app-only blocks).")
    ap.add_argument("--lockfile", default=DEFAULT_LOCKFILE)
    ap.add_argument("--books", nargs="*", default=None, help="Restrict to these books.")
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument("--timeout", type=float, default=1800.0, help="Seconds before giving up.")
    ap.add_argument("--lock-wait-seconds", type=float, default=900.0,
                    help="Wait this long for an in-flight hourly run to finish.")
    ap.add_argument("--no-lock", action="store_true",
                    help="Don't hold the hourly lock (hourly tab hygiene may close login tabs).")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip the fresh-tab verification pass.")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    try:
        return run_login_setup(_build_parser().parse_args(argv))
    except KeyboardInterrupt:  # lock + CDP released by the context managers
        print("\nStopped. Re-run any time; the hourly loop resumes on its next tick.")
        return EXIT_INCOMPLETE


if __name__ == "__main__":
    sys.exit(main())
