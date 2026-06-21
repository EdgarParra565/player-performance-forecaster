"""Direct web-link text ingestion utilities."""

import argparse
import hashlib
import logging
import os
import threading
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

import requests
from nba_api.stats.static import players as nba_players

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.scrapers import get_scraper_for_url

logger = logging.getLogger(__name__)

DEFAULT_WEB_TEXT_REQUEST_TIMEOUT = 20
DEFAULT_WEB_TEXT_REQUEST_RETRIES = 1
DEFAULT_WEB_TEXT_REQUEST_RETRY_DELAY_SECONDS = 0.75
DEFAULT_WEB_TEXT_REQUEST_RETRY_BACKOFF = 2.0
DEFAULT_WEB_TEXT_MAX_CHARS = 60000
DEFAULT_WEB_TEXT_BROWSER_WAIT_AFTER_LOAD_SECONDS = 4.0
DEFAULT_WEB_TEXT_BROWSER_PAGE_TIMEOUT_SECONDS = 45
DEFAULT_ACTIVE_PLAYERS_OUTPUT_FILE = "data/config/active_nba_players.txt"
DEFAULT_AUTH_STATE_DIR = "data/config/auth"
DEFAULT_WEB_TEXT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

def _match_book_domain(url: str) -> Optional[str]:
    """Return the matching book domain key from the scrapers registry."""
    scraper = get_scraper_for_url(url)
    return scraper.domain if scraper is not None else None


class _VisibleTextParser(HTMLParser):
    """Extract visible text while ignoring script/style/noscript blocks."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs):  # noqa: D401
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str):  # noqa: D401
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str):  # noqa: D401
        if self._skip_depth > 0:
            return
        text = str(data).strip()
        if text:
            self._parts.append(text)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _collapse_whitespace(text: str) -> str:
    """Collapse repeated whitespace into single spaces."""
    return " ".join(str(text).split())


def _extract_visible_text(raw_html: str) -> str:
    """Extract visible text from HTML content."""
    parser = _VisibleTextParser()
    parser.feed(raw_html or "")
    parser.close()
    return _collapse_whitespace(parser.get_text())


def _parse_utc_datetime(raw_value: object) -> Optional[datetime]:
    """Parse ISO/SQLite datetime values into timezone-aware UTC datetimes."""
    if raw_value is None:
        return None
    raw = str(raw_value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"

    parsed = None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue

    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_urls(urls: list[str]) -> list[str]:
    """Normalize and deduplicate URL list while preserving order."""
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in urls:
        url = str(raw).strip()
        if not url:
            continue
        if not (url.startswith("http://") or url.startswith("https://")):
            continue
        if url in seen:
            continue
        seen.add(url)
        normalized.append(url)
    return normalized


def load_urls_from_file(path: str) -> list[str]:
    """Read URLs from newline-delimited file (blank lines/comments ignored)."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"URL file does not exist: {file_path}")

    urls = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith("#"):
            continue
        urls.append(candidate)
    return _normalize_urls(urls)


def fetch_active_nba_players_reference() -> list[dict]:
    """Fetch current active NBA players from nba_api static dataset."""
    synced_at_utc = datetime.now(timezone.utc).isoformat()
    rows = []
    for player in nba_players.get_active_players():
        player_id = player.get("id")
        player_name = str(player.get("full_name", "")).strip()
        if player_id is None or not player_name:
            continue
        rows.append(
            {
                "player_id": int(player_id),
                "player_name": player_name,
                "synced_at_utc": synced_at_utc,
            }
        )

    rows.sort(key=lambda rec: rec["player_name"])
    return rows


def sync_active_nba_players_reference(
    db_path: str = "data/database/nba_data.db",
    output_file: Optional[str] = DEFAULT_ACTIVE_PLAYERS_OUTPUT_FILE,
) -> dict:
    """Sync active NBA players into DB reference table and optional text file."""
    records = fetch_active_nba_players_reference()
    with DatabaseManager(db_path=db_path) as db:
        db_summary = db.upsert_active_players_reference(records)

    output_path = None
    if output_file:
        output_path_obj = Path(output_file)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        output_path_obj.write_text(
            "\n".join(rec["player_name"] for rec in records) + "\n",
            encoding="utf-8",
        )
        output_path = str(output_path_obj)

    return {
        "players_synced": int(len(records)),
        "db_attempted": int(db_summary.get("attempted", 0)),
        "db_written": int(db_summary.get("written", 0)),
        "output_file": output_path,
    }



# Injected into every page before navigation to remove Playwright's automation
# fingerprints that Cloudflare Turnstile (and similar challenges) use to block
# automated browsers.  Must stay a single JS expression so Playwright can eval
# it safely as an init script.
_BROWSER_STEALTH_SCRIPT = """
(function () {
  // Remove the canonical webdriver flag
  Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

  // Restore the permissions API to normal browser behaviour
  try {
    const _origQuery = window.navigator.permissions.query.bind(navigator.permissions);
    window.navigator.permissions.query = (params) =>
      params.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : _origQuery(params);
  } catch (_) {}

  // Expose a non-empty plugins list (Playwright ships with zero plugins)
  Object.defineProperty(navigator, 'plugins', { get: () => Array.from({ length: 3 }) });

  // Make sure chrome runtime object is present (expected by Cloudflare)
  if (!window.chrome) {
    Object.defineProperty(window, 'chrome', {
      value: { runtime: {} },
      writable: true,
    });
  }
})();
"""


def extract_chrome_session(url: str, auth_state_file: str) -> dict:
    """Extract a live session from the user's installed Chrome browser and save
    it as a Playwright storage_state JSON so headless fetches can reuse it.

    The user must already be logged in to ``url`` in their regular Chrome.
    Chrome must be closed (or at least not actively writing cookies) for the
    SQLite database to be readable.

    Returns a summary dict with status and path.
    """
    try:
        from pycookiecheat import chrome_cookies  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "pycookiecheat is required for Chrome session extraction. "
            "Install with: pip install pycookiecheat"
        ) from exc

    import json as _json
    from urllib.parse import urlparse

    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path  # e.g. "app.prizepicks.com"

    logger.info("Extracting Chrome cookies for %s …", url)
    try:
        raw_cookies: dict = chrome_cookies(url)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read Chrome cookies for {url}: {exc}\n"
            "Make sure Chrome is fully closed before running this command."
        ) from exc

    if not raw_cookies:
        raise RuntimeError(
            f"No cookies found for {url}. "
            "Make sure you are logged in to the site in Chrome and Chrome is closed."
        )

    # Convert to Playwright storage_state format
    cookies = []
    for name, value in raw_cookies.items():
        cookies.append({
            "name": name,
            "value": str(value),
            "domain": domain,
            "path": "/",
            "expires": -1,
            "httpOnly": False,
            "secure": parsed.scheme == "https",
            "sameSite": "Lax",
        })

    storage_state = {
        "cookies": cookies,
        "origins": [],
    }

    auth_path = Path(auth_state_file)
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(_json.dumps(storage_state, indent=2))
    logger.info("Saved %d cookies to %s", len(cookies), auth_path)

    return {
        "status": "saved",
        "auth_state_file": str(auth_path),
        "cookie_count": len(cookies),
        "domain": domain,
    }


def extract_session_via_cdp(
    url: str,
    auth_state_file: str,
    debug_port: int = 9222,
) -> dict:
    """Connect to a running Chrome with remote-debugging enabled, extract its
    cookies AND localStorage for *url*, and save a Playwright-compatible
    storage_state JSON.

    This is the most reliable way to capture a session protected by Cloudflare,
    PerimeterX, DataDome, etc., because the requests come from REAL Chrome —
    no fingerprinting mismatch.

    Prerequisites
    -------------
    Launch Chrome with::

        open -na "Google Chrome" --args \\
            --remote-debugging-port=9222 \\
            --user-data-dir=/tmp/pp-chrome-profile

    Log in to the site in that Chrome window, then call this function.
    """
    import json as _json

    sync_playwright = _import_playwright()

    auth_path = Path(auth_state_file)
    auth_path.parent.mkdir(parents=True, exist_ok=True)

    cdp_url = f"http://localhost:{debug_port}"
    logger.info("Connecting to Chrome at %s …", cdp_url)

    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp(cdp_url)
        except Exception as exc:
            raise RuntimeError(
                f"Could not connect to Chrome on port {debug_port}: {exc}\n\n"
                "Make sure Chrome is running with remote debugging enabled:\n"
                "  open -na \"Google Chrome\" --args "
                "--remote-debugging-port=9222 "
                "--user-data-dir=/tmp/pp-chrome-profile"
            ) from exc

        # Find the context / page that matches the target URL
        context = browser.contexts[0] if browser.contexts else None
        if context is None:
            raise RuntimeError("No browser context found — is a tab open in Chrome?")

        # Prefer a page already on the target domain; fall back to first page.
        from urllib.parse import urlparse as _up
        target_domain = _up(url).netloc
        page = None
        for ctx in browser.contexts:
            for pg in ctx.pages:
                if target_domain in pg.url:
                    page = pg
                    context = ctx
                    break
            if page:
                break
        if page is None:
            page = context.pages[0] if context.pages else context.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            except Exception:
                pass

        # Extract localStorage so JWT auth tokens are captured too.
        try:
            local_storage: dict = page.evaluate(
                "() => Object.fromEntries(Object.entries(localStorage))"
            )
        except Exception:
            local_storage = {}

        ls_origin = _up(url)._replace(path="", query="", fragment="").geturl()
        origins = []
        if local_storage:
            origins.append({
                "origin": ls_origin,
                "localStorage": [{"name": k, "value": str(v)} for k, v in local_storage.items()],
            })

        # Capture full storage state (cookies + localStorage).
        try:
            state = context.storage_state()
        except Exception:
            state = {"cookies": [], "origins": []}

        # Merge our localStorage into the saved state.
        if origins:
            existing_origins = {o.get("origin"): o for o in state.get("origins", [])}
            existing_origins[ls_origin] = origins[0]
            state["origins"] = list(existing_origins.values())

        auth_path.write_text(_json.dumps(state, indent=2))
        cookie_count = len(state.get("cookies", []))
        ls_count = len(local_storage)
        logger.info(
            "CDP session saved — %d cookies, %d localStorage keys → %s",
            cookie_count, ls_count, auth_path,
        )

        # Look for a PrizePicks-style auth token in localStorage.
        token_keys = [k for k in local_storage if "token" in k.lower() or "auth" in k.lower()]
        if token_keys:
            logger.info("Auth-related localStorage keys found: %s", token_keys)

        # Disconnect the CDP client (does not kill the user's Chrome).
        try:
            browser.close()
        except Exception:
            pass

    return {
        "status": "saved",
        "auth_state_file": str(auth_path),
        "cookie_count": cookie_count,
        "local_storage_keys": ls_count,
        "auth_keys_found": token_keys if "token_keys" in dir() else [],
    }


def _resolve_ip_geolocation() -> Optional[dict]:
    """Return {latitude, longitude} based on the machine's public IP.

    Uses the free ipapi.co JSON endpoint (no API key needed for low-volume use).
    Returns None on any error so callers can degrade gracefully.
    """
    try:
        resp = requests.get("https://ipapi.co/json/", timeout=5)
        data = resp.json()
        lat = float(data["latitude"])
        lon = float(data["longitude"])
        return {"latitude": lat, "longitude": lon}
    except Exception as exc:
        logger.debug("IP geolocation lookup failed: %s", exc)
        return None


def login_and_save_session(
    login_url: str,
    auth_state_file: str,
    user_data_dir: Optional[str] = None,
    timeout_seconds: int = 120,
    user_agent: str = DEFAULT_WEB_TEXT_USER_AGENT,
) -> dict:
    """
    Launch a HEADED (visible) browser so you can log in manually, then save session.

    The browser opens to login_url. You log in manually in the browser window.
    When done, close the browser or press Ctrl+C in the terminal. Session
    cookies and storage are saved to auth_state_file for future headless reuse.

    Session state is saved once when the browser window is closed.  No periodic
    CDP calls are made while the window is open so that bot-detection challenges
    (e.g. Cloudflare Turnstile) are not reset by automation activity.

    Returns summary dict with session state path and status.
    """
    _ensure_local_playwright_browsers()
    sync_playwright = _import_playwright()

    auth_path = Path(auth_state_file)
    auth_path.parent.mkdir(parents=True, exist_ok=True)

    profile_state_path = ""
    if user_data_dir:
        profile_dir = Path(user_data_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_state_path = str(profile_dir / "storage_state.json")

    saved_path = str(auth_path)

    with sync_playwright() as p:
        # Anti-detection launch args: tell the browser not to announce itself as
        # being controlled by automation — this removes the "AutomationControlled"
        # feature flag that Cloudflare Turnstile uses to identify Playwright.
        browser = p.chromium.launch(
            headless=False,
            timeout=max(5000, timeout_seconds * 1000),
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-features=ChromeWhatsNewUI",
            ],
        )
        logger.info("Login browser: using Playwright's bundled Chromium with stealth flags.")

        context_kwargs: dict = {
            "user_agent": str(user_agent).strip() or DEFAULT_WEB_TEXT_USER_AGENT,
            # Grant geolocation permission so PrizePicks (and similar sites) can
            # confirm the user is in a valid state.  Without this the browser
            # silently denies every geolocation request and the site shows
            # "ensure you are in a valid location before signing in".
            "permissions": ["geolocation"],
        }
        # Resolve the user's approximate coordinates from their public IP so the
        # browser can report a real location.  Falls back silently if the lookup
        # fails (the permission is still granted; the site may then use IP geo).
        geo = _resolve_ip_geolocation()
        if geo:
            context_kwargs["geolocation"] = geo
            logger.info("Geolocation set to lat=%.4f lon=%.4f from IP lookup.", geo["latitude"], geo["longitude"])
        else:
            logger.info("IP geolocation lookup failed — geolocation permission granted but no coords set.")
        if auth_path.exists():
            context_kwargs["storage_state"] = str(auth_path)
            logger.info("Loading existing session from %s", auth_path)
        context = browser.new_context(**context_kwargs)
        page = context.new_page()
        # Apply comprehensive stealth patches (playwright-stealth covers ~20
        # fingerprinting signals: webdriver flag, Canvas, WebGL renderer, Chrome
        # internals, plugins, languages, permissions API, etc.).  Falls back to
        # our own lightweight script if the library is unavailable.
        try:
            from playwright_stealth import stealth_sync  # type: ignore[import]
            stealth_sync(page)
            logger.info("playwright-stealth applied to login page.")
        except Exception:
            context.add_init_script(_BROWSER_STEALTH_SCRIPT)
            logger.info("playwright-stealth unavailable — using built-in stealth script.")

        print(f"\n  Browser opened to: {login_url}")
        print("  Log in manually in the browser window.")
        print("  When finished, CLOSE the browser window — session will be saved automatically.")
        print(f"  (Timeout: {timeout_seconds}s)\n")

        try:
            page.goto(login_url, wait_until="domcontentloaded")
        except Exception as exc:
            logger.warning("Initial page load error (continuing): %s", exc)

        # Wait silently for the browser to be closed by the user.  We use a
        # threading.Event driven by the browser's "disconnected" callback so
        # that no CDP calls are made while the user is interacting — periodic
        # CDP activity can be detected by Cloudflare Turnstile and reset the
        # auth challenge.
        _closed = threading.Event()
        browser.on("disconnected", lambda _b=None: _closed.set())
        _closed.wait(timeout=timeout_seconds)
        logger.info("Browser closed (or timeout reached) — saving session state.")

        # Save once after the browser has been closed / disconnected.
        try:
            context.storage_state(path=saved_path)
            if profile_state_path:
                context.storage_state(path=profile_state_path)
        except Exception:
            pass
        finally:
            try:
                context.close()
            except Exception:
                pass
            try:
                browser.close()
            except Exception:
                pass

    return {
        "status": "saved" if auth_path.exists() else "not_saved",
        "auth_state_file": saved_path,
        "profile_state_path": profile_state_path or None,
        "login_url": login_url,
    }


def _check_session_content(text: str, url: str, min_content_length: int) -> tuple[bool, str]:
    """
    Determine whether page text represents an authenticated session.

    Returns (is_valid, reason_string). Uses book-specific marker sets when
    the URL matches a known sportsbook domain; falls back to generic heuristics.
    """
    text_lower = text.lower()
    scraper = get_scraper_for_url(url)
    markers = scraper.session_markers if scraper is not None else None

    if markers and (markers.login_wall or markers.authenticated):
        # Book-specific: check for explicit login-wall phrases.
        for phrase in markers.login_wall:
            if phrase in text_lower:
                return False, f"Login-wall phrase detected: '{phrase}'"

        # Book-specific: require a minimum number of authenticated content markers.
        auth_phrases = markers.authenticated
        min_hits = int(markers.min_authenticated_hits)
        hit_count = sum(1 for phrase in auth_phrases if phrase in text_lower)
        if len(text) < min_content_length:
            return False, (
                f"Content too short ({len(text)} < {min_content_length} chars); "
                "page may still be loading"
            )
        if hit_count < min_hits:
            return False, (
                f"Only {hit_count}/{min_hits} authenticated content markers found "
                f"({', '.join(p for p in auth_phrases if p in text_lower) or 'none'})"
            )
        return True, f"Session active ({hit_count} authenticated markers present)"

    # Generic fallback: repeated login-wall keywords indicate unauthenticated state.
    generic_wall_markers = ("sign in", "log in", "create account", "loading...")
    has_login_wall = any(
        marker in text_lower
        for marker in generic_wall_markers
        if text_lower.count(marker) > 2
    )
    if len(text) < min_content_length:
        return False, f"Content too short ({len(text)} < {min_content_length} chars)"
    if has_login_wall:
        return False, "Generic login-wall markers detected (>2 occurrences)"
    return True, "Session appears active (generic check passed)"


def validate_session(
    test_url: str,
    auth_state_file: str,
    user_data_dir: Optional[str] = None,
    timeout_seconds: int = DEFAULT_WEB_TEXT_BROWSER_PAGE_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_WEB_TEXT_USER_AGENT,
    min_content_length: int = 500,
    chrome_debug_port: Optional[int] = None,
) -> dict:
    """
    Headlessly validate whether a saved session is still active.

    Loads auth_state_file in headless mode, navigates to test_url, and checks
    whether the page contains meaningful content (not just a login wall).
    Uses book-specific authenticated/login-wall markers when the URL matches a
    known sportsbook domain.

    Pass chrome_debug_port to route the check through a running real Chrome
    instead of Playwright's Chromium (required for Cloudflare/PerimeterX sites).
    """
    try:
        result = _fetch_url_text_with_browser(
            url=test_url,
            timeout=timeout_seconds,
            user_agent=user_agent,
            max_chars=DEFAULT_WEB_TEXT_MAX_CHARS,
            browser_auth_state_file=auth_state_file,
            browser_user_data_dir=user_data_dir,
            chrome_debug_port=chrome_debug_port,
        )
    except Exception as exc:
        return {
            "valid": False,
            "reason": f"Fetch failed: {exc}",
            "auth_state_file": auth_state_file,
            "test_url": test_url,
            "text_length": 0,
        }

    text = result.get("text_content", "")
    is_valid, reason = _check_session_content(text, test_url, min_content_length)

    return {
        "valid": is_valid,
        "reason": reason,
        "auth_state_file": auth_state_file,
        "test_url": test_url,
        "text_length": len(text),
        "book_domain": _match_book_domain(test_url),
    }


def _fetch_url_text_with_requests(
    url: str,
    timeout: int,
    user_agent: str,
    max_chars: int,
) -> dict:
    """Fetch one URL via HTTP requests and return normalized text payload."""
    headers = {
        "User-Agent": str(user_agent).strip() or DEFAULT_WEB_TEXT_USER_AGENT,
        "Accept": "text/html,text/plain,*/*",
    }
    response = requests.get(url, headers=headers, timeout=int(timeout))
    response.raise_for_status()

    content_type = str(response.headers.get("Content-Type", "")).strip()
    body = response.text or ""
    looks_like_html = "html" in content_type.lower() or "<html" in body.lower()
    text_content = (
        _extract_visible_text(body)
        if looks_like_html
        else _collapse_whitespace(body)
    )

    max_chars = max(0, int(max_chars))
    if max_chars > 0 and len(text_content) > max_chars:
        text_content = text_content[:max_chars]

    content_sha256 = hashlib.sha256(text_content.encode("utf-8")).hexdigest()
    return {
        "source_url": url,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "http_status": int(response.status_code),
        "content_type": content_type or None,
        "text_content": text_content,
        "text_length": int(len(text_content)),
        "content_sha256": content_sha256,
    }


def _ensure_local_playwright_browsers() -> None:
    """Point PLAYWRIGHT_BROWSERS_PATH at project-local dir when available."""
    local_browser_dir = Path(__file__).resolve().parents[2] / ".playwright-browsers"
    current_browser_path = str(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")).strip()
    if local_browser_dir.exists() and (
        not current_browser_path
        or "cursor-sandbox-cache" in current_browser_path
    ):
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(local_browser_dir)


def _import_playwright():
    """Import and return playwright sync_api, raising clear message if missing."""
    try:
        from playwright.sync_api import sync_playwright
        return sync_playwright
    except Exception as exc:
        raise RuntimeError(
            "Playwright is required for browser web ingestion. "
            "Install with: pip install playwright && playwright install chromium"
        ) from exc


def playwright_is_available() -> bool:
    """True iff the Playwright package can be imported in the current venv.

    Used by hourly/daily ETL preflight to fail fast with an actionable message
    instead of looping over every URL and retrying browser fetches that will
    never succeed (e.g. when the user accidentally ran via system python3).
    """
    try:
        _import_playwright()
        return True
    except RuntimeError:
        return False


def check_chrome_cdp_reachable(
    chrome_debug_port: int,
    host: str = "127.0.0.1",
    timeout_seconds: float = 2.0,
) -> dict:
    """Verify a CDP-enabled Chrome is reachable on ``host:port``.

    Hits ``/json/version`` (same endpoint the curl preflight uses) and returns
    ``{"ok": bool, "version": str|None, "error": str|None}``. Callers should
    short-circuit the ETL when ``ok`` is False — silently falling back to a
    Playwright-managed Chromium loses the residential-Chrome TLS fingerprint
    that Cloudflare / PerimeterX / DataDome rely on.
    """
    import json as _json
    from urllib.error import URLError
    from urllib.request import urlopen

    url = f"http://{host}:{int(chrome_debug_port)}/json/version"
    try:
        with urlopen(url, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        info = _json.loads(body) if body else {}
        version = info.get("Browser") or info.get("browser") or ""
        return {"ok": True, "version": str(version), "error": None}
    except (URLError, OSError, _json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "version": None,
            "error": f"Chrome CDP unreachable at {url}: {exc}",
        }


def _resolve_auth_state_path(
    browser_auth_state_file: Optional[str],
    browser_user_data_dir: Optional[str],
) -> tuple[str, str]:
    """Resolve auth state file and profile state paths, creating dirs as needed."""
    auth_state_path = str(browser_auth_state_file).strip() if browser_auth_state_file else ""
    user_data_dir = str(browser_user_data_dir).strip() if browser_user_data_dir else ""
    profile_state_path = ""
    if user_data_dir:
        profile_dir = Path(user_data_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_state_path = str(profile_dir / "storage_state.json")

    if not auth_state_path and profile_state_path and Path(profile_state_path).exists():
        auth_state_path = profile_state_path

    if auth_state_path and not Path(auth_state_path).exists():
        raise FileNotFoundError(
            f"Browser auth state file does not exist: {auth_state_path}"
        )
    return auth_state_path, profile_state_path


_SELECTOR_TIMEOUT_MS = 2000  # per-selector timeout (keeps total wait bounded)


def _scroll_page_for_lazy_content(page) -> None:
    """Scroll the page incrementally to trigger lazy-loaded content."""
    try:
        page.evaluate(
            """
            () => {
                const step = Math.round(window.innerHeight * 0.8);
                const max = Math.max(
                    document.body.scrollHeight,
                    document.documentElement.scrollHeight,
                    3000,
                );
                let y = 0;
                while (y < max) {
                    window.scrollTo(0, y);
                    y += step;
                }
                window.scrollTo(0, 0);
            }
            """
        )
    except Exception:
        pass


def _detect_login_wall_early(page, url: str) -> bool:
    """
    Quick check whether the page already shows a login wall.

    Called immediately after domcontentloaded, before any selector waits.
    Returns True when book-specific login-wall phrases are found so we can
    skip the expensive per-selector wait loop and avoid multi-minute hangs.
    """
    scraper = get_scraper_for_url(url)
    if scraper is None or not scraper.session_markers.login_wall:
        return False
    login_wall_phrases = scraper.session_markers.login_wall
    try:
        text = page.inner_text("body")
        text_lower = str(text or "").lower()
        return any(phrase in text_lower for phrase in login_wall_phrases)
    except Exception:
        return False


def _wait_for_dynamic_content(page, url: str, base_wait_seconds: float) -> None:
    """Apply smart wait strategies: book-specific selector waits then networkidle fallback.

    After domcontentloaded we do a fast login-wall check first.  If the page is
    already showing a login wall we return immediately — this avoids the multi-minute
    hang that would otherwise occur as every selector times out one by one (up to
    ``_SELECTOR_TIMEOUT_MS`` × len(selectors) seconds per URL).
    """
    if _detect_login_wall_early(page, url):
        logger.info("Login wall detected early for %s — skipping extended content waits.", url)
        return

    scraper = get_scraper_for_url(url)
    selectors = scraper.wait_selectors if scraper is not None else ()
    extra_wait = scraper.extra_wait_seconds if scraper is not None else 0.0
    book_domain = scraper.domain if scraper is not None else None

    selector_found = False
    for selector in selectors:
        try:
            page.wait_for_selector(selector, timeout=_SELECTOR_TIMEOUT_MS, state="attached")
            selector_found = True
            logger.info("Selector '%s' found for %s", selector, url)
            break
        except Exception:
            continue

    if not selector_found:
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass

    # Only apply extra book-specific wait when content selectors were actually
    # found.  On login-wall pages that slipped past the early check, skip the
    # extra wait to avoid stalling unnecessarily.
    effective_extra = extra_wait if selector_found else 0.0
    total_wait = base_wait_seconds + effective_extra
    if total_wait > 0:
        page.wait_for_timeout(int(total_wait * 1000))

    # Scroll to trigger lazy-loaded content (SPAs like PrizePicks render cards on scroll).
    if book_domain in {"prizepicks.com", "underdogfantasy.com"}:
        _scroll_page_for_lazy_content(page)
        try:
            page.wait_for_timeout(1500)
        except Exception:
            pass


def _extract_page_text(page, url: str = "") -> str:
    """Extract visible text from page using multiple strategies."""
    body_html = page.content() or ""
    visible_text = _extract_visible_text(body_html)

    if not visible_text or len(visible_text) < 200:
        try:
            inner = _collapse_whitespace(page.inner_text("body"))
            if len(inner) > len(visible_text):
                visible_text = inner
        except Exception:
            pass

    if not visible_text or len(visible_text) < 200:
        try:
            all_texts = page.eval_on_selector_all(
                "*:not(script):not(style):not(noscript)",
                "els => els.map(e => e.innerText || '').filter(t => t.trim()).join(' ')",
            )
            collapsed = _collapse_whitespace(all_texts or "")
            if len(collapsed) > len(visible_text):
                visible_text = collapsed
        except Exception:
            pass

    # If this URL belongs to a book whose scraper exposes a JS extractor,
    # supplement with its targeted DOM extraction (handles cases where the
    # generic HTML / innerText strategies miss CSS-module React content).
    scraper = get_scraper_for_url(url)
    if scraper is not None and scraper.js_extractor is not None:
        extra = _collapse_whitespace(scraper.js_extractor(page) or "")
        if extra and len(extra) > len(visible_text):
            logger.info(
                "%s JS extraction yielded %d chars (vs %d HTML)",
                scraper.name, len(extra), len(visible_text),
            )
            visible_text = extra
        elif extra:
            visible_text = visible_text + " " + extra

    return visible_text


def _fetch_url_text_via_cdp(
    url: str,
    chrome_debug_port: int,
    timeout: int,
    max_chars: int,
) -> dict:
    """Fetch page text by connecting to a REAL Chrome via CDP.

    All requests go through Chrome's genuine TLS stack and existing session
    cookies, so PerimeterX, DataDome, and Cloudflare cannot distinguish them
    from normal user traffic.  Chrome must be running with::

        open -na "Google Chrome" --args \\
            --remote-debugging-port=<chrome_debug_port> \\
            --user-data-dir=/tmp/pp-chrome-profile
    """
    _ensure_local_playwright_browsers()
    sync_playwright = _import_playwright()

    cdp_endpoint = f"http://localhost:{chrome_debug_port}"
    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp(cdp_endpoint)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot connect to Chrome on port {chrome_debug_port}: {exc}. "
                "Launch Chrome with: open -na \"Google Chrome\" --args "
                f"--remote-debugging-port={chrome_debug_port} "
                "--user-data-dir=/tmp/pp-chrome-profile"
            ) from exc

        context = browser.contexts[0] if browser.contexts else None
        if not context:
            raise RuntimeError("No browser context in running Chrome.")

        # Prefer an existing tab already on the target domain so we reuse its
        # fully-loaded SPA state rather than starting a cold navigation.
        from urllib.parse import urlparse as _up
        target_domain = _up(url).netloc
        existing_page = None
        for ctx in browser.contexts:
            for pg in ctx.pages:
                if target_domain in (pg.url or ""):
                    existing_page = pg
                    context = ctx
                    break
            if existing_page:
                break

        opened_new_tab = existing_page is None
        page = existing_page or context.new_page()
        visible_text = ""
        try:
            page.set_default_timeout(max(1000, int(timeout) * 1000))
            response = None
            if opened_new_tab:
                response = page.goto(url, wait_until="domcontentloaded")
                _wait_for_dynamic_content(
                    page, url, DEFAULT_WEB_TEXT_BROWSER_WAIT_AFTER_LOAD_SECONDS,
                )
            # For an already-loaded existing tab, skip the scroll/wait so we
            # don't trigger SPA re-renders that wipe the rendered content.
            visible_text = _extract_page_text(page, url=page.url or url)
        finally:
            if opened_new_tab:
                page.close()
            # Disconnect the CDP client so we don't leak a websocket to the
            # real Chrome on every URL in the ingestion loop. close() on a
            # connect_over_cdp browser only disconnects — it does NOT kill the
            # user's Chrome.
            try:
                browser.close()
            except Exception:
                pass

    max_chars = max(0, int(max_chars))
    if max_chars > 0 and len(visible_text) > max_chars:
        visible_text = visible_text[:max_chars]

    status_code = None
    try:
        status_code = int(response.status) if response else None
    except Exception:
        pass

    content_sha256 = hashlib.sha256(visible_text.encode("utf-8")).hexdigest()
    return {
        "source_url": url,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "http_status": status_code,
        "content_type": None,
        "text_content": visible_text,
        "text_length": int(len(visible_text)),
        "content_sha256": content_sha256,
        "fetch_method": f"cdp:{chrome_debug_port}",
    }


def _fetch_url_text_with_browser(
    url: str,
    timeout: int,
    user_agent: str,
    max_chars: int,
    browser_auth_state_file: Optional[str],
    browser_user_data_dir: Optional[str],
    chrome_debug_port: Optional[int] = None,
) -> dict:
    """Fetch one URL in browser context, optionally with persisted session state."""
    if chrome_debug_port is not None:
        # CDP path: real Chrome may drop the websocket mid-run (user quit
        # Chrome, OS sleep, port re-mapped). Retry once after a short backoff
        # before bubbling the failure up — anything more would risk masking a
        # genuine "Chrome is dead" condition that needs human attention.
        try:
            return _fetch_url_text_via_cdp(
                url=url,
                chrome_debug_port=chrome_debug_port,
                timeout=timeout,
                max_chars=max_chars,
            )
        except Exception as exc:  # noqa: BLE001 — surfaced via retry log below
            disconnect_markers = (
                "disconnect", "closed", "target closed", "connection refused",
                "cannot connect", "websocket", "browser has been closed",
            )
            msg = str(exc).lower()
            if any(marker in msg for marker in disconnect_markers):
                import time as _time
                _time.sleep(2.0)
                preflight = check_chrome_cdp_reachable(chrome_debug_port)
                if not preflight["ok"]:
                    raise RuntimeError(
                        "Chrome dropped mid-run and the CDP endpoint is "
                        f"still unreachable on port {chrome_debug_port}: "
                        f"{preflight['error']}. Restart Chrome with "
                        f"--remote-debugging-port={chrome_debug_port} "
                        "and re-run."
                    ) from exc
                return _fetch_url_text_via_cdp(
                    url=url,
                    chrome_debug_port=chrome_debug_port,
                    timeout=timeout,
                    max_chars=max_chars,
                )
            raise

    _ensure_local_playwright_browsers()
    sync_playwright = _import_playwright()
    auth_state_path, profile_state_path = _resolve_auth_state_path(
        browser_auth_state_file, browser_user_data_dir,
    )

    response = None
    visible_text = ""
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            timeout=max(1000, int(timeout) * 1000),
        )
        try:
            context_kwargs = {
                "user_agent": str(user_agent).strip() or DEFAULT_WEB_TEXT_USER_AGENT,
            }
            if auth_state_path:
                context_kwargs["storage_state"] = auth_state_path
            context = browser.new_context(**context_kwargs)
            try:
                page = context.new_page()
                page.set_default_timeout(max(1000, int(timeout) * 1000))

                response = page.goto(url, wait_until="domcontentloaded")
                _wait_for_dynamic_content(
                    page, url, DEFAULT_WEB_TEXT_BROWSER_WAIT_AFTER_LOAD_SECONDS,
                )
                visible_text = _extract_page_text(page, url=url)

                if profile_state_path:
                    context.storage_state(path=profile_state_path)
            finally:
                context.close()
        finally:
            browser.close()

    max_chars = max(0, int(max_chars))
    if max_chars > 0 and len(visible_text) > max_chars:
        visible_text = visible_text[:max_chars]

    status_code = None
    content_type = None
    if response is not None:
        try:
            status_code = int(response.status)
        except Exception:
            status_code = None
        try:
            content_type = str(response.headers.get("content-type", "")).strip() or None
        except Exception:
            content_type = None

    content_sha256 = hashlib.sha256(visible_text.encode("utf-8")).hexdigest()
    return {
        "source_url": url,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "http_status": status_code,
        "content_type": content_type,
        "text_content": visible_text,
        "text_length": int(len(visible_text)),
        "content_sha256": content_sha256,
    }


def _fetch_url_text(
    url: str,
    timeout: int,
    retries: int,
    retry_delay_seconds: float,
    retry_backoff: float,
    user_agent: str,
    max_chars: int,
    browser_auth_state_file: Optional[str] = None,
    browser_user_data_dir: Optional[str] = None,
    chrome_debug_port: Optional[int] = None,
) -> dict:
    """Fetch one URL with retry policy and return normalized text payload."""
    attempts = max(1, int(retries) + 1)
    delay_base = max(0.0, float(retry_delay_seconds))
    backoff = max(1.0, float(retry_backoff))
    use_browser = bool(
        str(browser_auth_state_file or "").strip()
        or str(browser_user_data_dir or "").strip()
        or chrome_debug_port is not None
    )
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            if use_browser:
                return _fetch_url_text_with_browser(
                    url=url,
                    timeout=timeout,
                    user_agent=user_agent,
                    max_chars=max_chars,
                    browser_auth_state_file=browser_auth_state_file,
                    browser_user_data_dir=browser_user_data_dir,
                    chrome_debug_port=chrome_debug_port,
                )
            return _fetch_url_text_with_requests(
                url=url,
                timeout=timeout,
                user_agent=user_agent,
                max_chars=max_chars,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                raise
            delay = delay_base * (backoff ** (attempt - 1))
            logger.warning(
                "Web text fetch failed (%s/%s) for %s: %s. Retrying in %.2fs.",
                attempt,
                attempts,
                url,
                exc,
                delay,
            )
            if delay > 0:
                time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Request failed without response")


def fetch_and_store_web_text(
    urls: list[str],
    db_path: str = "data/database/nba_data.db",
    min_hours_between_polls: Optional[float] = 24.0,
    force_poll: bool = False,
    request_timeout: int = DEFAULT_WEB_TEXT_REQUEST_TIMEOUT,
    request_retries: int = DEFAULT_WEB_TEXT_REQUEST_RETRIES,
    request_retry_delay_seconds: float = DEFAULT_WEB_TEXT_REQUEST_RETRY_DELAY_SECONDS,
    request_retry_backoff: float = DEFAULT_WEB_TEXT_REQUEST_RETRY_BACKOFF,
    max_chars: int = DEFAULT_WEB_TEXT_MAX_CHARS,
    user_agent: str = DEFAULT_WEB_TEXT_USER_AGENT,
    browser_auth_state_file: Optional[str] = None,
    browser_user_data_dir: Optional[str] = None,
    chrome_debug_port: Optional[int] = None,
) -> dict:
    """Fetch text snapshots for URLs and store into web_text_snapshots table."""
    use_browser = bool(
        str(browser_auth_state_file or "").strip()
        or str(browser_user_data_dir or "").strip()
        or chrome_debug_port is not None
    )
    normalized_urls = _normalize_urls(urls or [])
    if not normalized_urls:
        return {
            "status": "skipped",
            "reason": "No valid URLs provided.",
            "urls_received": int(len(urls or [])),
            "urls_considered": 0,
            "fetch_mode": ("browser" if use_browser else "requests"),
            "fetched_count": 0,
            "skipped_recent_count": 0,
            "failed_count": 0,
            "db_inserted": 0,
            "db_attempted": 0,
            "results": [],
        }

    min_hours = None
    if min_hours_between_polls is not None:
        min_hours = max(0.0, float(min_hours_between_polls))
    with DatabaseManager(db_path=db_path) as db:
        latest_fetch_map = db.get_latest_web_text_fetch_times(normalized_urls)

    now_utc = datetime.now(timezone.utc)
    snapshot_records = []
    results = []
    fetched_count = 0
    skipped_recent_count = 0
    failed_count = 0

    for url in normalized_urls:
        latest_raw = latest_fetch_map.get(url)
        latest_dt = _parse_utc_datetime(latest_raw)
        hours_since_latest = None
        if latest_dt is not None:
            hours_since_latest = (now_utc - latest_dt).total_seconds() / 3600.0

        should_skip_recent = bool(
            not force_poll
            and min_hours is not None
            and min_hours > 0
            and isinstance(hours_since_latest, (int, float))
            and float(hours_since_latest) < min_hours
        )
        if should_skip_recent:
            skipped_recent_count += 1
            results.append(
                {
                    "url": url,
                    "status": "skipped_recent",
                    "last_fetched_at_utc": (
                        latest_dt.isoformat() if latest_dt is not None else latest_raw
                    ),
                    "hours_since_last_fetch": round(float(hours_since_latest), 3),
                }
            )
            continue

        try:
            record = _fetch_url_text(
                url=url,
                timeout=request_timeout,
                retries=request_retries,
                retry_delay_seconds=request_retry_delay_seconds,
                retry_backoff=request_retry_backoff,
                user_agent=user_agent,
                max_chars=max_chars,
                browser_auth_state_file=browser_auth_state_file,
                browser_user_data_dir=browser_user_data_dir,
                chrome_debug_port=chrome_debug_port,
            )
            snapshot_records.append(record)
            fetched_count += 1
            results.append(
                {
                    "url": url,
                    "status": "fetched",
                    "http_status": record.get("http_status"),
                    "text_length": record.get("text_length"),
                    "content_sha256": record.get("content_sha256"),
                }
            )
        except Exception as exc:
            failed_count += 1
            results.append(
                {
                    "url": url,
                    "status": "failed",
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
            )

    db_insert_summary = {"inserted": 0, "attempted": 0}
    if snapshot_records:
        with DatabaseManager(db_path=db_path) as db:
            db_insert_summary = db.insert_web_text_snapshots(snapshot_records)

    status = "success"
    if failed_count > 0 and fetched_count == 0:
        status = "failed"
    elif failed_count > 0:
        status = "partial_success"

    return {
        "status": status,
        "urls_received": int(len(urls or [])),
        "urls_considered": int(len(normalized_urls)),
        "fetch_mode": ("browser" if use_browser else "requests"),
        "fetched_count": int(fetched_count),
        "skipped_recent_count": int(skipped_recent_count),
        "failed_count": int(failed_count),
        "min_hours_between_polls": min_hours,
        "force_poll": bool(force_poll),
        "db_inserted": int(db_insert_summary.get("inserted", 0)),
        "db_attempted": int(db_insert_summary.get("attempted", 0)),
        "results": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch/store visible text snapshots from direct website URLs."
    )
    parser.add_argument("--urls", nargs="*", default=None)
    parser.add_argument(
        "--urls-file",
        default=None,
        help="Path to newline-delimited URL list. Blank lines and # comments allowed.",
    )
    parser.add_argument("--db-path", default="data/database/nba_data.db")
    parser.add_argument(
        "--sync-active-players-ref",
        action="store_true",
        help=(
            "Sync active NBA players into nba_active_players_ref table "
            "and optional output file."
        ),
    )
    parser.add_argument(
        "--active-players-output-file",
        default=DEFAULT_ACTIVE_PLAYERS_OUTPUT_FILE,
        help="Output file path for active NBA player names when syncing reference.",
    )
    parser.add_argument("--min-hours-between-polls", type=float, default=24.0)
    parser.add_argument("--force-poll", action="store_true")
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_WEB_TEXT_REQUEST_TIMEOUT)
    parser.add_argument("--request-retries", type=int, default=DEFAULT_WEB_TEXT_REQUEST_RETRIES)
    parser.add_argument(
        "--request-retry-delay-seconds",
        type=float,
        default=DEFAULT_WEB_TEXT_REQUEST_RETRY_DELAY_SECONDS,
    )
    parser.add_argument(
        "--request-retry-backoff",
        type=float,
        default=DEFAULT_WEB_TEXT_REQUEST_RETRY_BACKOFF,
    )
    parser.add_argument(
        "--browser-auth-state-file",
        default=None,
        help=(
            "Playwright storage-state JSON file for authenticated "
            "browser session fetches."
        ),
    )
    parser.add_argument(
        "--browser-user-data-dir",
        default=None,
        help=(
            "Playwright user-data directory for persistent "
            "authenticated browser profile."
        ),
    )
    parser.add_argument("--max-chars", type=int, default=DEFAULT_WEB_TEXT_MAX_CHARS)
    parser.add_argument("--user-agent", default=DEFAULT_WEB_TEXT_USER_AGENT)

    parser.add_argument(
        "--login",
        default=None,
        metavar="URL",
        help=(
            "Launch a HEADED browser to URL for manual login. "
            "After you log in and close the browser, the session is saved to "
            "--browser-auth-state-file for future headless reuse."
        ),
    )
    parser.add_argument(
        "--extract-chrome-session",
        default=None,
        metavar="URL",
        help=(
            "Extract the live session for URL from your installed Chrome browser "
            "and save it to --browser-auth-state-file. "
            "Log in to the site in regular Chrome first, then close Chrome, "
            "then run this command."
        ),
    )
    parser.add_argument(
        "--connect-chrome",
        default=None,
        metavar="URL",
        help=(
            "Connect to a running Chrome (with --remote-debugging-port) and extract "
            "the full session (cookies + localStorage) for URL. "
            "First launch Chrome with: "
            "open -na 'Google Chrome' --args --remote-debugging-port=9222 "
            "--user-data-dir=/tmp/pp-chrome-profile"
        ),
    )
    parser.add_argument(
        "--chrome-debug-port",
        type=int,
        default=None,
        help=(
            "Chrome remote debugging port. Required for --connect-chrome "
            "(defaults to 9222 if omitted there). For --validate-session, "
            "passing this flag opts into routing the check through a running "
            "real Chrome instead of Playwright's headless Chromium."
        ),
    )
    parser.add_argument(
        "--login-timeout",
        type=int,
        default=120,
        help="Max seconds to keep the login browser open (default 120).",
    )
    parser.add_argument(
        "--validate-session",
        default=None,
        metavar="URL",
        help=(
            "Headlessly test whether --browser-auth-state-file is still valid "
            "by fetching URL and checking for meaningful content."
        ),
    )
    return parser


def main():
    args = _build_parser().parse_args()

    if args.connect_chrome:
        state_file = args.browser_auth_state_file
        if not state_file:
            state_file = str(Path(DEFAULT_AUTH_STATE_DIR) / "session_state.json")
        result = extract_session_via_cdp(
            url=args.connect_chrome,
            auth_state_file=state_file,
            debug_port=(9222 if args.chrome_debug_port is None
                        else args.chrome_debug_port),
        )
        print("CDP session extraction result:")
        for key, val in result.items():
            print(f"  {key}: {val}")
        return

    if args.extract_chrome_session:
        state_file = args.browser_auth_state_file
        if not state_file:
            state_file = str(
                Path(DEFAULT_AUTH_STATE_DIR) / "session_state.json"
            )
        result = extract_chrome_session(
            url=args.extract_chrome_session,
            auth_state_file=state_file,
        )
        print("Chrome session extraction result:")
        for key, val in result.items():
            print(f"  {key}: {val}")
        return

    if args.login:
        state_file = args.browser_auth_state_file
        if not state_file:
            state_file = str(
                Path(DEFAULT_AUTH_STATE_DIR) / "session_state.json"
            )
        result = login_and_save_session(
            login_url=args.login,
            auth_state_file=state_file,
            user_data_dir=args.browser_user_data_dir,
            timeout_seconds=args.login_timeout,
            user_agent=args.user_agent,
        )
        print("Login session save result:")
        for key, val in result.items():
            print(f"  {key}: {val}")
        return

    if args.validate_session:
        state_file = args.browser_auth_state_file
        if not state_file:
            state_file = str(
                Path(DEFAULT_AUTH_STATE_DIR) / "session_state.json"
            )
        result = validate_session(
            test_url=args.validate_session,
            auth_state_file=state_file,
            user_data_dir=args.browser_user_data_dir,
            user_agent=args.user_agent,
            chrome_debug_port=args.chrome_debug_port,
        )
        print("Session validation result:")
        for key, val in result.items():
            print(f"  {key}: {val}")
        return

    urls = list(args.urls or [])
    if args.urls_file:
        urls.extend(load_urls_from_file(args.urls_file))

    if args.sync_active_players_ref:
        sync_summary = sync_active_nba_players_reference(
            db_path=args.db_path,
            output_file=args.active_players_output_file,
        )
        print("Active players reference sync summary:")
        print(f"- players_synced: {sync_summary.get('players_synced')}")
        print(f"- db_attempted: {sync_summary.get('db_attempted')}")
        print(f"- db_written: {sync_summary.get('db_written')}")
        if sync_summary.get("output_file"):
            print(f"- output_file: {sync_summary.get('output_file')}")

    if urls:
        summary = fetch_and_store_web_text(
            urls=urls,
            db_path=args.db_path,
            min_hours_between_polls=(
                args.min_hours_between_polls
                if args.min_hours_between_polls > 0
                else None
            ),
            force_poll=args.force_poll,
            request_timeout=args.request_timeout,
            request_retries=args.request_retries,
            request_retry_delay_seconds=args.request_retry_delay_seconds,
            request_retry_backoff=args.request_retry_backoff,
            max_chars=args.max_chars,
            user_agent=args.user_agent,
            browser_auth_state_file=args.browser_auth_state_file,
            browser_user_data_dir=args.browser_user_data_dir,
        )

        print("Web text ingestion summary:")
        for key in (
            "status",
            "urls_received",
            "urls_considered",
            "fetch_mode",
            "fetched_count",
            "skipped_recent_count",
            "failed_count",
            "db_attempted",
            "db_inserted",
        ):
            print(f"- {key}: {summary.get(key)}")

        for row in summary.get("results", [])[:20]:
            print(f"- url[{row.get('url')}]: {row.get('status')}")


if __name__ == "__main__":
    main()
