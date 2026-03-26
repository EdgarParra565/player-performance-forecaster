"""Direct web-link text ingestion utilities."""

import argparse
import hashlib
import logging
import os
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

import requests
from nba_api.stats.static import players as nba_players

from nba_model.data.database.db_manager import DatabaseManager

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

_BOOK_WAIT_SELECTORS: dict[str, list[str]] = {
    "underdogfantasy.com": [
        "[class*='pick-em']",
        "[class*='player-pick']",
        "[class*='stat-line']",
        "[data-testid*='pick']",
        "[class*='higher-lower']",
    ],
    "draftkings.com": [
        "[class*='sportsbook-outcome']",
        "[class*='player-prop']",
    ],
    "fanduel.com": [
        "[class*='player-prop']",
        "[class*='alternate-line']",
    ],
}

_BOOK_EXTRA_WAIT_SECONDS: dict[str, float] = {
    "underdogfantasy.com": 5.0,
    "draftkings.com": 4.0,
    "fanduel.com": 4.0,
}


def _match_book_domain(url: str) -> Optional[str]:
    """Return the book domain key from _BOOK_WAIT_SELECTORS that matches url."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower()
    except Exception:
        return None
    for domain in _BOOK_WAIT_SELECTORS:
        if host == domain or host.endswith(f".{domain}"):
            return domain
    return None


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
        browser = p.chromium.launch(
            headless=False,
            timeout=max(5000, timeout_seconds * 1000),
        )
        context_kwargs = {
            "user_agent": str(user_agent).strip() or DEFAULT_WEB_TEXT_USER_AGENT,
        }
        if auth_path.exists():
            context_kwargs["storage_state"] = str(auth_path)
            logger.info("Loading existing session from %s", auth_path)
        context = browser.new_context(**context_kwargs)
        page = context.new_page()

        print(f"\n  Browser opened to: {login_url}")
        print("  Log in manually in the browser window.")
        print("  When finished, CLOSE the browser window to save the session.\n")

        try:
            page.goto(login_url, wait_until="domcontentloaded")
            page.wait_for_timeout(timeout_seconds * 1000)
        except Exception:
            pass

        try:
            context.storage_state(path=saved_path)
            if profile_state_path:
                context.storage_state(path=profile_state_path)
        except Exception as exc:
            logger.warning("Could not save storage state: %s", exc)
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


def validate_session(
    test_url: str,
    auth_state_file: str,
    user_data_dir: Optional[str] = None,
    timeout_seconds: int = DEFAULT_WEB_TEXT_BROWSER_PAGE_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_WEB_TEXT_USER_AGENT,
    min_content_length: int = 500,
) -> dict:
    """
    Headlessly validate whether a saved session is still active.

    Loads auth_state_file in headless mode, navigates to test_url, and checks
    whether the page contains meaningful content (not just a login wall).
    """
    try:
        result = _fetch_url_text_with_browser(
            url=test_url,
            timeout=timeout_seconds,
            user_agent=user_agent,
            max_chars=DEFAULT_WEB_TEXT_MAX_CHARS,
            browser_auth_state_file=auth_state_file,
            browser_user_data_dir=user_data_dir,
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
    text_lower = text.lower()
    has_login_wall = any(
        marker in text_lower
        for marker in ("sign in", "log in", "create account", "loading...")
        if text_lower.count(marker) > 2
    )
    has_content = len(text) >= min_content_length and not has_login_wall

    return {
        "valid": has_content,
        "reason": (
            "Session appears active"
            if has_content
            else "Content too short or login wall detected"
        ),
        "auth_state_file": auth_state_file,
        "test_url": test_url,
        "text_length": len(text),
        "has_login_wall_markers": has_login_wall,
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


def _wait_for_dynamic_content(page, url: str, base_wait_seconds: float) -> None:
    """Apply smart wait strategies: book-specific selector waits then networkidle fallback."""
    book_domain = _match_book_domain(url)
    selectors = _BOOK_WAIT_SELECTORS.get(book_domain or "", [])
    extra_wait = _BOOK_EXTRA_WAIT_SECONDS.get(book_domain or "", 0.0)

    selector_found = False
    for selector in selectors:
        try:
            page.wait_for_selector(selector, timeout=8000, state="attached")
            selector_found = True
            logger.info("Selector '%s' found for %s", selector, url)
            break
        except Exception:
            continue

    if not selector_found:
        try:
            page.wait_for_load_state("networkidle", timeout=12000)
        except Exception:
            pass

    total_wait = base_wait_seconds + extra_wait
    if total_wait > 0:
        page.wait_for_timeout(int(total_wait * 1000))


def _extract_page_text(page) -> str:
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

    return visible_text


def _fetch_url_text_with_browser(
    url: str,
    timeout: int,
    user_agent: str,
    max_chars: int,
    browser_auth_state_file: Optional[str],
    browser_user_data_dir: Optional[str],
) -> dict:
    """Fetch one URL in browser context, optionally with persisted session state."""
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
                visible_text = _extract_page_text(page)

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
) -> dict:
    """Fetch one URL with retry policy and return normalized text payload."""
    attempts = max(1, int(retries) + 1)
    delay_base = max(0.0, float(retry_delay_seconds))
    backoff = max(1.0, float(retry_backoff))
    use_browser = bool(
        str(browser_auth_state_file or "").strip()
        or str(browser_user_data_dir or "").strip()
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
) -> dict:
    """Fetch text snapshots for URLs and store into web_text_snapshots table."""
    use_browser = bool(
        str(browser_auth_state_file or "").strip()
        or str(browser_user_data_dir or "").strip()
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
