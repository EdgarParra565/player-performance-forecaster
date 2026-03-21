"""Direct web-link text ingestion utilities (no API key required)."""

import argparse
import hashlib
import logging
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
DEFAULT_ACTIVE_PLAYERS_OUTPUT_FILE = "data/config/active_nba_players.txt"
DEFAULT_WEB_TEXT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


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


def _fetch_url_text(
    url: str,
    timeout: int,
    retries: int,
    retry_delay_seconds: float,
    retry_backoff: float,
    user_agent: str,
    max_chars: int,
) -> dict:
    """Fetch one URL with retry policy and return normalized text payload."""
    attempts = max(1, int(retries) + 1)
    delay_base = max(0.0, float(retry_delay_seconds))
    backoff = max(1.0, float(retry_backoff))

    headers = {
        "User-Agent": str(user_agent).strip() or DEFAULT_WEB_TEXT_USER_AGENT,
        "Accept": "text/html,text/plain,*/*",
    }
    last_error: Optional[Exception] = None
    response = None

    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, headers=headers, timeout=int(timeout))
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt < attempts:
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
            else:
                raise

    if response is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Request failed without response")

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
) -> dict:
    """Fetch text snapshots for URLs and store into web_text_snapshots table."""
    normalized_urls = _normalize_urls(urls or [])
    if not normalized_urls:
        return {
            "status": "skipped",
            "reason": "No valid URLs provided.",
            "urls_received": int(len(urls or [])),
            "urls_considered": 0,
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
    parser.add_argument("--max-chars", type=int, default=DEFAULT_WEB_TEXT_MAX_CHARS)
    parser.add_argument("--user-agent", default=DEFAULT_WEB_TEXT_USER_AGENT)
    return parser


def main():
    args = _build_parser().parse_args()

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
        )

        print("Web text ingestion summary:")
        for key in (
            "status",
            "urls_received",
            "urls_considered",
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
