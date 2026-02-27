"""Sportsbook odds ingestion utilities (The Odds API)."""

import argparse
from collections import Counter
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from nba_api.stats.static import players as static_players

from nba_model.data.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_REQUEST_TIMEOUT = 25
DEFAULT_REQUEST_RETRIES = 2
DEFAULT_REQUEST_RETRY_DELAY_SECONDS = 0.75
DEFAULT_REQUEST_RETRY_BACKOFF = 2.0
PLAYER_PROP_MARKETS = {
    "player_points": "points",
    "player_assists": "assists",
    "player_rebounds": "rebounds",
    "player_points_rebounds_assists": "pra",
}
VALID_STAT_TYPES = set(PLAYER_PROP_MARKETS.values())


def _default_api_key() -> Optional[str]:
    """Resolve The Odds API key from common environment variable names."""
    return os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDSAPI_KEY")


def _are_player_prop_markets(markets: List[str]) -> bool:
    """Return True when all requested markets are player-prop market keys."""
    if not markets:
        return False
    allowed = set(PLAYER_PROP_MARKETS.keys())
    return all(m in allowed for m in markets)


def _coerce_int(value) -> Optional[int]:
    """Safely coerce value to int."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value) -> Optional[float]:
    """Safely coerce value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_game_date(value) -> Optional[str]:
    """Normalize candidate date input to YYYY-MM-DD."""
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")


def _dedupe_records(records: List[dict]) -> Tuple[List[dict], int]:
    """Deduplicate betting-line records and return (records, duplicates_removed)."""
    deduped = []
    seen = set()
    duplicates_removed = 0

    for rec in records:
        key = (
            rec.get("player_id"),
            rec.get("game_date"),
            rec.get("book"),
            rec.get("stat_type"),
            rec.get("line_value"),
            rec.get("over_odds"),
            rec.get("under_odds"),
        )
        if key in seen:
            duplicates_removed += 1
            continue
        seen.add(key)
        deduped.append(rec)

    return deduped, duplicates_removed


def validate_betting_line_records(records: List[dict]) -> Tuple[List[dict], dict]:
    """
    Validate and sanitize normalized betting-line records.

    Returns:
        valid_records, validation_summary
    """
    valid_records = []
    invalid_reasons = Counter()

    for rec in records:
        if not isinstance(rec, dict):
            invalid_reasons["invalid_record_type"] += 1
            continue

        player_id = _coerce_int(rec.get("player_id"))
        game_date = _normalize_game_date(rec.get("game_date"))
        book = str(rec.get("book", "")).strip()
        stat_type = str(rec.get("stat_type", "")).strip().lower()
        line_value = _coerce_float(rec.get("line_value"))
        over_odds = _coerce_int(rec.get("over_odds"))
        under_odds = _coerce_int(rec.get("under_odds"))
        player_name = str(rec.get("player_name", "")).strip() or None

        row_errors = []
        if player_id is None or player_id <= 0:
            row_errors.append("invalid_player_id")
        if game_date is None:
            row_errors.append("invalid_game_date")
        if not book:
            row_errors.append("invalid_book")
        if stat_type not in VALID_STAT_TYPES:
            row_errors.append("invalid_stat_type")
        if line_value is None:
            row_errors.append("invalid_line_value")
        if rec.get("over_odds") is not None and over_odds is None:
            row_errors.append("invalid_over_odds")
        if rec.get("under_odds") is not None and under_odds is None:
            row_errors.append("invalid_under_odds")

        if row_errors:
            for reason in row_errors:
                invalid_reasons[reason] += 1
            continue

        valid_records.append(
            {
                "player_id": int(player_id),
                "player_name": player_name,
                "game_date": game_date,
                "book": book,
                "stat_type": stat_type,
                "line_value": float(line_value),
                "over_odds": over_odds,
                "under_odds": under_odds,
            }
        )

    summary = {
        "records_received": int(len(records)),
        "records_valid": int(len(valid_records)),
        "records_invalid": int(len(records) - len(valid_records)),
        "invalid_reason_counts": dict(invalid_reasons),
    }
    return valid_records, summary


def _get_json(
    url: str,
    params: dict,
    timeout: int = DEFAULT_REQUEST_TIMEOUT,
    retries: int = DEFAULT_REQUEST_RETRIES,
    retry_delay_seconds: float = DEFAULT_REQUEST_RETRY_DELAY_SECONDS,
    retry_backoff: float = DEFAULT_REQUEST_RETRY_BACKOFF,
):
    """Execute GET request with retry policy and return parsed JSON payload."""
    attempts = max(1, int(retries) + 1)
    base_delay = max(0.0, float(retry_delay_seconds))
    backoff = max(1.0, float(retry_backoff))
    last_exc: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            snippet = exc.response.text[:500] if getattr(exc, "response", None) is not None else ""
            last_exc = requests.HTTPError(f"{exc} | response={snippet}")
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc

        if attempt < attempts:
            delay = base_delay * (backoff ** (attempt - 1))
            logger.warning(
                f"Request failed ({attempt}/{attempts}) for {url}: {last_exc}. "
                f"Retrying in {delay:.2f}s."
            )
            if delay > 0:
                time.sleep(delay)

    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown request error without captured exception")


def _extract_api_message(payload) -> Optional[str]:
    """Extract human-readable message from API dict payloads when present."""
    if not isinstance(payload, dict):
        return None
    for key in ("message", "error", "detail", "details"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def fetch_events(
    api_key: str,
    sport: str = "basketball_nba",
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    request_retries: int = DEFAULT_REQUEST_RETRIES,
    request_retry_delay_seconds: float = DEFAULT_REQUEST_RETRY_DELAY_SECONDS,
    request_retry_backoff: float = DEFAULT_REQUEST_RETRY_BACKOFF,
) -> list:
    """Fetch upcoming events for a sport from The Odds API."""
    url = f"{BASE_URL}/sports/{sport}/events"
    params = {"apiKey": api_key, "dateFormat": "iso"}
    data = _get_json(
        url,
        params,
        timeout=request_timeout,
        retries=request_retries,
        retry_delay_seconds=request_retry_delay_seconds,
        retry_backoff=request_retry_backoff,
    )
    if isinstance(data, list):
        return data
    message = _extract_api_message(data)
    if message:
        raise RuntimeError(f"The Odds API events response: {message}")
    return []


def fetch_events_from_sport_odds(
    api_key: str,
    sport: str = "basketball_nba",
    regions: str = "us",
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    request_retries: int = DEFAULT_REQUEST_RETRIES,
    request_retry_delay_seconds: float = DEFAULT_REQUEST_RETRY_DELAY_SECONDS,
    request_retry_backoff: float = DEFAULT_REQUEST_RETRY_BACKOFF,
) -> list:
    """
    Fallback event discovery using sport odds endpoint with h2h market.

    Useful when /events returns an empty list but sport odds still contains events.
    """
    url = f"{BASE_URL}/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    data = _get_json(
        url,
        params,
        timeout=request_timeout,
        retries=request_retries,
        retry_delay_seconds=request_retry_delay_seconds,
        retry_backoff=request_retry_backoff,
    )
    if isinstance(data, list):
        return data
    message = _extract_api_message(data)
    if message:
        raise RuntimeError(f"The Odds API event-discovery response: {message}")
    return []


def fetch_sport_player_props(
    api_key: str,
    sport: str = "basketball_nba",
    regions: str = "us",
    markets: Optional[List[str]] = None,
    bookmakers: Optional[List[str]] = None,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    request_retries: int = DEFAULT_REQUEST_RETRIES,
    request_retry_delay_seconds: float = DEFAULT_REQUEST_RETRY_DELAY_SECONDS,
    request_retry_backoff: float = DEFAULT_REQUEST_RETRY_BACKOFF,
) -> list:
    """
    Fetch sport-level odds payload for requested player prop markets.

    Returns a list of event payloads (each with bookmakers + markets).
    """
    url = f"{BASE_URL}/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": ",".join(markets or list(PLAYER_PROP_MARKETS.keys())),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)
    data = _get_json(
        url,
        params,
        timeout=request_timeout,
        retries=request_retries,
        retry_delay_seconds=request_retry_delay_seconds,
        retry_backoff=request_retry_backoff,
    )
    if isinstance(data, list):
        return data
    message = _extract_api_message(data)
    if message:
        raise RuntimeError(f"The Odds API sport odds response: {message}")
    return []


def fetch_event_player_props(
    api_key: str,
    event_id: str,
    sport: str = "basketball_nba",
    regions: str = "us",
    markets: Optional[List[str]] = None,
    bookmakers: Optional[List[str]] = None,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    request_retries: int = DEFAULT_REQUEST_RETRIES,
    request_retry_delay_seconds: float = DEFAULT_REQUEST_RETRY_DELAY_SECONDS,
    request_retry_backoff: float = DEFAULT_REQUEST_RETRY_BACKOFF,
) -> dict:
    """Fetch player-prop odds for one event."""
    url = f"{BASE_URL}/sports/{sport}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": ",".join(markets or list(PLAYER_PROP_MARKETS.keys())),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)
    data = _get_json(
        url,
        params,
        timeout=request_timeout,
        retries=request_retries,
        retry_delay_seconds=request_retry_delay_seconds,
        retry_backoff=request_retry_backoff,
    )
    if isinstance(data, dict):
        message = _extract_api_message(data)
        # Event payloads are dicts as well; only raise when API explicitly reports an error.
        if message and "bookmakers" not in data:
            raise RuntimeError(f"The Odds API event odds response: {message}")
        return data
    return {}


def _parse_game_date(commence_time: str) -> str:
    """Convert API commence timestamp to YYYY-MM-DD date string."""
    ts = pd.to_datetime(commence_time, errors="coerce", utc=True)
    if pd.isna(ts):
        return str(commence_time)[:10]
    return ts.strftime("%Y-%m-%d")


def _extract_player_name(outcome: dict) -> Optional[str]:
    """Extract player name from a player-prop outcome payload."""
    description = outcome.get("description")
    if description:
        return str(description).strip()

    name = str(outcome.get("name", "")).strip()
    lower = name.lower()
    if lower.endswith(" over") or lower.endswith(" under"):
        return name.rsplit(" ", 1)[0].strip()
    return None


def _resolve_player_id(player_name: str, cache: Dict[str, Optional[int]]) -> Optional[int]:
    """Resolve player full name to NBA player id with memoization."""
    key = player_name.strip()
    if key in cache:
        return cache[key]

    candidates = static_players.find_players_by_full_name(key)
    if not candidates:
        # Try a light normalization fallback.
        normalized = key.replace(".", "")
        candidates = static_players.find_players_by_full_name(normalized)

    player_id = candidates[0]["id"] if candidates else None
    cache[key] = player_id
    return player_id


def normalize_event_player_props(
    event_payload: dict,
    player_id_cache: Optional[Dict[str, Optional[int]]] = None,
) -> Tuple[List[dict], List[str]]:
    """
    Normalize one event odds payload into betting_lines-ready records.

    Returns:
        records, missing_player_names
    """
    if player_id_cache is None:
        player_id_cache = {}

    records = []
    missing_players = []
    game_date = _parse_game_date(event_payload.get("commence_time", ""))

    for bookmaker in event_payload.get("bookmakers", []):
        book_name = bookmaker.get("title") or bookmaker.get("key")
        for market in bookmaker.get("markets", []):
            stat_type = PLAYER_PROP_MARKETS.get(market.get("key"))
            if not stat_type:
                continue

            grouped = {}
            for outcome in market.get("outcomes", []):
                side = str(outcome.get("name", "")).strip().lower()
                if side not in {"over", "under"}:
                    continue

                player_name = _extract_player_name(outcome)
                line_value = outcome.get("point")
                price = outcome.get("price")
                if player_name is None or line_value is None or price is None:
                    continue

                key = (player_name, float(line_value))
                row = grouped.setdefault(
                    key,
                    {
                        "player_name": player_name,
                        "line_value": float(line_value),
                        "over_odds": None,
                        "under_odds": None,
                    },
                )
                if side == "over":
                    row["over_odds"] = int(price)
                else:
                    row["under_odds"] = int(price)

            for row in grouped.values():
                player_name = row["player_name"]
                player_id = _resolve_player_id(player_name, player_id_cache)
                if player_id is None:
                    missing_players.append(player_name)
                    continue
                records.append(
                    {
                        "player_id": player_id,
                        "player_name": player_name,
                        "game_date": game_date,
                        "book": book_name,
                        "stat_type": stat_type,
                        "line_value": row["line_value"],
                        "over_odds": row["over_odds"],
                        "under_odds": row["under_odds"],
                    }
                )
    return records, sorted(set(missing_players))


def fetch_and_store_betting_lines(
    api_key: str,
    sport: str = "basketball_nba",
    regions: str = "us",
    markets: Optional[List[str]] = None,
    bookmakers: Optional[List[str]] = None,
    db_path: str = "data/database/nba_data.db",
    max_events: Optional[int] = None,
    sleep_seconds: float = 0.15,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    request_retries: int = DEFAULT_REQUEST_RETRIES,
    request_retry_delay_seconds: float = DEFAULT_REQUEST_RETRY_DELAY_SECONDS,
    request_retry_backoff: float = DEFAULT_REQUEST_RETRY_BACKOFF,
) -> dict:
    """Fetch upcoming player props and store normalized rows in betting_lines table."""
    selected_markets = markets or list(PLAYER_PROP_MARKETS.keys())

    all_records = []
    unresolved_names = set()
    player_id_cache: Dict[str, Optional[int]] = {}
    ingestion_mode = None
    event_odds_failures = 0
    first_event_error = None
    diagnostic = None

    events = []
    should_try_sport_level = not _are_player_prop_markets(selected_markets)
    if should_try_sport_level:
        # Primary path for non-player markets: pull sport odds directly.
        try:
            events = fetch_sport_player_props(
                api_key=api_key,
                sport=sport,
                regions=regions,
                markets=selected_markets,
                bookmakers=bookmakers,
                request_timeout=request_timeout,
                request_retries=request_retries,
                request_retry_delay_seconds=request_retry_delay_seconds,
                request_retry_backoff=request_retry_backoff,
            )
        except Exception as exc:
            logger.warning(f"Sport-level odds fetch failed: {exc}")
            events = []
    else:
        logger.info("Requested markets are player props; using event-based odds ingestion flow.")

    if max_events is not None:
        events = events[:max_events]

    if events:
        ingestion_mode = "sport_odds"
        for payload in events:
            records, missing = normalize_event_player_props(payload, player_id_cache=player_id_cache)
            all_records.extend(records)
            unresolved_names.update(missing)
    else:
        # Fallback path: get event ids then pull per-event odds payloads.
        try:
            events = fetch_events(
                api_key=api_key,
                sport=sport,
                request_timeout=request_timeout,
                request_retries=request_retries,
                request_retry_delay_seconds=request_retry_delay_seconds,
                request_retry_backoff=request_retry_backoff,
            )
        except Exception as exc:
            logger.warning(f"Direct events fetch failed: {exc}")
            events = []

        if not events:
            try:
                events = fetch_events_from_sport_odds(
                    api_key=api_key,
                    sport=sport,
                    regions=regions,
                    request_timeout=request_timeout,
                    request_retries=request_retries,
                    request_retry_delay_seconds=request_retry_delay_seconds,
                    request_retry_backoff=request_retry_backoff,
                )
                if events:
                    logger.info("Discovered events via sport odds fallback (h2h).")
            except Exception as exc:
                logger.warning(f"Fallback event discovery via sport odds failed: {exc}")
        if not events:
            diagnostic = (
                "No upcoming events discovered for this sport/region at request time. "
                "This can happen during schedule gaps (e.g., All-Star break) or when API plan/region "
                "does not expose events."
            )

        if max_events is not None:
            events = events[:max_events]
        ingestion_mode = "event_odds"
        for event in events:
            event_id = event.get("id")
            if not event_id:
                continue
            try:
                payload = fetch_event_player_props(
                    api_key=api_key,
                    event_id=event_id,
                    sport=sport,
                    regions=regions,
                    markets=selected_markets,
                    bookmakers=bookmakers,
                    request_timeout=request_timeout,
                    request_retries=request_retries,
                    request_retry_delay_seconds=request_retry_delay_seconds,
                    request_retry_backoff=request_retry_backoff,
                )
            except Exception as exc:
                logger.warning(f"Skipping event {event_id}: {exc}")
                event_odds_failures += 1
                if first_event_error is None:
                    first_event_error = str(exc)
                continue

            records, missing = normalize_event_player_props(payload, player_id_cache=player_id_cache)
            all_records.extend(records)
            unresolved_names.update(missing)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    valid_records, validation_summary = validate_betting_line_records(all_records)
    deduped_records, duplicates_in_payload = _dedupe_records(valid_records)

    db_insert_summary = {
        "inserted": 0,
        "duplicates_ignored": 0,
        "attempted": 0,
    }
    with DatabaseManager(db_path=db_path) as db:
        seen_players = {}
        for row in deduped_records:
            pid = row["player_id"]
            player_name = row.get("player_name")
            if pid not in seen_players and player_name:
                seen_players[pid] = row["player_name"]
                db.insert_player(pid, row["player_name"])
        db_insert_summary = db.insert_betting_lines_records(deduped_records)

    events_with_books = sum(1 for event in events if isinstance(event, dict) and event.get("bookmakers"))

    return {
        "ingestion_mode": ingestion_mode,
        "events_processed": len(events),
        "events_with_bookmakers": events_with_books,
        "event_odds_failures": event_odds_failures,
        "first_event_error": first_event_error,
        "diagnostic": diagnostic,
        "markets_requested": selected_markets,
        "records_parsed": len(all_records),
        "records_valid": validation_summary["records_valid"],
        "records_invalid": validation_summary["records_invalid"],
        "invalid_reason_counts": validation_summary["invalid_reason_counts"],
        "duplicates_in_payload": duplicates_in_payload,
        "db_attempted": db_insert_summary.get("attempted", 0),
        "db_inserted": db_insert_summary.get("inserted", 0),
        "db_duplicates_ignored": db_insert_summary.get("duplicates_ignored", 0),
        "distinct_players": len({r["player_id"] for r in deduped_records}),
        "unresolved_player_names": sorted(unresolved_names),
    }


def fetch_odds(api_key: str, sport: str = "basketball_nba"):
    """Backward-compatible helper returning event list payload."""
    return fetch_events(api_key=api_key, sport=sport)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and store player prop odds from The Odds API.")
    parser.add_argument("--api-key", default=_default_api_key())
    parser.add_argument("--sport", default="basketball_nba")
    parser.add_argument("--regions", default="us")
    parser.add_argument("--bookmakers", nargs="*", default=None)
    parser.add_argument("--markets", nargs="*", default=list(PLAYER_PROP_MARKETS.keys()))
    parser.add_argument("--db-path", default="data/database/nba_data.db")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--request-retries", type=int, default=DEFAULT_REQUEST_RETRIES)
    parser.add_argument(
        "--request-retry-delay-seconds",
        type=float,
        default=DEFAULT_REQUEST_RETRY_DELAY_SECONDS,
    )
    parser.add_argument("--request-retry-backoff", type=float, default=DEFAULT_REQUEST_RETRY_BACKOFF)
    return parser


def main():
    args = _build_parser().parse_args()
    if not args.api_key:
        raise ValueError(
            "Missing The Odds API key. Set --api-key or one of "
            "ODDS_API_KEY / THE_ODDS_API_KEY / ODDSAPI_KEY environment variables."
        )

    summary = fetch_and_store_betting_lines(
        api_key=args.api_key,
        sport=args.sport,
        regions=args.regions,
        markets=args.markets,
        bookmakers=args.bookmakers,
        db_path=args.db_path,
        max_events=args.max_events,
        sleep_seconds=args.sleep_seconds,
        request_timeout=args.request_timeout,
        request_retries=args.request_retries,
        request_retry_delay_seconds=args.request_retry_delay_seconds,
        request_retry_backoff=args.request_retry_backoff,
    )
    print("Odds ingestion summary:")
    for key, value in summary.items():
        if isinstance(value, list):
            print(f"- {key}: {len(value)}")
            if value:
                print(f"  sample: {value[:10]}")
        else:
            print(f"- {key}: {value}")
    print(f"- ingested_at_utc: {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
