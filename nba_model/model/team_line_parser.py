"""Parse team-level (game) markets from web_text_snapshots.

Mirrors ``browser_prop_parser`` but for game lines (spread / total /
moneyline / team_total) instead of player props.  Per-book extractors live
on each scraper as ``team_line_extractor``; this module orchestrates: pulls
snapshots, runs each book's extractor, and writes to ``web_team_lines``.
"""

from __future__ import annotations

import argparse
import hashlib
from typing import Optional

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.scrapers import get_scraper_for_url

PARSER_VERSION = "team_lines_v1"
DEFAULT_MAX_SNAPSHOTS_PER_URL = 1
DEFAULT_MAX_TOTAL_SNAPSHOTS = 250
DEFAULT_MIN_PARSE_CONFIDENCE = 0.45


def _build_record_sha256(
    snapshot_id: int,
    source_url: str,
    book: str,
    away_team: str,
    home_team: str,
    market_type: str,
    side: str,
    line_value: Optional[float],
    odds: Optional[int],
) -> str:
    """Deterministic dedupe hash for a (snapshot, game, market, side) row."""
    line_repr = "" if line_value is None else f"{line_value:.3f}"
    odds_repr = "" if odds is None else str(int(odds))
    payload = (
        f"{snapshot_id}|{source_url}|{book}|{away_team}|{home_team}|"
        f"{market_type}|{side}|{line_repr}|{odds_repr}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compute_parse_confidence(record: dict) -> float:
    """Heuristic confidence score for a team-line record."""
    score = 0.4
    if record.get("away_team") and record.get("home_team"):
        score += 0.20
    if record.get("market_type") in {"spread", "total", "moneyline", "team_total"}:
        score += 0.15
    if record.get("odds_american") is not None:
        score += 0.15
    if record.get("market_type") in {"spread", "total"} and record.get("line_value") is not None:
        score += 0.10
    return round(min(score, 0.99), 3)


def extract_team_lines_from_snapshot(
    text_content: str,
    source_url: str,
    snapshot_id: int,
    observed_at_utc: str,
) -> list[dict]:
    """Run the matching book's extractor over a single snapshot's text."""
    scraper = get_scraper_for_url(source_url)
    if scraper is None or scraper.team_line_extractor is None:
        return []
    try:
        raw_records = scraper.team_line_extractor(text_content or "")
    except Exception:
        return []

    out: list[dict] = []
    seen: set[tuple] = set()
    for rec in raw_records:
        away = (rec.get("away_team") or "").strip()
        home = (rec.get("home_team") or "").strip()
        market = (rec.get("market_type") or "").strip()
        side = (rec.get("side") or "").strip()
        if not (away and home and market and side):
            continue
        line = rec.get("line_value")
        odds = rec.get("odds_american")
        # Dedupe within one snapshot.
        key = (away.lower(), home.lower(), market, side,
               None if line is None else round(float(line), 3),
               None if odds is None else int(odds))
        if key in seen:
            continue
        seen.add(key)

        confidence = _compute_parse_confidence(rec)
        record_sha = _build_record_sha256(
            snapshot_id=int(snapshot_id),
            source_url=str(source_url).strip(),
            book=scraper.name,
            away_team=away,
            home_team=home,
            market_type=market,
            side=side,
            line_value=None if line is None else float(line),
            odds=None if odds is None else int(odds),
        )
        out.append(
            {
                "snapshot_id": int(snapshot_id),
                "source_url": str(source_url).strip(),
                "book": scraper.name,
                "observed_at_utc": str(observed_at_utc).strip(),
                "away_team": away,
                "home_team": home,
                "market_type": market,
                "side": side,
                "team": rec.get("team"),
                "line_value": None if line is None else float(line),
                "odds_american": None if odds is None else int(odds),
                "parse_confidence": float(confidence),
                "raw_text": (rec.get("raw_text") or "")[:300],
                "parser_version": PARSER_VERSION,
                "record_sha256": record_sha,
            }
        )
    return out


def parse_and_store_web_team_lines(
    db_path: str = "data/database/nba_data.db",
    source_urls: Optional[list[str]] = None,
    max_snapshots_per_url: int = DEFAULT_MAX_SNAPSHOTS_PER_URL,
    max_total_snapshots: int = DEFAULT_MAX_TOTAL_SNAPSHOTS,
    min_parse_confidence: float = DEFAULT_MIN_PARSE_CONFIDENCE,
) -> dict:
    """Parse stored snapshots into ``web_team_lines`` rows and persist them."""
    with DatabaseManager(db_path=db_path) as db:
        snapshots = db.get_recent_web_text_snapshots(
            source_urls=list(source_urls) if source_urls else None,
            max_snapshots_per_url=max_snapshots_per_url,
            limit_total=max_total_snapshots,
        )

    if not snapshots:
        return {
            "status": "skipped",
            "snapshots_considered": 0,
            "lines_extracted": 0,
            "lines_retained": 0,
            "db_inserted": 0,
            "db_attempted": 0,
            "results": [],
        }

    threshold = float(min_parse_confidence)
    all_records: list[dict] = []
    results: list[dict] = []
    total_extracted = 0
    total_retained = 0

    for snap in snapshots:
        rows = extract_team_lines_from_snapshot(
            text_content=snap.get("text_content", ""),
            source_url=str(snap.get("source_url", "")),
            snapshot_id=int(snap.get("snapshot_id")),
            observed_at_utc=str(snap.get("fetched_at_utc", "")),
        )
        total_extracted += len(rows)
        retained = [
            r for r in rows
            if float(r.get("parse_confidence", 0.0)) >= threshold
        ]
        total_retained += len(retained)
        all_records.extend(retained)
        results.append(
            {
                "snapshot_id": snap.get("snapshot_id"),
                "source_url": snap.get("source_url"),
                "extracted_count": len(rows),
                "retained_count": len(retained),
            }
        )

    db_summary = {"inserted": 0, "attempted": 0, "skipped_unchanged": 0}
    if all_records:
        with DatabaseManager(db_path=db_path) as db:
            db_summary = db.insert_web_team_lines(all_records)

    status = "success" if total_extracted else "partial_success"
    return {
        "status": status,
        "snapshots_considered": len(snapshots),
        "lines_extracted": total_extracted,
        "lines_retained": total_retained,
        "min_parse_confidence": threshold,
        "db_inserted": int(db_summary.get("inserted", 0)),
        "db_attempted": int(db_summary.get("attempted", 0)),
        "db_skipped_unchanged": int(db_summary.get("skipped_unchanged", 0)),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse game-level team markets from stored web text snapshots.",
    )
    parser.add_argument("--db-path", default="data/database/nba_data.db")
    parser.add_argument("--urls", nargs="*", default=None)
    parser.add_argument(
        "--max-snapshots-per-url",
        type=int, default=DEFAULT_MAX_SNAPSHOTS_PER_URL,
    )
    parser.add_argument(
        "--max-total-snapshots",
        type=int, default=DEFAULT_MAX_TOTAL_SNAPSHOTS,
    )
    parser.add_argument(
        "--min-parse-confidence",
        type=float, default=DEFAULT_MIN_PARSE_CONFIDENCE,
    )
    args = parser.parse_args()
    summary = parse_and_store_web_team_lines(
        db_path=args.db_path,
        source_urls=args.urls,
        max_snapshots_per_url=max(1, int(args.max_snapshots_per_url)),
        max_total_snapshots=max(1, int(args.max_total_snapshots)),
        min_parse_confidence=float(args.min_parse_confidence),
    )
    print("Team-line parser summary:")
    for k, v in summary.items():
        if k != "results":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
