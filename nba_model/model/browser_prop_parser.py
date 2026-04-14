"""Parse visible web snapshot text into structured prop-card records."""

import argparse
import hashlib
import re
from typing import Optional
from urllib.parse import urlparse

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model.web_text_ingestion import load_urls_from_file

PARSER_VERSION = "visible_text_v2"
DEFAULT_MIN_PARSE_CONFIDENCE = 0.45
DEFAULT_MAX_SNAPSHOTS_PER_URL = 1
DEFAULT_MAX_TOTAL_SNAPSHOTS = 250

_SIDE_MAP = {
    "higher": "over",
    "more": "over",
    "over": "over",
    "lower": "under",
    "less": "under",
    "under": "under",
}
_SIDE_PATTERN = r"(?:higher|lower|more|less|over|under)"

_STAT_ALIASES = {
    "points": "points",
    "point": "points",
    "pts": "points",
    "assists": "assists",
    "assist": "assists",
    "ast": "assists",
    "asts": "assists",
    "rebounds": "rebounds",
    "rebound": "rebounds",
    "reb": "rebounds",
    "rebs": "rebounds",
    "pra": "pra",
    "points rebounds assists": "pra",
    "points+rebounds+assists": "pra",
    "pts+rebs+asts": "pra",
    "points assists": "pa",
    "points+assists": "pa",
    "pts+asts": "pa",
    "pa": "pa",
    "points rebounds": "pr",
    "points+rebounds": "pr",
    "pts+rebs": "pr",
    "pr": "pr",
    "rebounds assists": "ra",
    "rebounds+assists": "ra",
    "rebs+asts": "ra",
    "ra": "ra",
    "threes": "three_pointers_made",
    "3pm": "three_pointers_made",
    "three pointers": "three_pointers_made",
    "three pointers made": "three_pointers_made",
    "steals": "steals",
    "blocks": "blocks",
    "turnovers": "turnovers",
    "fantasy points": "fantasy_points",
    "pts + rebs + asts": "pra",
    "pts rebs asts": "pra",
    "rebounds + assists": "ra",
    "points + rebounds": "pr",
    "points + assists": "pa",
    "blocks + steals": "blocks_steals",
    "3 pointers made": "three_pointers_made",
    "3-pointers made": "three_pointers_made",
    "ft made": "ft_made",
    "double doubles": "double_doubles",
    "triple doubles": "triple_doubles",
    "1q points": "1q_points",
    "1q rebounds": "1q_rebounds",
    "1q assists": "1q_assists",
    "1q 3-pointers made": "1q_three_pointers_made",
    "1q pts + rebs + asts": "1q_pra",
    "1h points": "1h_points",
    "1h 3-pointers made": "1h_three_pointers_made",
    "1h rebounds": "1h_rebounds",
    "1h pts + rebs + asts": "1h_pra",
    "3s attempted": "three_pointers_attempted",
    "fg attempted": "fg_attempted",
}

_STAT_PATTERN = "|".join(
    re.escape(item)
    for item in sorted(_STAT_ALIASES.keys(), key=len, reverse=True)
)

# Words that should never appear at the start of a player-name token.
# Defined here (before _NAME_PATTERN) so _STOP_WORD_LOOKAHEAD can reference it.
_NAME_STOP_WORDS = {
    "and",
    "nba",
    "nfl",
    "mlb",
    "pick",
    "pickem",
    "higher",
    "lower",
    "over",
    "under",
    "more",
    "less",
    "loading",
    "login",
    "signin",
    "sign",
    "edt",
    "est",
    "cdt",
    "cst",
    "pdt",
    "pst",
    "popular",
    "featured",
    "champions",
    "drafts",
    "live",
    "results",
    "rankings",
    "mobile",
    "web",
    "coming",
    "soon",
    "download",
    "app",
    "standard",
    "flex",
    "play",
    "boost",
    # sportsbook UI labels that appear before prop cards
    "projections",
    "projection",
    "board",
    "slate",
    "lineup",
    "entry",
}

# Negative-lookahead fragment: prevents any word in _NAME_STOP_WORDS from
# starting a player-name token.  Applied case-insensitively (via IGNORECASE on
# the parent regex), this stops words like "More", "Less", "Projections", "NBA"
# from being captured as the beginning of a player name — which would otherwise
# cause finditer to skip over the real player name that follows.
_STOP_WORD_LOOKAHEAD = (
    "(?!"
    + "|".join(re.escape(w) + r"\b" for w in sorted(_NAME_STOP_WORDS, key=len, reverse=True))
    + ")"
)
# (?-i:[A-Z]) turns off IGNORECASE for just this character class so the first
# letter of every player-name word must be a genuine uppercase letter.  Without
# this, IGNORECASE would let lowercase letters (e.g. "r" from "rojections")
# satisfy [A-Z] after the stop-word lookahead rejects the actual uppercase start.
_NAME_WORD = _STOP_WORD_LOOKAHEAD + r"(?-i:[A-Z])[A-Za-z\.\'\-]{1,}"
_NAME_PATTERN = rf"(?:{_NAME_WORD}(?:\s+{_NAME_WORD}){{1,3}})"

_LINE_PATTERN = r"(?:\d{1,3}(?:\.\d+)?)"

_CARD_PATTERNS = [
    re.compile(
        rf"(?P<player>{_NAME_PATTERN})\s+"
        rf"(?P<side>{_SIDE_PATTERN})\s+"
        rf"(?P<line>{_LINE_PATTERN})\s+"
        rf"(?P<stat>{_STAT_PATTERN})\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<player>{_NAME_PATTERN})\s+"
        rf"(?P<line>{_LINE_PATTERN})\s+"
        rf"(?P<stat>{_STAT_PATTERN})\s+"
        rf"(?P<side>{_SIDE_PATTERN})\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<player>{_NAME_PATTERN})\s+"
        rf"(?P<stat>{_STAT_PATTERN})\s+"
        rf"(?P<line>{_LINE_PATTERN})\s+"
        rf"(?P<side>{_SIDE_PATTERN})\b",
        flags=re.IGNORECASE,
    ),
]

# UnderDog name pattern: mixed-case words only (each word must contain at least
# one lowercase letter), so all-uppercase team abbreviations are excluded.
_UD_NAME_PAT = (
    r"(?P<player>"
    r"[A-Z][A-Za-z.\'\-]*[a-z][A-Za-z.\'\-]*"
    r"(?:\s+[A-Z][A-Za-z.\'\-]*[a-z][A-Za-z.\'\-]*){1,3}"
    r")"
)
_UNDERDOG_PROP_RE = re.compile(
    _UD_NAME_PAT
    + r"\s+[A-Z]{2,4}\s+(?:@|vs\.?)\s+[A-Z]{2,4}"
      r"\s+-\s+.*?(?:EDT|EST|CDT|CST|PDT|PST|PM|AM)\s+"
      r"(?P<line>\d{1,3}(?:\.\d+)?)"
      r"\s+(?P<stat>[A-Za-z][A-Za-z +\']{1,30}?)"
      r"\s+(?P<side>Higher|Lower|Over|Under)",
    flags=re.IGNORECASE,
)

# PrizePicks name pattern: same mixed-case requirement as UnderDog, PLUS a
# stop-word negative lookahead so that words like "Less", "More", "Projections"
# cannot start a player-name token (even though they have lowercase letters).
# Compiled WITHOUT re.IGNORECASE so [a-z] is truly case-sensitive.
_PP_STOP_LOOKAHEAD = (
    "(?!"
    + "|".join(
        # Include both lowercase and title-case forms since patterns are case-sensitive.
        re.escape(w[0].upper() + w[1:]) + r"\b"
        for w in sorted(_NAME_STOP_WORDS, key=len, reverse=True)
        if w
    )
    + ")"
)
_PRIZEPICKS_NAME_WORD = _PP_STOP_LOOKAHEAD + r"[A-Z][A-Za-z.\'\-]*[a-z][A-Za-z.\'\-]*"
_PRIZEPICKS_NAME_PAT = (
    r"(?P<player>"
    + _PRIZEPICKS_NAME_WORD
    + r"(?:\s+" + _PRIZEPICKS_NAME_WORD + r"){1,3}"
    + r")"
)
# Primary PrizePicks pattern: "Player TEAM Stat Line More|Less"
# Not IGNORECASE: case sensitivity is needed to exclude all-uppercase tokens.
_PRIZEPICKS_PROP_RE = re.compile(
    _PRIZEPICKS_NAME_PAT
    + r"\s+[A-Z]{2,4}\s+"             # team abbreviation (all-uppercase 2-4 chars)
      r"(?P<stat>[A-Z][A-Za-z +\']{1,30}?)"  # stat label (starts uppercase)
      r"\s+(?P<line>\d{1,3}(?:\.\d+)?)"
      r"\s+(?P<side>More|Less|Over|Under|Higher|Lower)",
)

# Alternate PrizePicks pattern: "Player Line Stat More|Less" (line before stat)
_PRIZEPICKS_ALT_RE = re.compile(
    _PRIZEPICKS_NAME_PAT
    + r"(?:\s+[A-Z]{2,5}(?:\s+(?:@|vs\.?)\s+[A-Z]{2,5})?)?"  # optional team context
      r"\s+(?P<line>\d{1,3}(?:\.\d+)?)"
      r"\s+(?P<stat>[A-Za-z][A-Za-z +\']{1,30}?)"
      r"\s+(?P<side>More|Less|Over|Under|Higher|Lower)",
)


def _preprocess_underdog_text(text: str) -> str:
    """
    Strip UnderDog game-info segments so generic patterns can match.

    Converts 'Player TEAM @ TEAM - TIME 11.5 Rebounds Higher 1.06x'
    into     'Player 11.5 Rebounds Higher'
    """
    segments: list[str] = []
    for m in _UNDERDOG_PROP_RE.finditer(text):
        player = m.group("player").strip()
        line = m.group("line").strip()
        stat = m.group("stat").strip()
        side = m.group("side").strip()
        segments.append(f"{player} {line} {stat} {side}")
    return " ".join(segments) if segments else ""


def _preprocess_prizepicks_text(text: str) -> str:
    """
    Strip PrizePicks team/game segments so generic patterns can match.

    Handles formats like:
      'Player TEAM Stat 27.5 More Less'
      'Player TEAM @ TEAM 27.5 Stat More Less'

    Player names are validated via ``_clean_player_name`` before being added so
    that UI labels like "Projections" that slip through the regex are filtered out.
    """
    segments: list[str] = []
    seen: set[tuple] = set()
    for pat in (_PRIZEPICKS_PROP_RE, _PRIZEPICKS_ALT_RE):
        for m in pat.finditer(text):
            raw_player = m.group("player").strip()
            player = _clean_player_name(raw_player)
            if not player:
                continue
            line = m.group("line").strip()
            stat = m.group("stat").strip()
            side = m.group("side").strip()
            key = (player.lower(), line, stat.lower())
            if key in seen:
                continue
            seen.add(key)
            segments.append(f"{player} {line} {stat} {side}")
    return " ".join(segments) if segments else ""

_NOISE_HINTS = {"loading", "login", "sign in", "create account", "mobile web is coming"}
_BOOK_NAME_MAP = {
    "prizepicks.com": "prizepicks",
    "fliff.com": "fliff",
    "kalshi.com": "kalshi",
    "underdogfantasy.com": "underdog",
    "draftkings.com": "draftkings",
    "fanduel.com": "fanduel",
    "caesars.com": "caesars",
    "betrivers.com": "betrivers",
    "betmgm.com": "betmgm",
}


def _collapse_whitespace(text: str) -> str:
    """Collapse repeated whitespace into one space."""
    return " ".join(str(text or "").split())


def _normalize_urls(urls: list[str]) -> list[str]:
    """Normalize URL list and remove duplicates while preserving order."""
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


def _normalize_name_key(name: str) -> str:
    """Create punctuation-insensitive key for player-name matching."""
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def _clean_player_name(raw_name: str) -> str:
    """Clean extracted player token into display name.

    Strips trailing team-abbreviation tokens (all-uppercase 2-4 chars, e.g. LAL,
    GSW, DEN) that the generic name pattern sometimes captures from PrizePicks
    or UnderDog text formatted as ``"Player TEAM Stat Line Side"``.
    """
    name = _collapse_whitespace(raw_name).strip(" -:|,;")
    parts = [part for part in name.split(" ") if part]
    if len(parts) < 2:
        return ""
    # Strip trailing team-abbreviation tokens (e.g. "LAL", "GSW", "DEN").
    while parts and re.match(r"^[A-Z]{2,4}$", parts[-1]):
        parts = parts[:-1]
    if len(parts) < 2:
        return ""
    lowered = [part.lower().strip(".") for part in parts]
    if any(token in _NAME_STOP_WORDS for token in lowered):
        return ""
    return " ".join(parts)


def _canonicalize_side(raw_side: str) -> Optional[str]:
    """Map side labels from page text into over/under."""
    normalized = str(raw_side or "").strip().lower()
    return _SIDE_MAP.get(normalized)


def _canonicalize_stat(raw_stat: str) -> Optional[str]:
    """Map stat labels from page text into canonical stat types."""
    normalized = str(raw_stat or "").lower()
    normalized = re.sub(r"[^a-z0-9+ ]", "", normalized)
    normalized = _collapse_whitespace(normalized)
    if normalized in _STAT_ALIASES:
        return _STAT_ALIASES[normalized]

    compact = normalized.replace(" ", "")
    for alias, canonical in _STAT_ALIASES.items():
        if alias.replace(" ", "") == compact:
            return canonical
    return None


def _parse_line_value(raw_line: str) -> Optional[float]:
    """Parse numeric line value and keep reasonable prop ranges."""
    try:
        value = float(raw_line)
    except (TypeError, ValueError):
        return None
    if value < 0 or value > 100:
        return None
    return value


def _infer_book_from_url(source_url: str) -> str:
    """Infer sportsbook label from URL hostname."""
    parsed = urlparse(str(source_url or "").strip())
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    host = host.split(":")[0]
    if host in _BOOK_NAME_MAP:
        return _BOOK_NAME_MAP[host]
    for domain, book_name in _BOOK_NAME_MAP.items():
        if host.endswith(f".{domain}"):
            return book_name
    if host:
        return host.split(".")[0]
    return "unknown"


def _compute_parse_confidence(
    player_name: str,
    stat_type: str,
    line_value: float,
    side: str,
    classification: str,
    raw_card_text: str,
) -> float:
    """Compute heuristic confidence score for extracted prop row."""
    score = 0.15
    if len(player_name.split()) >= 2:
        score += 0.20
    if stat_type:
        score += 0.20
    if line_value is not None:
        score += 0.20
    if side in {"over", "under"}:
        score += 0.10
    if classification == "active_nba":
        score += 0.20
    else:
        score += 0.05
    raw_lower = str(raw_card_text or "").lower()
    if any(token in raw_lower for token in _NOISE_HINTS):
        score -= 0.20
    return round(max(0.05, min(score, 0.99)), 3)


def _build_record_sha256(
    snapshot_id: int,
    source_url: str,
    player_name: str,
    stat_type: str,
    line_value: float,
    side: str,
    raw_card_text: str,
) -> str:
    """Build deterministic hash used for parser dedupe."""
    payload = (
        f"{snapshot_id}|{source_url}|{player_name}|{stat_type}|"
        f"{line_value:.3f}|{side}|{raw_card_text}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_prop_cards_from_text(
    text_content: str,
    source_url: str,
    snapshot_id: int,
    observed_at_utc: str,
    active_name_keys: set[str],
) -> list[dict]:
    """
    Extract prop card candidates from visible page text.

    Returns parsed rows in DB-ready shape, including classification + confidence.
    """
    text = _collapse_whitespace(text_content)
    if not text:
        return []

    preprocessed_parts = []
    ud = _preprocess_underdog_text(text)
    if ud:
        preprocessed_parts.append(ud)
    pp = _preprocess_prizepicks_text(text)
    if pp:
        preprocessed_parts.append(pp)
    if preprocessed_parts:
        text = " ".join(preprocessed_parts) + " " + text

    records: list[dict] = []
    seen_card_keys: set[tuple] = set()
    book = _infer_book_from_url(source_url)

    for pattern in _CARD_PATTERNS:
        for match in pattern.finditer(text):
            player_name = _clean_player_name(match.group("player"))
            side = _canonicalize_side(match.group("side"))
            line_value = _parse_line_value(match.group("line"))
            stat_type = _canonicalize_stat(match.group("stat"))
            if not player_name or side is None or line_value is None or stat_type is None:
                continue

            dedupe_key = (
                _normalize_name_key(player_name),
                stat_type,
                round(line_value, 3),
                side,
            )
            if dedupe_key in seen_card_keys:
                continue
            seen_card_keys.add(dedupe_key)

            player_key = _normalize_name_key(player_name)
            classification = (
                "active_nba" if player_key in active_name_keys else "non_nba"
            )
            raw_card_text = _collapse_whitespace(match.group(0))[:300]
            parse_confidence = _compute_parse_confidence(
                player_name=player_name,
                stat_type=stat_type,
                line_value=line_value,
                side=side,
                classification=classification,
                raw_card_text=raw_card_text,
            )
            records.append(
                {
                    "snapshot_id": int(snapshot_id),
                    "source_url": str(source_url).strip(),
                    "book": book,
                    "observed_at_utc": str(observed_at_utc).strip(),
                    "player_name": player_name,
                    "player_classification": classification,
                    "stat_type": stat_type,
                    "line_value": float(line_value),
                    "side": side,
                    "parse_confidence": float(parse_confidence),
                    "raw_card_text": raw_card_text,
                    "parser_version": PARSER_VERSION,
                    "record_sha256": _build_record_sha256(
                        snapshot_id=int(snapshot_id),
                        source_url=str(source_url).strip(),
                        player_name=player_name,
                        stat_type=stat_type,
                        line_value=float(line_value),
                        side=side,
                        raw_card_text=raw_card_text,
                    ),
                }
            )
    return records


def parse_and_store_web_prop_cards(
    db_path: str = "data/database/nba_data.db",
    source_urls: Optional[list[str]] = None,
    max_snapshots_per_url: int = DEFAULT_MAX_SNAPSHOTS_PER_URL,
    max_total_snapshots: int = DEFAULT_MAX_TOTAL_SNAPSHOTS,
    min_parse_confidence: float = DEFAULT_MIN_PARSE_CONFIDENCE,
) -> dict:
    """Parse recent web snapshots into structured prop-card rows."""
    normalized_urls = _normalize_urls(source_urls or [])
    with DatabaseManager(db_path=db_path) as db:
        snapshots = db.get_recent_web_text_snapshots(
            source_urls=normalized_urls if normalized_urls else None,
            max_snapshots_per_url=max_snapshots_per_url,
            limit_total=max_total_snapshots,
        )
        active_names = db.get_active_players_reference_names()

    if not snapshots:
        return {
            "status": "skipped",
            "reason": "No web_text snapshots available for parsing.",
            "urls_considered": int(len(normalized_urls)),
            "snapshots_considered": 0,
            "cards_extracted": 0,
            "cards_retained": 0,
            "db_inserted": 0,
            "db_attempted": 0,
            "active_reference_count": int(len(active_names)),
            "results": [],
        }

    threshold = float(min_parse_confidence)
    active_name_keys = {
        _normalize_name_key(name)
        for name in active_names
        if _normalize_name_key(name)
    }
    all_records: list[dict] = []
    results: list[dict] = []
    total_extracted = 0
    total_retained = 0
    retained_active = 0
    retained_non_nba = 0

    for snapshot in snapshots:
        parsed = extract_prop_cards_from_text(
            text_content=snapshot.get("text_content", ""),
            source_url=str(snapshot.get("source_url", "")),
            snapshot_id=int(snapshot.get("snapshot_id")),
            observed_at_utc=str(snapshot.get("fetched_at_utc", "")),
            active_name_keys=active_name_keys,
        )
        total_extracted += len(parsed)
        retained = [
            row for row in parsed if float(row.get("parse_confidence", 0.0)) >= threshold
        ]
        total_retained += len(retained)
        retained_active += sum(
            1 for row in retained if row.get("player_classification") == "active_nba"
        )
        retained_non_nba += sum(
            1 for row in retained if row.get("player_classification") == "non_nba"
        )
        all_records.extend(retained)
        results.append(
            {
                "snapshot_id": snapshot.get("snapshot_id"),
                "source_url": snapshot.get("source_url"),
                "book": _infer_book_from_url(str(snapshot.get("source_url", ""))),
                "extracted_count": int(len(parsed)),
                "retained_count": int(len(retained)),
            }
        )

    db_summary = {"inserted": 0, "attempted": 0}
    if all_records:
        with DatabaseManager(db_path=db_path) as db:
            db_summary = db.insert_web_prop_cards(all_records)

    status = "success"
    if total_extracted == 0:
        status = "partial_success"

    return {
        "status": status,
        "urls_considered": int(len(normalized_urls)),
        "snapshots_considered": int(len(snapshots)),
        "cards_extracted": int(total_extracted),
        "cards_retained": int(total_retained),
        "cards_retained_active_nba": int(retained_active),
        "cards_retained_non_nba": int(retained_non_nba),
        "min_parse_confidence": float(threshold),
        "db_inserted": int(db_summary.get("inserted", 0)),
        "db_attempted": int(db_summary.get("attempted", 0)),
        "active_reference_count": int(len(active_names)),
        "results": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse structured prop cards from stored web text snapshots.",
    )
    parser.add_argument("--db-path", default="data/database/nba_data.db")
    parser.add_argument("--urls", nargs="*", default=None)
    parser.add_argument("--urls-file", default=None)
    parser.add_argument(
        "--max-snapshots-per-url",
        type=int,
        default=DEFAULT_MAX_SNAPSHOTS_PER_URL,
    )
    parser.add_argument(
        "--max-total-snapshots",
        type=int,
        default=DEFAULT_MAX_TOTAL_SNAPSHOTS,
    )
    parser.add_argument(
        "--min-parse-confidence",
        type=float,
        default=DEFAULT_MIN_PARSE_CONFIDENCE,
    )
    return parser


def main() -> None:
    """CLI entry point for browser/visible-text prop parser."""
    args = _build_parser().parse_args()
    urls = list(args.urls or [])
    if args.urls_file:
        urls.extend(load_urls_from_file(args.urls_file))

    summary = parse_and_store_web_prop_cards(
        db_path=args.db_path,
        source_urls=urls,
        max_snapshots_per_url=max(1, int(args.max_snapshots_per_url)),
        max_total_snapshots=max(1, int(args.max_total_snapshots)),
        min_parse_confidence=float(args.min_parse_confidence),
    )

    print("Browser parser summary:")
    print(f"- status: {summary.get('status')}")
    print(f"- snapshots_considered: {summary.get('snapshots_considered')}")
    print(f"- cards_extracted: {summary.get('cards_extracted')}")
    print(f"- cards_retained: {summary.get('cards_retained')}")
    print(f"- db_attempted: {summary.get('db_attempted')}")
    print(f"- db_inserted: {summary.get('db_inserted')}")


if __name__ == "__main__":
    main()
