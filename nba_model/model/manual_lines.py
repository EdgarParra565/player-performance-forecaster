"""Shared parsing helpers for the "Manual Lines Import" surfaces.

Both the Tk desktop UI (`simple_ui.SimpleModelUI`) and the Streamlit web app
need to take a free-form text blob — either pipe/CSV/TSV rows or a noisy
sportsbook board dump — and turn it into structured `betting_lines` records.

Keeping the logic here (instead of inside the Tk widget) gives the web app a
clean import surface, makes the parser unit-testable, and lets Agent B drop
the temporary fallback that imports private methods from `simple_ui`.

Public API:
    parse_manual_lines_text(text, default_game_date, default_book)
        -> (records, errors)

Records have the shape consumed by ``DatabaseManager.insert_betting_lines_records``:

    {
        "player_id":   int,
        "player_name": str,
        "game_date":   "YYYY-MM-DD",
        "book":        str,
        "stat_type":   str,
        "line_value":  float,
        "over_odds":   int | None,
        "under_odds":  int | None,
    }
"""
from __future__ import annotations

import csv
import re
from datetime import datetime
from hashlib import sha1
from typing import Optional

from nba_api.stats.static import players as static_players


_MANUAL_STAT_ALIASES: dict[str, str] = {
    "points": "points",
    "point": "points",
    "pts": "points",
    "playerpoints": "points",
    "assists": "assists",
    "assist": "assists",
    "ast": "assists",
    "playerassists": "assists",
    "rebounds": "rebounds",
    "rebound": "rebounds",
    "reb": "rebounds",
    "playerrebounds": "rebounds",
    "pra": "pra",
    "pointsreboundsassists": "pra",
    "playerpointsreboundsassists": "pra",
}


# Module-level cache so repeated parses don't re-hit nba_api lookups.
_PLAYER_LOOKUP_CACHE: dict[str, dict] = {}


def _canonical_stat_key(value: str) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _slug_stat_type(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_",
                  str(value).strip().lower()).strip("_")
    return slug or "custom_stat"


def normalize_stat_type(value: str, allow_custom: bool = False) -> str:
    """Resolve a stat label (case-insensitive, aliases) to a canonical key."""
    key = _canonical_stat_key(value)
    stat_type = _MANUAL_STAT_ALIASES.get(key)
    if not stat_type:
        if allow_custom:
            return _slug_stat_type(value)
        supported = sorted(set(_MANUAL_STAT_ALIASES.values()))
        raise ValueError(
            f"Unsupported stat '{value}'. Supported: {supported}"
        )
    return stat_type


def _parse_optional_american_odds(value) -> Optional[int]:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "null", "na", "n/a", "-"}:
        return None
    if text.lower() == "even":
        return 100
    try:
        odds = int(round(float(text)))
    except ValueError as exc:
        raise ValueError(f"Invalid American odds '{value}'") from exc
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    return odds


def _tokenize_manual_line(raw_line: str) -> list[str]:
    if "|" in raw_line:
        parts = [part.strip() for part in raw_line.split("|")]
    elif "\t" in raw_line:
        parts = [part.strip() for part in raw_line.split("\t")]
    else:
        parts = next(csv.reader([raw_line], skipinitialspace=True))
        parts = [part.strip() for part in parts]
    return [part for part in parts if part != ""]


def _looks_like_header(parts: list[str]) -> bool:
    if len(parts) < 3:
        return False
    lower_parts = [p.strip().lower() for p in parts[:5]]
    return (
        lower_parts[0] in {"player", "player_name", "name"}
        and any("stat" in token for token in lower_parts)
        and any("line" in token for token in lower_parts)
    )


def _is_date_token(value: str) -> bool:
    text = str(value).strip()
    if not text:
        return False
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            datetime.strptime(text, fmt)
            return True
        except ValueError:
            continue
    return False


def normalize_game_date(value: str) -> str:
    """Coerce date strings to ``YYYY-MM-DD``; raise on garbage."""
    token = str(value).strip()
    if not token:
        raise ValueError("Game date is required (YYYY-MM-DD)")
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(token, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(
            token.replace("Z", "+00:00")
        ).strftime("%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Invalid date '{value}'. Use YYYY-MM-DD format."
        ) from exc


def _synthetic_player_id(player_name: str) -> int:
    digest = sha1(player_name.strip().lower().encode("utf-8")).hexdigest()[:10]
    return 900_000_000 + (int(digest, 16) % 99_000_000)


def resolve_player_identity(player_name: str, allow_synthetic: bool = True) -> dict:
    """Resolve a free-form player name to an ``{player_id, player_name}`` dict.

    Uses the bundled ``nba_api`` static-player table; falls back to a stable
    synthetic id (900_xxx_xxx) for non-NBA names when ``allow_synthetic`` is on,
    so manual board imports for non-NBA sports still produce usable records.
    """
    raw_name = str(player_name).strip()
    if not raw_name:
        raise ValueError("Player name is required")
    if raw_name in _PLAYER_LOOKUP_CACHE:
        return _PLAYER_LOOKUP_CACHE[raw_name]

    candidates = static_players.find_players_by_full_name(raw_name)
    if not candidates:
        candidates = static_players.find_players_by_full_name(
            raw_name.replace(".", "")
        )
    if not candidates:
        if allow_synthetic:
            resolved = {
                "player_id": _synthetic_player_id(raw_name),
                "player_name": raw_name,
            }
            _PLAYER_LOOKUP_CACHE[raw_name] = resolved
            return resolved
        raise ValueError(f"Player '{raw_name}' not found in NBA player list")

    exact = next(
        (c for c in candidates if c.get("full_name", "").lower() == raw_name.lower()),
        candidates[0],
    )
    resolved = {
        "player_id": int(exact["id"]),
        "player_name": exact["full_name"],
    }
    _PLAYER_LOOKUP_CACHE[raw_name] = resolved
    return resolved


def _is_matchup_line(value: str) -> bool:
    token = str(value or "").strip().lower()
    return token.startswith("vs ") or token.startswith("@ ")


def _is_team_role_line(value: str) -> bool:
    token = str(value or "").strip()
    if " - " not in token:
        return False
    role = token.split(" - ", 1)[1].strip().lower()
    return any(
        key in role
        for key in (
            "attacker", "defender", "midfielder", "goalkeeper",
            "guard", "forward", "center", "wing",
        )
    )


def _is_noise_line(value: str) -> bool:
    token = str(value or "").strip()
    if not token:
        return True
    lower = token.lower()
    if re.match(r"^\d+(\.\d+)?k$", lower):
        return True
    if re.match(r"^\$\d+(\.\d+)?$", lower):
        return True
    noise_tokens = {
        "refresh board", "scoring chart", "how to play", "help center",
        "enable accessibility", "board", "my lineups", "promotions",
        "invite friends", "get $25", "ep", "demongoblin", "demon", "goblin",
        "trending", "swap", "less", "more", "privacy policy",
        "your privacy choices", "responsible gaming", "prizepicks blog",
        "terms", "careers", "contact us", "accessibility statement",
    }
    return lower in noise_tokens


def _parse_manual_line_to_record(
    parts: list[str],
    default_game_date: str,
    default_book: str,
) -> dict:
    if len(parts) < 3:
        raise ValueError("Expected at least 3 fields: player, stat, line")

    has_explicit_date = len(parts) >= 5 and _is_date_token(parts[1])
    if has_explicit_date:
        player_name = parts[0]
        game_date = normalize_game_date(parts[1])
        book = (parts[2] or default_book).strip()
        stat_token = parts[3]
        line_token = parts[4]
        over_token = parts[5] if len(parts) >= 6 else None
        under_token = parts[6] if len(parts) >= 7 else None
    else:
        player_name = parts[0]
        game_date = default_game_date
        book = default_book
        stat_token = parts[1]
        line_token = parts[2]
        over_token = parts[3] if len(parts) >= 4 else None
        under_token = parts[4] if len(parts) >= 5 else None

    if not book:
        raise ValueError("Sportsbook is required (row field or default)")
    try:
        line_value = float(line_token)
    except ValueError as exc:
        raise ValueError(f"Invalid line value '{line_token}'") from exc

    player = resolve_player_identity(player_name, allow_synthetic=True)
    stat_type = normalize_stat_type(stat_token, allow_custom=True)
    over_odds = _parse_optional_american_odds(over_token)
    under_odds = _parse_optional_american_odds(under_token)

    return {
        "player_id": player["player_id"],
        "player_name": player["player_name"],
        "game_date": game_date,
        "book": book,
        "stat_type": stat_type,
        "line_value": line_value,
        "over_odds": over_odds,
        "under_odds": under_odds,
    }


def _parse_board_style_text(
    lines: list[str],
    default_game_date: str,
    default_book: str,
) -> tuple[list[dict], list[str]]:
    """Extract repeating ``player / matchup / line / stat`` blocks from a board dump."""
    records: list[dict] = []
    errors: list[str] = []
    seen: set = set()
    number_re = re.compile(r"^\d+(?:\.\d+)?$")

    for idx, token in enumerate(lines):
        line_value_token = token.strip()
        if not number_re.match(line_value_token):
            continue
        if idx + 1 >= len(lines):
            continue

        stat_token = lines[idx + 1].strip()
        if (
            not stat_token
            or _is_noise_line(stat_token)
            or _is_matchup_line(stat_token)
        ):
            continue

        matchup_idx = None
        for j in range(idx - 1, max(-1, idx - 8), -1):
            if _is_matchup_line(lines[j]):
                matchup_idx = j
                break
        if matchup_idx is None:
            continue

        player_name = None
        for j in range(matchup_idx - 1, max(-1, matchup_idx - 8), -1):
            candidate = lines[j].strip()
            if (
                not candidate
                or _is_noise_line(candidate)
                or _is_matchup_line(candidate)
                or _is_team_role_line(candidate)
                or number_re.match(candidate)
            ):
                continue
            player_name = re.sub(
                r"(Goblin|Demon)+$", "", candidate, flags=re.IGNORECASE
            ).strip()
            if player_name:
                break
        if not player_name:
            continue

        try:
            player = resolve_player_identity(player_name, allow_synthetic=True)
            stat_type = normalize_stat_type(stat_token, allow_custom=True)
            line_value = float(line_value_token)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"board parse near '{player_name}' / '{stat_token}': {exc}"
            )
            continue

        record = {
            "player_id": player["player_id"],
            "player_name": player["player_name"],
            "game_date": default_game_date,
            "book": default_book,
            "stat_type": stat_type,
            "line_value": line_value,
            "over_odds": None,
            "under_odds": None,
        }
        dedupe_key = (
            record["player_id"], record["game_date"], record["book"],
            record["stat_type"], record["line_value"],
            record["over_odds"], record["under_odds"],
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        records.append(record)
    return records, errors


def parse_manual_lines_text(
    text: str,
    default_game_date: str,
    default_book: str,
) -> tuple[list[dict], list[str]]:
    """Parse a free-form text blob into ``betting_lines`` records + errors.

    Mixed input is supported: rows that contain ``|``, ``\\t``, or two-or-more
    commas are parsed as delimiter records; everything else falls through to
    the board-style extractor. Returns ``(records, errors)`` so callers can
    surface partial successes without raising.
    """
    records: list[dict] = []
    errors: list[str] = []
    normalized_date = normalize_game_date(default_game_date)
    normalized_book = (default_book or "").strip() or "manual_ui"
    unstructured_lines: list[str] = []

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        has_structured_delimiter = (
            ("|" in stripped) or ("\t" in stripped) or (stripped.count(",") >= 2)
        )
        if not has_structured_delimiter:
            unstructured_lines.append(stripped)
            continue
        try:
            parts = _tokenize_manual_line(stripped)
            if not parts or _looks_like_header(parts):
                continue
            record = _parse_manual_line_to_record(
                parts,
                default_game_date=normalized_date,
                default_book=normalized_book,
            )
            records.append(record)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"line {line_no}: {exc} | raw='{stripped}'")

    if unstructured_lines:
        board_records, board_errors = _parse_board_style_text(
            unstructured_lines,
            default_game_date=normalized_date,
            default_book=normalized_book,
        )
        existing = {
            (
                row["player_id"], row["game_date"], row["book"],
                row["stat_type"], row["line_value"],
                row.get("over_odds"), row.get("under_odds"),
            )
            for row in records
        }
        for row in board_records:
            key = (
                row["player_id"], row["game_date"], row["book"],
                row["stat_type"], row["line_value"],
                row.get("over_odds"), row.get("under_odds"),
            )
            if key not in existing:
                records.append(row)
                existing.add(key)
        errors.extend(board_errors)

    if not records and unstructured_lines and not errors:
        errors.append(
            "No parseable props. Use delimiter format or paste board text "
            "with player + matchup + line + stat."
        )
    return records, errors


def persist_manual_lines_records(records: list[dict], db=None) -> dict:
    """Insert parsed manual-line records into the ``betting_lines`` table.

    Thin wrapper over ``DatabaseManager.insert_betting_lines_records`` that
    keeps the Streamlit/Tk callers from having to import the DB manager
    themselves. Opens its own ``DatabaseManager`` when ``db`` is ``None``.

    Returns the insert summary dict (``inserted`` / ``duplicates_ignored`` /
    ``attempted``) from the underlying writer.
    """
    if not records:
        return {"inserted": 0, "duplicates_ignored": 0, "attempted": 0}

    if db is None:
        # Imported lazily so this module stays importable in environments
        # (tests, the Streamlit app first-load path) that haven't yet
        # configured a database.
        from nba_model.data.database.db_manager import DatabaseManager

        db = DatabaseManager()

    return db.insert_betting_lines_records(records)
