"""Shared NBA player-name resolver.

Different scrapers + APIs render the same player differently:
  - PrizePicks / Underdog / canonical: ``"Jalen Brunson"``
  - Pick6 (DFS pickem): ``"J. Brunson"``
  - Some odds feeds: ``"Brunson, Jalen"`` or ``"J Brunson"``
  - Old data: ``"C.J. McCollum"`` vs ``"CJ McCollum"``, ``"Tim Hardaway Jr"`` vs
    ``"Tim Hardaway Jr."``

Joining player-prop / betting-line / web-prop tables by name fractures on
these variants and silently drops rows from the cross-book consensus and
chart pipelines.  This module centralizes the resolution:

  - ``normalize_name_key(s)`` — case- and punctuation-insensitive comparison
    key (mirrors ``browser_prop_parser._normalize_name_key`` so callers can
    swap to this without changing behaviour).
  - ``resolve_player_name(raw, active_names=...)`` — returns the canonical
    full name for *any* of the variants above when there's a unique match
    in the active-players reference; ``None`` when ambiguous or unknown.

The Pick6 abbreviated-name expansion that lives in
``nba_model/scrapers/pick6.py`` is a thin wrapper over this so other
scrapers can reuse the same expansion without copy-pasting.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Iterable, Optional


_SUFFIX_RE = re.compile(r"\b(?:jr|sr|ii|iii|iv|v)\.?$", flags=re.IGNORECASE)


def normalize_name_key(name: str) -> str:
    """Lowercase + strip non-alphanumerics for tolerant comparisons.

    Matches the existing key in ``browser_prop_parser._normalize_name_key``
    so callers can centralize without changing join behaviour.
    """
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def _strip_suffix(name: str) -> str:
    """Drop a trailing ``Jr`` / ``Sr`` / Roman numeral suffix for matching."""
    return _SUFFIX_RE.sub("", str(name or "").strip()).strip()


def _split_first_last(name: str) -> tuple[str, str]:
    """Return ``(first, last)`` from a free-form ``"First Last"`` string.

    Handles ``"Last, First"`` too.  Multi-word last names ("Van Gundy",
    "Trail Blazers" — irrelevant for players but defensive) are joined
    back into ``last``.
    """
    text = str(name or "").strip()
    if "," in text:
        # "Brunson, Jalen" form
        last, _, first = text.partition(",")
        return (first.strip(), last.strip())
    parts = [p for p in re.split(r"\s+", text) if p]
    if not parts:
        return ("", "")
    return (parts[0], " ".join(parts[1:]))


def _looks_like_initial(token: str) -> bool:
    """True for tokens like 'J' or 'J.' (single letter ± dot)."""
    t = str(token or "").strip().rstrip(".")
    return len(t) == 1 and t.isalpha()


def resolve_player_name(
    raw: str,
    active_names: Iterable[str],
) -> Optional[str]:
    """Map a free-form / abbreviated name to a canonical active-player name.

    Resolution strategy (each step is checked in order; first unique match
    wins, ambiguous matches return ``None``):

    1. Exact case-insensitive match (after punctuation stripping).
    2. Suffix-stripped match (``"Tim Hardaway Jr"`` ↔ ``"Tim Hardaway"``).
    3. Initial + surname (``"J. Brunson"`` matches the single active player
       whose first name starts with ``J`` and last name is ``Brunson``).
    4. Surname-only when unique (e.g. ``"Wembanyama"`` → ``"Victor Wembanyama"``
       — only one Wembanyama in the league).
    """
    if not raw:
        return None
    raw_stripped = str(raw).strip()
    if not raw_stripped:
        return None

    active = [n for n in active_names if n]
    if not active:
        return None

    raw_key = normalize_name_key(raw_stripped)
    raw_no_suffix_key = normalize_name_key(_strip_suffix(raw_stripped))

    by_key = {}
    by_key_no_suffix = {}
    by_lastname: dict[str, list[str]] = {}
    by_initial_last: dict[tuple[str, str], list[str]] = {}
    for name in active:
        k = normalize_name_key(name)
        if k:
            by_key.setdefault(k, name)
        kns = normalize_name_key(_strip_suffix(name))
        if kns:
            by_key_no_suffix.setdefault(kns, name)
        first, last = _split_first_last(_strip_suffix(name))
        if last:
            by_lastname.setdefault(normalize_name_key(last), []).append(name)
            if first:
                by_initial_last.setdefault(
                    (first[0].lower(), normalize_name_key(last)),
                    [],
                ).append(name)

    # 1. Exact key match.
    direct = by_key.get(raw_key)
    if direct:
        return direct

    # 2. Suffix-stripped match.
    direct_ns = by_key_no_suffix.get(raw_no_suffix_key)
    if direct_ns:
        return direct_ns

    first, last = _split_first_last(_strip_suffix(raw_stripped))
    first_key = normalize_name_key(first)
    last_key = normalize_name_key(last)

    # 3a. "Last, First" form was normalized by ``_split_first_last`` already,
    #     but the resulting concatenation differs from the canonical
    #     ``"First Last"``.  Try the reversed concatenation too.
    if first_key and last_key:
        reversed_key = first_key + last_key
        m = by_key.get(reversed_key) or by_key_no_suffix.get(reversed_key)
        if m:
            return m

    # 3b. Initial + surname (e.g. "J. Brunson", "J Brunson").
    if first and last_key and _looks_like_initial(first):
        matches = by_initial_last.get((first[0].lower(), last_key), [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None  # ambiguous initial — refuse to guess

    # 4. Single-token input: treat it as the surname and resolve when unique.
    #    Covers ``"Wembanyama"`` → ``"Victor Wembanyama"``, ``"Brunson"`` →
    #    ``"Jalen Brunson"``.  ``_split_first_last`` puts a single token in
    #    ``first`` and leaves ``last`` empty, so we have to check both
    #    orientations.
    sole_token_key = last_key if last_key and not first_key else first_key
    if sole_token_key and not (first_key and last_key):
        matches = by_lastname.get(sole_token_key, [])
        if len(matches) == 1:
            return matches[0]

    # 5. Surname-only when explicit ``(first='', last='Foo')`` form was used.
    if last_key and not first_key:
        matches = by_lastname.get(last_key, [])
        if len(matches) == 1:
            return matches[0]

    return None


def load_active_player_names(
    db_path: str = "data/database/nba_data.db",
) -> list[str]:
    """Load active-player names from ``nba_active_players_ref``.

    Tiny helper so resolver callers don't have to repeat the SQL.  Returns
    an empty list rather than raising when the DB / table is missing — the
    parsers degrade gracefully (they'll skip name expansion).
    """
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT player_name FROM nba_active_players_ref"
        ).fetchall()
        conn.close()
    except Exception:
        return []
    return [r[0] for r in rows if r and r[0]]


__all__ = [
    "normalize_name_key",
    "resolve_player_name",
    "load_active_player_names",
]
