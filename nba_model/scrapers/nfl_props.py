"""Shared NFL player-prop preprocessing (WS6 scaffolding).

FanDuel and DraftKings render NFL player props with the same logical shape —
``<Player> <Stat phrase> <line> <Over|Under>`` — so both NFL scraper configs
reuse this preprocessor. It normalizes recognized NFL stat phrases to the
canonical keys in ``sports/nfl.py`` and emits canonical
``"<Player> <line> <stat_key> <side>"`` segments.

TODO(real-capture + parse-path): these were authored against representative
fixtures, NOT live authenticated NFL snapshots — capture real ones via the
Chrome :9222 CDP host. The shared ``browser_prop_parser`` stat vocabulary is
NBA-only, so NFL segments do not yet flow into ``web_prop_cards``; a
sport-aware parse path is the follow-up. This module + its tests pin the
fixture-parsing behaviour the scaffolding will plug into.
"""

from __future__ import annotations

import re

# NFL stat phrase (lowercased) -> canonical key (subset of sports/nfl.py).
NFL_STAT_ALIASES: dict[str, str] = {
    "passing yards": "passing_yards",
    "pass yards": "passing_yards",
    "passing touchdowns": "passing_touchdowns",
    "passing tds": "passing_touchdowns",
    "pass tds": "passing_touchdowns",
    "completions": "passing_completions",
    "pass completions": "passing_completions",
    "pass attempts": "passing_attempts",
    "interceptions": "interceptions",
    "rushing yards": "rushing_yards",
    "rush yards": "rushing_yards",
    "rushing attempts": "rushing_attempts",
    "carries": "rushing_attempts",
    "rushing touchdowns": "rushing_touchdowns",
    "rushing tds": "rushing_touchdowns",
    "receiving yards": "receiving_yards",
    "rec yards": "receiving_yards",
    "receptions": "receptions",
    "receiving touchdowns": "receiving_touchdowns",
    "receiving tds": "receiving_touchdowns",
    "longest reception": "longest_reception",
    "longest rush": "longest_rush",
    "kicking points": "kicking_points",
    "field goals made": "field_goals_made",
}

# Longest phrases first so "passing touchdowns" wins over "passing".
_STAT_PATTERN = "|".join(
    re.escape(p) for p in sorted(NFL_STAT_ALIASES, key=len, reverse=True)
)
# Full "First Last" style names (allow middle name / Jr. / III).
_NAME = r"[A-Z][A-Za-z'\.\-]+(?:\s+[A-Z][A-Za-z'\.\-]+){1,3}"
_LINE = r"\d{1,3}(?:\.\d+)?"
_SIDE = r"(?:Over|Under|More|Less)"

_ROW_RE = re.compile(
    rf"(?P<player>{_NAME})\s+"
    rf"(?P<stat>{_STAT_PATTERN})\s+"
    rf"(?P<line>{_LINE})\s+"
    rf"(?P<side>{_SIDE})",
    flags=re.IGNORECASE,
)

_SIDE_MAP = {"over": "over", "more": "over", "under": "under", "less": "under"}


def canonical_nfl_stat(raw: str) -> str | None:
    """Map an NFL stat phrase to its canonical key, or None when unknown."""
    return NFL_STAT_ALIASES.get(re.sub(r"\s+", " ", str(raw or "")).strip().lower())


def preprocess_nfl_props(text: str) -> str:
    """Normalize NFL prop rows into ``"<Player> <line> <stat_key> <side>"`` segments.

    Returns a single space-joined string of canonical segments (deduped),
    mirroring the DFS preprocessors in ``pick6.py`` / ``underdog.py``.
    """
    segments: list[str] = []
    seen: set[tuple] = set()
    for m in _ROW_RE.finditer(text or ""):
        stat_key = canonical_nfl_stat(m.group("stat"))
        if stat_key is None:
            continue
        try:
            line = float(m.group("line"))
        except ValueError:
            continue
        side = _SIDE_MAP.get(m.group("side").strip().lower())
        if side is None:
            continue
        player = re.sub(r"\s+", " ", m.group("player")).strip()
        key = (player.lower(), round(line, 3), stat_key, side)
        if key in seen:
            continue
        seen.add(key)
        segments.append(f"{player} {line} {stat_key} {side}")
    return " ".join(segments)
