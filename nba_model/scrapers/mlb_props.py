"""Shared MLB player-prop preprocessing (WS6 — MLB-first).

DraftKings and FanDuel render MLB player props in two shapes, and this
preprocessor normalizes both into canonical
``"<Player> <line> <stat_key> <side>"`` segments (the form the DFS
preprocessors in ``pick6.py`` / ``underdog.py`` / ``nfl_props.py`` emit):

  * Over/Under count markets — "Aaron Judge Total Bases Over 1.5",
    "Aaron Judge Over 1.5 Total Bases", "Gerrit Cole Strikeouts Over 6.5".
  * Yes/No markets — "Aaron Judge To Hit A Home Run Yes" → the
    ``anytime_home_run`` market at line 0.5 (Yes → over, No → under).

MLB realities honored (see ``sports/mlb.py``):
  * Hitters and pitchers are SEPARATE populations + stat lists — use
    ``stat_group(stat_key)`` to route a parsed line to the right model.
  * Both over/under and yes/no line shapes are handled.

TODO(real-capture): authored against representative fixtures informed by the
live DK/FD MLB lobby token order; re-validate against a fresh authenticated
CDP snapshot (the per-book markup drifts). NFL/NBA-style note: the shared
``browser_prop_parser`` stat vocabulary is NBA-only, so MLB segments don't
yet flow into ``web_prop_cards`` — a sport-aware parse path is the follow-up;
this module + its tests pin the fixture-parsing behaviour.
"""

from __future__ import annotations

import re

# Hitter stat phrase (lowercased) -> canonical key (subset of sports/mlb.py).
HITTER_STAT_ALIASES: dict[str, str] = {
    "hits": "hits",
    "total bases": "total_bases",
    "home runs": "home_runs",
    "rbis": "rbis",
    "runs batted in": "rbis",
    "runs scored": "runs_scored",
    "runs": "runs_scored",
    "stolen bases": "stolen_bases",
    "walks": "walks",
    "strikeouts": "strikeouts_batter",  # batter K's; pitcher K uses "pitcher strikeouts"
    "singles": "singles",
}
# Pitcher stat phrase (lowercased) -> canonical key.
PITCHER_STAT_ALIASES: dict[str, str] = {
    "pitcher strikeouts": "strikeouts_pitcher",
    "strikeouts (pitcher)": "strikeouts_pitcher",
    "earned runs": "earned_runs",
    "outs recorded": "outs_recorded",
    "outs": "outs_recorded",
    "hits allowed": "hits_allowed",
    "walks allowed": "walks_allowed",
}
MLB_STAT_ALIASES: dict[str, str] = {**HITTER_STAT_ALIASES, **PITCHER_STAT_ALIASES}

# Yes/No markets -> canonical key (settled at the 0.5 line, Bernoulli).
MLB_YESNO_MARKETS: dict[str, str] = {
    "to hit a home run": "anytime_home_run",
    "anytime home run": "anytime_home_run",
    "home run": "anytime_home_run",
    "to record the first run": "first_run_scorer",
    "first run scorer": "first_run_scorer",
}

_HITTER_KEYS = set(HITTER_STAT_ALIASES.values())
_PITCHER_KEYS = set(PITCHER_STAT_ALIASES.values())

# A player-name word must NOT be a stat-phrase / market / side word, or the
# greedy name pattern absorbs e.g. "Pitcher" out of "Pitcher Strikeouts" and
# the wrong stat alias matches. Build the stop set from every phrase word.
_STOP_WORDS: set[str] = {"over", "under", "more", "less", "yes", "no", "a"}
for _phrase in list(MLB_STAT_ALIASES) + list(MLB_YESNO_MARKETS):
    _STOP_WORDS.update(_phrase.split())
_NAME_STOP = (
    "(?!(?:" + "|".join(re.escape(w) for w in sorted(_STOP_WORDS, key=len, reverse=True)) + r")\b)"
)
_NAME_WORD = _NAME_STOP + r"[A-Za-z][A-Za-z'\.\-]+"
_NAME = rf"{_NAME_WORD}(?:\s+{_NAME_WORD}){{1,3}}"
_LINE = r"\d{1,2}(?:\.\d+)?"
_OU_SIDE = r"(?:Over|Under|More|Less)"
_YN_SIDE = r"(?:Yes|No)"

# Longest stat phrases first so "pitcher strikeouts" beats "strikeouts".
_STAT_PATTERN = "|".join(
    re.escape(p) for p in sorted(MLB_STAT_ALIASES, key=len, reverse=True)
)
_YESNO_PATTERN = "|".join(
    re.escape(p) for p in sorted(MLB_YESNO_MARKETS, key=len, reverse=True)
)

# Over/Under, three observed orderings.
_OU_RES = [
    # "Player Stat Over 1.5"
    re.compile(
        rf"(?P<player>{_NAME})\s+(?P<stat>{_STAT_PATTERN})\s+"
        rf"(?P<side>{_OU_SIDE})\s+(?P<line>{_LINE})",
        flags=re.IGNORECASE,
    ),
    # "Player Over 1.5 Stat"
    re.compile(
        rf"(?P<player>{_NAME})\s+(?P<side>{_OU_SIDE})\s+"
        rf"(?P<line>{_LINE})\s+(?P<stat>{_STAT_PATTERN})",
        flags=re.IGNORECASE,
    ),
    # "Player 1.5 Stat Over"
    re.compile(
        rf"(?P<player>{_NAME})\s+(?P<line>{_LINE})\s+"
        rf"(?P<stat>{_STAT_PATTERN})\s+(?P<side>{_OU_SIDE})",
        flags=re.IGNORECASE,
    ),
]
# Yes/No: "Player To Hit A Home Run Yes"
_YN_RE = re.compile(
    rf"(?P<player>{_NAME})\s+(?P<market>{_YESNO_PATTERN})\s+(?P<side>{_YN_SIDE})",
    flags=re.IGNORECASE,
)

_OU_SIDE_MAP = {"over": "over", "more": "over", "under": "under", "less": "under"}
_YN_SIDE_MAP = {"yes": "over", "no": "under"}


def canonical_mlb_stat(raw: str) -> str | None:
    """Map an MLB stat phrase to its canonical key, or None when unknown."""
    return MLB_STAT_ALIASES.get(re.sub(r"\s+", " ", str(raw or "")).strip().lower())


def stat_group(stat_key: str) -> str:
    """Return 'hitting', 'pitching', or 'combined' for a canonical stat key.

    Hitters and pitchers are separate populations, so callers route a parsed
    line to the right model by group. ``anytime_home_run`` / ``first_run_scorer``
    are combined/either-population markets.
    """
    key = str(stat_key or "").strip().lower()
    if key in _PITCHER_KEYS:
        return "pitching"
    if key in _HITTER_KEYS:
        return "hitting"
    return "combined"


def preprocess_mlb_props(text: str) -> str:
    """Normalize MLB prop rows → ``"<Player> <line> <stat_key> <side>"`` segments.

    Handles both over/under count markets and yes/no markets (mapped to a
    0.5-line Bernoulli market, Yes→over / No→under). Deduped, space-joined.
    """
    text = str(text or "")
    segments: list[str] = []
    seen: set[tuple] = set()

    def _add(player: str, line: float, stat_key: str, side: str) -> None:
        player = re.sub(r"\s+", " ", player).strip()
        key = (player.lower(), round(line, 3), stat_key, side)
        if key in seen:
            return
        seen.add(key)
        segments.append(f"{player} {line} {stat_key} {side}")

    # Yes/No markets first (so "Home Run Yes" isn't half-consumed by an OU regex).
    for m in _YN_RE.finditer(text):
        market_key = MLB_YESNO_MARKETS.get(
            re.sub(r"\s+", " ", m.group("market")).strip().lower()
        )
        side = _YN_SIDE_MAP.get(m.group("side").strip().lower())
        if market_key and side:
            _add(m.group("player"), 0.5, market_key, side)

    # Over/Under count markets, across the observed orderings.
    for rx in _OU_RES:
        for m in rx.finditer(text):
            stat_key = canonical_mlb_stat(m.group("stat"))
            side = _OU_SIDE_MAP.get(m.group("side").strip().lower())
            if not (stat_key and side):
                continue
            try:
                line = float(m.group("line"))
            except ValueError:
                continue
            _add(m.group("player"), line, stat_key, side)

    return " ".join(segments)
