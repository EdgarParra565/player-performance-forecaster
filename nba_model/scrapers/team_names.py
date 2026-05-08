"""NBA team-name normalization.

Different sportsbooks render the same team in different forms — "Knicks",
"NY Knicks", "NYK", "New York Knicks" — so cross-book consensus needs a
single canonical short name (e.g. "Knicks") to GROUP BY on.
"""

from __future__ import annotations

import re

# Canonical short name (key) → tuple of every form we've seen on book pages.
# Lowercase comparison is used everywhere, so casing here is just for clarity.
_NBA_TEAMS: dict[str, tuple[str, ...]] = {
    "Hawks":         ("Hawks", "ATL Hawks", "Atlanta Hawks", "Atlanta", "ATL"),
    "Celtics":       ("Celtics", "BOS Celtics", "Boston Celtics", "Boston", "BOS"),
    "Nets":          ("Nets", "BKN Nets", "Brooklyn Nets", "Brooklyn", "BKN", "BRK"),
    "Hornets":       ("Hornets", "CHA Hornets", "Charlotte Hornets", "Charlotte", "CHA"),
    "Bulls":         ("Bulls", "CHI Bulls", "Chicago Bulls", "Chicago", "CHI"),
    "Cavaliers":     ("Cavaliers", "CLE Cavaliers", "Cleveland Cavaliers",
                      "Cleveland", "CLE", "Cavs"),
    "Mavericks":     ("Mavericks", "DAL Mavericks", "Dallas Mavericks",
                      "Dallas", "DAL", "Mavs"),
    "Nuggets":       ("Nuggets", "DEN Nuggets", "Denver Nuggets", "Denver", "DEN"),
    "Pistons":       ("Pistons", "DET Pistons", "Detroit Pistons", "Detroit", "DET"),
    "Warriors":      ("Warriors", "GS Warriors", "GSW Warriors", "Golden State Warriors",
                      "Golden State", "GSW", "GS"),
    "Rockets":       ("Rockets", "HOU Rockets", "Houston Rockets", "Houston", "HOU"),
    "Pacers":        ("Pacers", "IND Pacers", "Indiana Pacers", "Indiana", "IND"),
    "Clippers":      ("Clippers", "LA Clippers", "LAC Clippers",
                      "Los Angeles Clippers", "LAC",
                      # Kalshi truncates "Los Angeles Clippers" → "Los Angeles C"
                      "Los Angeles C"),
    "Lakers":        ("Lakers", "LA Lakers", "LAL Lakers",
                      "Los Angeles Lakers", "LAL",
                      # Kalshi truncates "Los Angeles Lakers" → "Los Angeles L"
                      "Los Angeles L"),
    "Grizzlies":     ("Grizzlies", "MEM Grizzlies", "Memphis Grizzlies",
                      "Memphis", "MEM"),
    "Heat":          ("Heat", "MIA Heat", "Miami Heat", "Miami", "MIA"),
    "Bucks":         ("Bucks", "MIL Bucks", "Milwaukee Bucks", "Milwaukee", "MIL"),
    "Timberwolves":  ("Timberwolves", "MIN Timberwolves", "Minnesota Timberwolves",
                      "Minnesota", "MIN", "Wolves"),
    "Pelicans":      ("Pelicans", "NO Pelicans", "NOP Pelicans",
                      "New Orleans Pelicans", "New Orleans", "NOP", "NO"),
    "Knicks":        ("Knicks", "NY Knicks", "NYK Knicks", "New York Knicks",
                      "New York", "NYK", "NY"),
    "Thunder":       ("Thunder", "OKC Thunder", "Oklahoma City Thunder",
                      "Oklahoma City", "OKC"),
    "Magic":         ("Magic", "ORL Magic", "Orlando Magic", "Orlando", "ORL"),
    "76ers":         ("76ers", "Sixers", "PHI 76ers", "PHI Sixers",
                      "Philadelphia 76ers", "Philadelphia", "PHI"),
    "Suns":          ("Suns", "PHX Suns", "PHO Suns", "Phoenix Suns", "Phoenix", "PHX", "PHO"),
    "Trail Blazers": ("Trail Blazers", "POR Trail Blazers", "Portland Trail Blazers",
                      "Portland", "POR", "Blazers"),
    "Kings":         ("Kings", "SAC Kings", "Sacramento Kings", "Sacramento", "SAC"),
    "Spurs":         ("Spurs", "SA Spurs", "SAS Spurs", "San Antonio Spurs",
                      "San Antonio", "SAS", "SA"),
    "Raptors":       ("Raptors", "TOR Raptors", "Toronto Raptors", "Toronto", "TOR"),
    "Jazz":          ("Jazz", "UTA Jazz", "Utah Jazz", "Utah", "UTA"),
    "Wizards":       ("Wizards", "WAS Wizards", "Washington Wizards", "Washington",
                      "WAS", "WSH"),
}


def _key(value: str) -> str:
    """Lowercase + strip non-alphanumerics for tolerant matching."""
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in _NBA_TEAMS.items():
    for alias in aliases:
        _ALIAS_TO_CANONICAL[_key(alias)] = canonical
    _ALIAS_TO_CANONICAL[_key(canonical)] = canonical


# Pattern that matches any team name we know about, longest-first so multi-
# word names ("Trail Blazers", "Los Angeles Lakers") win over the shorter
# fragments inside them.  Word-boundary anchored so it doesn't match inside
# unrelated text.
_ALL_NAMES = sorted(
    {alias for aliases in _NBA_TEAMS.values() for alias in aliases}
    | set(_NBA_TEAMS.keys()),
    key=len,
    reverse=True,
)
TEAM_NAME_PATTERN = (
    r"(?:" + "|".join(re.escape(name) for name in _ALL_NAMES) + r")"
)


def team_code_to_canonical(code: str) -> str | None:
    """Map a 2-3 letter team code (as found in game_logs.matchup) to its
    canonical short name (as stored in web_team_lines.away_team / home_team).
    """
    if not code:
        return None
    cleaned = re.sub(r"[^A-Za-z]", "", str(code)).upper()
    if not cleaned:
        return None
    return _ALIAS_TO_CANONICAL.get(_key(cleaned))


def normalize_team(value: str) -> str | None:
    """Return the canonical short name for any known team alias.

    Returns None when ``value`` doesn't match a known NBA team.
    """
    if not value:
        return None
    direct = _ALIAS_TO_CANONICAL.get(_key(value))
    if direct:
        return direct
    # Try last token alone — handles "Atlanta Hawks" → "Hawks" via fallback
    # if the full string didn't match (e.g. extra whitespace / punctuation).
    parts = re.findall(r"[A-Za-z0-9]+", str(value))
    if parts:
        last = _ALIAS_TO_CANONICAL.get(_key(parts[-1]))
        if last:
            return last
    return None


__all__ = ["normalize_team", "team_code_to_canonical", "TEAM_NAME_PATTERN"]
