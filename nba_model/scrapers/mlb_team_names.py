"""MLB team-name normalization (analog to ``team_names.py`` for the NBA).

Books render MLB teams as full names ("New York Yankees"), nicknames
("Yankees") or abbreviations ("NYY"); cross-book consensus needs one canonical
short name to GROUP BY on. We deliberately do NOT register bare ambiguous
cities ("New York" → Yankees or Mets?, "Los Angeles", "Chicago"); the full
name / nickname is always unambiguous.
"""

from __future__ import annotations

import re

# Canonical nickname -> every form seen on book pages (full name, nickname,
# abbrev, and city ONLY when unambiguous).
_MLB_TEAMS: dict[str, tuple[str, ...]] = {
    "Diamondbacks": ("Arizona Diamondbacks", "Diamondbacks", "D-backs", "ARI", "Arizona"),
    "Braves":       ("Atlanta Braves", "Braves", "ATL", "Atlanta"),
    "Orioles":      ("Baltimore Orioles", "Orioles", "BAL", "Baltimore"),
    "Red Sox":      ("Boston Red Sox", "Red Sox", "BOS", "Boston"),
    "Cubs":         ("Chicago Cubs", "Cubs", "CHC"),
    "White Sox":    ("Chicago White Sox", "White Sox", "CWS", "CHW"),
    "Reds":         ("Cincinnati Reds", "Reds", "CIN", "Cincinnati"),
    "Guardians":    ("Cleveland Guardians", "Guardians", "CLE", "Cleveland"),
    "Rockies":      ("Colorado Rockies", "Rockies", "COL", "Colorado"),
    "Tigers":       ("Detroit Tigers", "Tigers", "DET", "Detroit"),
    "Astros":       ("Houston Astros", "Astros", "HOU", "Houston"),
    "Royals":       ("Kansas City Royals", "Royals", "KC", "Kansas City"),
    "Angels":       ("Los Angeles Angels", "Angels", "LAA"),
    "Dodgers":      ("Los Angeles Dodgers", "Dodgers", "LAD"),
    "Marlins":      ("Miami Marlins", "Marlins", "MIA", "Miami"),
    "Brewers":      ("Milwaukee Brewers", "Brewers", "MIL", "Milwaukee"),
    "Twins":        ("Minnesota Twins", "Twins", "MIN", "Minnesota"),
    "Mets":         ("New York Mets", "Mets", "NYM"),
    "Yankees":      ("New York Yankees", "Yankees", "NYY"),
    "Athletics":    ("Athletics", "Oakland Athletics", "A's", "OAK", "Oakland"),
    "Phillies":     ("Philadelphia Phillies", "Phillies", "PHI", "Philadelphia"),
    "Pirates":      ("Pittsburgh Pirates", "Pirates", "PIT", "Pittsburgh"),
    "Padres":       ("San Diego Padres", "Padres", "SD", "San Diego"),
    "Giants":       ("San Francisco Giants", "Giants", "SF", "San Francisco"),
    "Mariners":     ("Seattle Mariners", "Mariners", "SEA", "Seattle"),
    "Cardinals":    ("St. Louis Cardinals", "St Louis Cardinals", "Cardinals", "STL"),
    "Rays":         ("Tampa Bay Rays", "Rays", "TB", "Tampa Bay"),
    "Rangers":      ("Texas Rangers", "Rangers", "TEX", "Texas"),
    "Blue Jays":    ("Toronto Blue Jays", "Blue Jays", "TOR", "Toronto"),
    "Nationals":    ("Washington Nationals", "Nationals", "WSH", "WAS", "Washington"),
}


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canon, _aliases in _MLB_TEAMS.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_key(_alias)] = _canon
    _ALIAS_TO_CANONICAL[_key(_canon)] = _canon

# Longest names first so "New York Yankees" beats "Yankees", and multi-word
# nicknames ("Red Sox", "Blue Jays", "White Sox") aren't split.
_ALL_NAMES = sorted(
    {a for aliases in _MLB_TEAMS.values() for a in aliases} | set(_MLB_TEAMS.keys()),
    key=len,
    reverse=True,
)
MLB_TEAM_NAME_PATTERN = "(?:" + "|".join(re.escape(n) for n in _ALL_NAMES) + ")"


def normalize_mlb_team(value: str) -> str | None:
    """Return the canonical nickname for any known MLB team alias, else None."""
    if not value:
        return None
    direct = _ALIAS_TO_CANONICAL.get(_key(value))
    if direct:
        return direct
    parts = re.findall(r"[A-Za-z0-9]+", str(value))
    if parts:
        last = _ALIAS_TO_CANONICAL.get(_key(parts[-1]))
        if last:
            return last
    return None


__all__ = ["normalize_mlb_team", "MLB_TEAM_NAME_PATTERN"]
