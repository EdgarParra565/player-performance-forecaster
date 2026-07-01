"""Per-book scraper registry.

Each book lives in its own module and exports a ``BookScraper`` instance
named ``SCRAPER``.  This module aggregates them and exposes lookup helpers
keyed by either the host domain or the canonical book name.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

from nba_model.scrapers.base import BookScraper, SessionMarkers
from nba_model.scrapers import (
    betmgm,
    betrivers,
    bettingpros,
    bovada,
    caesars,
    dabble,
    draftkings,
    draftkings_mlb,
    draftkings_nfl,
    espnbet,
    fanatics,
    fanduel,
    fanduel_mlb,
    fanduel_nfl,
    fliff,
    hardrockbet,
    kalshi,
    oddsshark,
    parlayplay,
    pick6,
    prizepicks,
    sleeper,
    underdog,
    vegasinsider,
)


SCRAPERS: tuple[BookScraper, ...] = (
    # ---- DFS pickem (More/Less or Higher/Lower) ----
    prizepicks.SCRAPER,
    underdog.SCRAPER,
    fliff.SCRAPER,
    pick6.SCRAPER,           # subdomain — must come before draftkings for resolution
    sleeper.SCRAPER,
    dabble.SCRAPER,
    parlayplay.SCRAPER,
    # ---- Traditional sportsbooks (American odds) ----
    draftkings.SCRAPER,
    fanduel.SCRAPER,
    betmgm.SCRAPER,
    caesars.SCRAPER,
    betrivers.SCRAPER,
    fanatics.SCRAPER,
    espnbet.SCRAPER,
    hardrockbet.SCRAPER,
    bovada.SCRAPER,
    # ---- Prediction market ----
    kalshi.SCRAPER,
    # ---- Public aggregators (no auth) ----
    oddsshark.SCRAPER,
    vegasinsider.SCRAPER,
    bettingpros.SCRAPER,
    # ---- Multi-sport (NFL stub + live MLB; same domains as their NBA configs) ----
    fanduel_nfl.SCRAPER,
    draftkings_nfl.SCRAPER,
    fanduel_mlb.SCRAPER,
    draftkings_mlb.SCRAPER,
)

# BY_DOMAIN / BY_NAME are the NBA-default lookups: a book may now have more
# than one config (one per sport) sharing a domain/name, so prefer the live
# NBA entry (listed first) and never let a later non-NBA config clobber it.
# Use get_scraper_for_book_sport / get_scraper_for_url(..., sport=) for the
# multi-sport lookups.
BY_DOMAIN: dict[str, BookScraper] = {}
BY_NAME: dict[str, BookScraper] = {}
for _s in SCRAPERS:
    BY_DOMAIN.setdefault(_s.domain, _s)
    BY_NAME.setdefault(_s.name, _s)
del _s

# (book, sport) -> scraper. The canonical multi-sport lookup.
BY_NAME_SPORT: dict[tuple, BookScraper] = {(s.name, s.sport): s for s in SCRAPERS}


def get_scraper_for_url(url: str, sport: Optional[str] = None) -> Optional[BookScraper]:
    """Return the scraper whose domain (or alias) matches ``url``'s host.

    When multiple scrapers match (e.g. ``pick6.draftkings.com`` matches both
    ``pick6.draftkings.com`` and ``draftkings.com``), the one with the
    longest matching domain wins so subdomain-specific scrapers take
    precedence over their parents. When ``sport`` is given, only scrapers for
    that sport are considered (the multi-sport rollout: same domain can host
    a different per-sport config).
    """
    try:
        host = urlparse(str(url or "")).netloc.lower()
    except Exception:
        return None
    if host.startswith("www."):
        host = host[4:]
    host = host.split(":")[0]
    sport_key = str(sport).strip().lower() if sport else None
    best: Optional[BookScraper] = None
    best_specificity = -1
    for scraper in SCRAPERS:
        if sport_key is not None and scraper.sport != sport_key:
            continue
        for candidate in (scraper.domain,) + scraper.aliases:
            if host == candidate or host.endswith(f".{candidate}"):
                if len(candidate) > best_specificity:
                    best = scraper
                    best_specificity = len(candidate)
                break
    return best


def get_scraper_by_name(name: str) -> Optional[BookScraper]:
    """Return the scraper whose canonical name matches ``name``."""
    return BY_NAME.get(str(name or "").strip().lower())


def get_scraper_for_book_sport(book: str, sport: str = "nba") -> Optional[BookScraper]:
    """Resolve a scraper by ``(book, sport)`` — the multi-sport lookup."""
    return BY_NAME_SPORT.get(
        (str(book or "").strip().lower(), str(sport or "nba").strip().lower())
    )


__all__ = [
    "BookScraper",
    "SessionMarkers",
    "SCRAPERS",
    "BY_DOMAIN",
    "BY_NAME",
    "BY_NAME_SPORT",
    "get_scraper_for_url",
    "get_scraper_by_name",
    "get_scraper_for_book_sport",
]
