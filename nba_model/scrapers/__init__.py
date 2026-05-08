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
    espnbet,
    fanatics,
    fanduel,
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
)

BY_DOMAIN: dict[str, BookScraper] = {s.domain: s for s in SCRAPERS}
BY_NAME: dict[str, BookScraper] = {s.name: s for s in SCRAPERS}


def get_scraper_for_url(url: str) -> Optional[BookScraper]:
    """Return the scraper whose domain (or alias) matches ``url``'s host.

    When multiple scrapers match (e.g. ``pick6.draftkings.com`` matches both
    ``pick6.draftkings.com`` and ``draftkings.com``), the one with the
    longest matching domain wins so subdomain-specific scrapers take
    precedence over their parents.
    """
    try:
        host = urlparse(str(url or "")).netloc.lower()
    except Exception:
        return None
    if host.startswith("www."):
        host = host[4:]
    host = host.split(":")[0]
    best: Optional[BookScraper] = None
    best_specificity = -1
    for scraper in SCRAPERS:
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


__all__ = [
    "BookScraper",
    "SessionMarkers",
    "SCRAPERS",
    "BY_DOMAIN",
    "BY_NAME",
    "get_scraper_for_url",
    "get_scraper_by_name",
]
