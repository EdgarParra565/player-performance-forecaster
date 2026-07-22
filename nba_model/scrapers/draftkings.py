"""DraftKings sportsbook scraper config + team-line extractor.

Format observed on the basketball lobby page:

    "NY Knicks at PHI 76ers +1.5 −108 O 213.5 −105 +102 -1.5 −112 U 213.5 −115 −122"

NB: DraftKings renders American-odds minus signs as Unicode U+2212 (−) not
ASCII -. We normalize them to ASCII before parsing.
"""

from __future__ import annotations

import re

from nba_model.scrapers.base import BookScraper, SessionMarkers
from nba_model.scrapers.team_names import TEAM_NAME_PATTERN, normalize_team


_ODDS = r"[+\-]\d{2,4}"
_SPREAD = r"[+\-]\d+(?:\.\d+)?"
_TOTAL = r"\d{2,3}(?:\.\d+)?"

# Team mention is "<abbrev> <Team Short>" (e.g. "NY Knicks", "PHI 76ers").
_TEAM = r"[A-Z]{2,4}\s+(?P<NAME>" + TEAM_NAME_PATTERN + r")"

# Each game block:
#   <away_abbrev> <Away> at <home_abbrev> <Home>
#   <away_spread> <away_spread_odds> O <total> <over_odds> <away_ml>
#   <home_spread> <home_spread_odds> U <total> <under_odds> <home_ml>
_GAME_RE = re.compile(
    _TEAM.replace("NAME", "away")
    + r"\s+at\s+"
    + _TEAM.replace("NAME", "home")
    + r"\s+(?P<away_spread>" + _SPREAD + r")"
      r"\s+(?P<away_spread_odds>" + _ODDS + r")"
      r"\s+O\s+(?P<total>" + _TOTAL + r")"
      r"\s+(?P<over_odds>" + _ODDS + r")"
      r"\s+(?P<away_ml>" + _ODDS + r")"
      r"\s+(?P<home_spread>" + _SPREAD + r")"
      r"\s+(?P<home_spread_odds>" + _ODDS + r")"
      r"\s+U\s+" + _TOTAL +
      r"\s+(?P<under_odds>" + _ODDS + r")"
      r"\s+(?P<home_ml>" + _ODDS + r")"
)


def extract_grid_text(page) -> str:
    """Return the lobby grid's rendered text via ``body.innerText``.

    DK renders its odds grid into virtualized nodes whose text the generic
    HTML-parse extraction (``page.content()`` → visible text) drops — it
    captures the nav/footer shell but not the game rows. ``body.innerText``
    reflects the live rendered DOM (after the fetcher's scroll pass) and does
    include the grid, so we supply it as a targeted extractor. Best-effort:
    returns "" on failure so the generic result stands.
    """
    try:
        text = page.evaluate("document.body ? document.body.innerText : ''")
    except Exception:
        return ""
    return " ".join(str(text or "").split())


def _normalize_minus_signs(text: str) -> str:
    """Map Unicode minus (U+2212) and en-dash (U+2013) to ASCII '-'."""
    return text.replace("−", "-").replace("–", "-")


def extract_team_lines(text: str) -> list[dict]:
    """Return one record per (game, market, side) found in DraftKings text."""
    text = _normalize_minus_signs(text)
    out: list[dict] = []
    for m in _GAME_RE.finditer(text):
        away = normalize_team(m.group("away"))
        home = normalize_team(m.group("home"))
        if not away or not home:
            continue
        raw = m.group(0)[:300]
        common = {"away_team": away, "home_team": home, "raw_text": raw}

        out.append({**common, "market_type": "spread", "side": "away",
                    "team": away,
                    "line_value": float(m.group("away_spread")),
                    "odds_american": int(m.group("away_spread_odds"))})
        out.append({**common, "market_type": "spread", "side": "home",
                    "team": home,
                    "line_value": float(m.group("home_spread")),
                    "odds_american": int(m.group("home_spread_odds"))})

        total = float(m.group("total"))
        out.append({**common, "market_type": "total", "side": "over",
                    "team": None, "line_value": total,
                    "odds_american": int(m.group("over_odds"))})
        out.append({**common, "market_type": "total", "side": "under",
                    "team": None, "line_value": total,
                    "odds_american": int(m.group("under_odds"))})

        out.append({**common, "market_type": "moneyline", "side": "away",
                    "team": away, "line_value": None,
                    "odds_american": int(m.group("away_ml"))})
        out.append({**common, "market_type": "moneyline", "side": "home",
                    "team": home, "line_value": None,
                    "odds_american": int(m.group("home_ml"))})
    return out


SCRAPER = BookScraper(
    name="draftkings",
    domain="draftkings.com",
    # DK migrated its lobby grid to `event-cell`-classed nodes (live-probed
    # 2026-07-22); the older `sportsbook-outcome`/`outcome-cell` selectors match
    # nothing now, so the fetcher skipped the extra content wait and captured
    # only the nav shell. `event-cell` is listed first so the selector wait
    # fires and the lazy grid renders before we scroll.
    wait_selectors=(
        "[class*='event-cell']",
        "[class*='sportsbook-outcome']",
        "[class*='player-prop']",
        "[class*='outcome-cell']",
        "[data-testid*='outcome']",
    ),
    extra_wait_seconds=4.0,
    js_extractor=extract_grid_text,
    session_markers=SessionMarkers(
        login_wall=(
            "sign up",
            "log in",
            "create account",
            "verify identity",
        ),
        authenticated=(
            "spread",
            "total",
            "moneyline",
            "nba",
            "today",
            "tomorrow",
        ),
        min_authenticated_hits=4,
    ),
    team_line_extractor=extract_team_lines,
    # DK renders its odds grid lazily as rows scroll into view — scroll to the
    # bottom after load so the snapshot captures the full slate, not just the
    # first viewport.
    scroll_page=True,
)
