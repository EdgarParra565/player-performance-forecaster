"""Fliff scraper config + iframe-reaching JS extractor.

Fliff is a sweepstakes/social-pickem book whose prop UI is PrizePicks-like
(More/Less). Its board at ``sports.getfliff.com`` renders inside an **emulator
iframe** (``sports.getfliff.com/sports?channelId=...&emulator=t``): the parent
page's visible text is only marketing ("The #1 Social Sportsbook", promo
banners), while the real sports nav + picks live in the child frame. A plain
visible-text fetch therefore captures ~190 chars of marketing and misses the
board entirely, so this module ships a ``js_extractor`` that reaches into that
child frame and returns its text.

Parser status (2026-07-21): the fetch path is now iframe-aware, but the NBA
board is empty in the offseason (the iframe nav shows "NBA" with no event
count while MLB carries a slate), so there are no More/Less rows to pin a
``prop_preprocess`` against yet. Add the PrizePicks-style preprocessor
(reuse ``base.build_pp_style_name_pattern``; see ``prizepicks.py`` /
``parlayplay.py``) once an in-season NBA capture with picks is on disk.
"""

from __future__ import annotations

from nba_model.scrapers.base import BookScraper, SessionMarkers

# Substring identifying Fliff's board iframe (vs the marketing parent frame).
_BOARD_FRAME_MARKER = "getfliff.com/sports"


def extract_iframe_text(page) -> str:
    """Return the visible text of Fliff's emulator board iframe.

    The parent frame shows only marketing; the board is a nested frame whose
    URL contains ``getfliff.com/sports``. Best-effort: returns "" when the
    frame isn't present (e.g. the parent hasn't finished loading it), so the
    generic extraction result stands unchanged.
    """
    try:
        frames = list(getattr(page, "frames", []) or [])
    except Exception:
        return ""
    for frame in frames:
        url = str(getattr(frame, "url", "") or "")
        if _BOARD_FRAME_MARKER not in url:
            continue
        try:
            text = frame.locator("body").inner_text(timeout=3000)
        except Exception:
            continue
        if text and text.strip():
            return " ".join(str(text).split())
    return ""


SCRAPER = BookScraper(
    name="fliff",
    # Rebranded web board lives at sports.getfliff.com; fliff.com kept as alias.
    domain="getfliff.com",
    aliases=("fliff.com",),
    wait_selectors=(
        "[class*='player-prop']",
        "[class*='pick']",
        "[class*='prop']",
        "[data-testid*='prop']",
        "[data-testid*='pick']",
        "article",
    ),
    extra_wait_seconds=5.0,
    js_extractor=extract_iframe_text,
    session_markers=SessionMarkers(
        login_wall=(
            "log in",
            "sign up",
            "create account",
            "verify your phone",
        ),
        authenticated=(
            "more",
            "less",
            "nba",
            "points",
            "rebounds",
            "assists",
        ),
        min_authenticated_hits=4,
    ),
)
