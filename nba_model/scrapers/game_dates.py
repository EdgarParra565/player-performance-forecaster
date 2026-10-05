"""Game-date capture for team-line extractors.

Book lobbies print each game's date next to its lines, in book-specific
forms: Caesars "OCT 20 12:10 PM" (after the game), Bovada "10/20/2026 12:00 PM"
(before), FanDuel "Oct 20, 3:00pm ET" / "Today 7:10pm". Extractors attach the
raw token as ``game_date_hint``; ``team_line_parser`` resolves it against the
snapshot's capture time with ``resolve_game_date`` so ``web_team_lines`` /
``team_priors`` can tell two games of the same team apart.

Relative tokens (Today / Tomorrow / weekday) resolve in the scraping host's
local timezone — that is the clock the book page rendered them in.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Optional

_MONTHS = {
    m: i for i, m in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"), start=1)
}
_WEEKDAYS = {d: i for i, d in enumerate(("mon", "tue", "wed", "thu", "fri", "sat", "sun"))}

DATE_TOKEN_PATTERN = (
    r"(?:\d{1,2}/\d{1,2}(?:/\d{2,4})?"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?!\d)"
    r"|today|tonight|tomorrow"
    r"|(?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)(?:day|nesday|rsday|urday)?\b)"
)
_TOKEN_RE = re.compile(DATE_TOKEN_PATTERN, re.IGNORECASE)

# Year inference for month/day tokens: a board lists games from a couple of
# days back (live/final) up to the season's far future (futures excluded).
_MIN_PAST_DAYS = 3
_MAX_FUTURE_DAYS = 240


def find_date_hint(text: str, start: int, end: int, *, after: int = 0, before: int = 0) -> Optional[str]:
    """First date token within ``after`` chars following ``end`` (or, when
    ``before`` > 0, the LAST token within ``before`` chars preceding
    ``start``). Returns the raw token or None."""
    if after:
        m = _TOKEN_RE.search(text, end, min(len(text), end + after))
        if m:
            return m.group(0)
    if before:
        hits = list(_TOKEN_RE.finditer(text, max(0, start - before), start))
        if hits:
            return hits[-1].group(0)
    return None


def _observed_local_date(observed_at_utc, tz: Optional[tzinfo]) -> date:
    ts = None
    if isinstance(observed_at_utc, datetime):
        ts = observed_at_utc
    elif observed_at_utc:
        try:
            ts = datetime.fromisoformat(str(observed_at_utc).replace("Z", "+00:00"))
        except ValueError:
            ts = None
    if ts is None:
        ts = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(tz).date()  # tz=None → host local time


def _infer_year(month: int, day: int, ref: date) -> Optional[date]:
    best = None
    for year in (ref.year - 1, ref.year, ref.year + 1):
        try:
            cand = date(year, month, day)
        except ValueError:
            continue
        delta = (cand - ref).days
        if -_MIN_PAST_DAYS <= delta <= _MAX_FUTURE_DAYS and (best is None or abs(delta) < abs((best - ref).days)):
            best = cand
    return best


def resolve_game_date(hint: Optional[str], observed_at_utc=None, tz: Optional[tzinfo] = None) -> Optional[str]:
    """ISO ``YYYY-MM-DD`` for a raw date token, or None when unparseable."""
    if not hint:
        return None
    token = str(hint).strip().lower().rstrip(",.")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
        return token
    ref = _observed_local_date(observed_at_utc, tz)
    if token in ("today", "tonight"):
        return ref.isoformat()
    if token == "tomorrow":
        return (ref + timedelta(days=1)).isoformat()
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", token)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        if m.group(3):
            year = int(m.group(3))
            year += 2000 if year < 100 else 0
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                return None
        got = _infer_year(month, day, ref)
        return got.isoformat() if got else None
    m = re.fullmatch(r"([a-z]{3})[a-z]*\.?\s+(\d{1,2})", token)
    if m and m.group(1) in _MONTHS:
        got = _infer_year(_MONTHS[m.group(1)], int(m.group(2)), ref)
        return got.isoformat() if got else None
    if token[:3] in _WEEKDAYS:
        ahead = (_WEEKDAYS[token[:3]] - ref.weekday()) % 7
        return (ref + timedelta(days=ahead)).isoformat()
    return None


__all__ = ["DATE_TOKEN_PATTERN", "find_date_hint", "resolve_game_date"]
