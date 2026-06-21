"""Input validation for user-supplied numeric values + ingested data.

Covers two surfaces:
  - **Numeric** (line, odds, n_games, n_sims, parlay legs) used when fitting
    distributions or pricing parlays.  Bad numbers here would corrupt the
    model output or DoS the request.
  - **Categorical** (stat_type, team_code, season, rolling window) used by
    the chart UIs.  Bad categoricals here mostly produce empty charts or
    SQL noise — but validating them lets the UI show a friendly message
    instead of stack-tracing.

Why this exists
---------------
Even though our "AI" is statistical (scipy distributions + Monte Carlo + rolling
means, no learned weights), several adversarial-ML threat-model items still
translate to concrete defenses we should implement:

- **Adversarial input manipulation**: a user could send `nan`, `inf`, or
  absurdly out-of-range values to make `fitted_prob_over`, `expected_value`,
  or the Monte Carlo simulation produce garbage (or raise OverflowError).
- **Data poisoning**: scraped sportsbook content is third-party-controlled.
  A compromised page could feed us a "Jokic 9999.5 points" line that then
  pollutes the consensus mean, hit-rate, and EV computations.
- **Compute DoS**: large `n_sims` or `n_games` could make a free-tier user's
  request dominate a single worker.

Every defensive helper here is a tiny pure function so it composes cleanly
with the pre-existing tier gating in `auth.py`.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Optional

# Per-stat plausible ranges. Anything outside is almost certainly an attack
# or a scraper bug. Numbers are deliberately generous - real outliers should
# pass (Wilt's 100 doesn't, but no one's pricing 100-point lines anyway).
STAT_LINE_RANGES: dict[str, tuple[float, float]] = {
    "points":              (0.0, 100.0),
    "assists":             (0.0,  35.0),
    "rebounds":            (0.0,  40.0),
    "pra":                 (0.0, 150.0),
    "ra":                  (0.0,  60.0),
    "three_pointers_made": (0.0,  20.0),
    "field_goals_made":    (0.0,  35.0),
    "minutes":             (0.0,  60.0),
}

# American-odds boundaries that catch typos but allow real "longshot" lines.
ODDS_MIN = -100_000
ODDS_MAX =  100_000

# Cap on how much history a single request can pull. Premium users still hit
# this; the point is to bound per-request compute regardless of tier.
N_GAMES_HARD_CAP = 200

# Max parlay legs the model will simulate. Prevents combinatorial blow-up.
PARLAY_LEG_HARD_CAP = 6

# Rolling-window cap. Larger than ~50 makes the rolling line look like a
# global mean and is rarely useful for chart visualization.
ROLLING_WINDOW_HARD_CAP = 60

# Recognized chart stat types — UI dropdowns + URL params funnel through here.
KNOWN_STAT_TYPES: frozenset[str] = frozenset({
    "points", "assists", "rebounds", "pra", "ra",
    "three_pointers_made", "field_goals_made", "minutes",
    "steals", "blocks", "turnovers",
})

# Common UI aliases collapsed to canonical stat keys.
STAT_TYPE_ALIASES: dict[str, str] = {
    "3pm": "three_pointers_made",
    "3-pm": "three_pointers_made",
    "threes": "three_pointers_made",
    "three pointers": "three_pointers_made",
    "three_pointers": "three_pointers_made",
    "fgm": "field_goals_made",
    "field_goals": "field_goals_made",
    "fg": "field_goals_made",
    "min": "minutes",
}

# 30 NBA team codes (matches the ``team_names`` registry).
KNOWN_TEAM_CODES: frozenset[str] = frozenset({
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
})

# NBA season strings: "YYYY-YY" where the second segment is two digits.
_SEASON_RE = re.compile(r"^(?P<start>\d{4})-(?P<end>\d{2})$")


class ValidationError(ValueError):
    """Raised by any of the helpers below; callers catch as ValueError too."""


def _is_finite_number(x: Any) -> bool:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def validate_line(stat_type: str, line: Any) -> float:
    """Coerce + range-check a hypothetical line value.

    SECURITY: rejects NaN, inf, unknown stat types, and values outside the
    per-stat plausible range. The range check is the data-poisoning defense -
    even if a scraper fed us a 9999-point line, this rejects it before it
    ever feeds into consensus / EV math.

    Unknown stat types are rejected (fail-closed) rather than mapped to a
    generic default range, because a bet on a stat the model doesn't
    recognize is ALWAYS a bug or attack.
    """
    if not _is_finite_number(line):
        raise ValidationError(f"line must be a finite number; got {line!r}")
    val = float(line)
    stat = (stat_type or "").strip().lower()
    if stat not in STAT_LINE_RANGES:
        raise ValidationError(f"unknown stat_type {stat!r}")
    lo, hi = STAT_LINE_RANGES[stat]
    if not (lo <= val <= hi):
        raise ValidationError(
            f"line {val} for stat {stat!r} out of range [{lo}, {hi}]"
        )
    return val


def validate_american_odds(odds: Any) -> Optional[int]:
    """Coerce + range-check American odds. Returns None for "no odds given"."""
    if odds is None:
        return None
    if isinstance(odds, str):
        text = odds.strip().lower()
        if text in {"", "none", "na", "n/a", "-"}:
            return None
        if text == "even":
            return 100
    if not _is_finite_number(odds):
        raise ValidationError(f"odds must be a finite number; got {odds!r}")
    val = int(round(float(odds)))
    if val == 0:
        raise ValidationError("American odds cannot be 0")
    if not (ODDS_MIN <= val <= ODDS_MAX):
        raise ValidationError(
            f"American odds {val} out of range [{ODDS_MIN}, {ODDS_MAX}]"
        )
    return val


def validate_n_games(n: Any, *, min_value: int = 1) -> int:
    """Coerce + cap a `last N games` request."""
    if not _is_finite_number(n):
        raise ValidationError(f"n_games must be a finite integer; got {n!r}")
    val = int(round(float(n)))
    if val < min_value:
        raise ValidationError(f"n_games must be >= {min_value}; got {val}")
    return min(val, N_GAMES_HARD_CAP)


def validate_n_sims(n: Any, *, default: int = 20_000,
                    hard_cap: int = 200_000) -> int:
    """Coerce + cap a Monte Carlo `n_sims` request."""
    if n is None:
        return default
    if not _is_finite_number(n):
        raise ValidationError(f"n_sims must be a finite integer; got {n!r}")
    val = int(round(float(n)))
    if val < 1_000:
        return 1_000
    return min(val, hard_cap)


def validate_parlay_legs_count(n: int) -> int:
    """The model's correlated-Gaussian SGP gets pricey past ~6 legs."""
    if n < 2:
        raise ValidationError("parlay needs at least 2 legs")
    if n > PARLAY_LEG_HARD_CAP:
        raise ValidationError(
            f"parlay capped at {PARLAY_LEG_HARD_CAP} legs; got {n}"
        )
    return n


# -- Ingestion-side bounds ----------------------------------------------------
# Used by the daily ETL + web_prop_cards parser to drop scraper-poisoned rows
# before they ever reach the betting_lines table.

def _known_stats_for_sport(sport: Optional[str]) -> set:
    """Canonical stat-type set for a sport. NBA uses the hardcoded
    ``KNOWN_STAT_TYPES``; other sports read the ``sports/`` registry so we
    don't hardcode NFL/MLB/etc. stat lists here."""
    if not sport or str(sport).strip().lower() == "nba":
        return set(KNOWN_STAT_TYPES)
    from sports import get_sport
    return {str(s).strip().lower() for s in get_sport(sport).stat_types}


def validate_stat_type(
    stat_type: Any,
    *,
    allowed: Optional[Iterable[str]] = None,
    sport: Optional[str] = None,
) -> str:
    """Coerce + validate a chart stat-type input.

    Resolves common aliases ("3pm" → "three_pointers_made") and rejects
    anything not known for the sport.  For NBA (default) the allow-set is
    ``KNOWN_STAT_TYPES``; for other sports it comes from the ``sports/``
    registry (``Sport.stat_types``).  When ``allowed`` is supplied
    (e.g. the free-tier preview list), the result must also be in that
    subset — used to enforce per-tier caps without leaking premium-only
    stats through URL params.
    """
    if stat_type is None:
        raise ValidationError("stat_type is required")
    text = str(stat_type).strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        raise ValidationError("stat_type is required")
    canonical = STAT_TYPE_ALIASES.get(text, text)
    known = _known_stats_for_sport(sport)
    if not known:
        raise ValidationError(f"no stat types registered for sport {sport!r}")
    if canonical not in known:
        raise ValidationError(
            f"unknown stat_type {stat_type!r}; "
            f"expected one of {sorted(known)}"
        )
    if allowed is not None:
        allowed_set = {str(a).strip().lower() for a in allowed}
        if canonical not in allowed_set:
            raise ValidationError(
                f"stat_type {canonical!r} not allowed for this view "
                f"(allowed: {sorted(allowed_set)})"
            )
    return canonical


def validate_team_code(code: Any) -> str:
    """Normalize + validate an NBA team code (e.g. "nyk" → "NYK")."""
    if code is None:
        raise ValidationError("team code is required")
    cleaned = re.sub(r"[^A-Za-z]", "", str(code)).upper()
    if not cleaned:
        raise ValidationError("team code is required")
    if cleaned not in KNOWN_TEAM_CODES:
        raise ValidationError(
            f"unknown team code {code!r}; "
            f"expected one of {sorted(KNOWN_TEAM_CODES)}"
        )
    return cleaned


def validate_season(value: Any) -> str:
    """Validate a season string in ``YYYY-YY`` form (NBA convention).

    Sanity-bounded to seasons between 1946 (BAA) and 2099, and the second
    segment must equal ``(start + 1) % 100`` so "2024-26" is rejected.
    """
    if value is None:
        raise ValidationError("season is required")
    text = str(value).strip()
    m = _SEASON_RE.match(text)
    if not m:
        raise ValidationError(
            f"season {value!r} not in YYYY-YY form (e.g. '2024-25')"
        )
    start = int(m.group("start"))
    end = int(m.group("end"))
    if not (1946 <= start <= 2099):
        raise ValidationError(f"season start {start} out of range 1946-2099")
    if end != (start + 1) % 100:
        raise ValidationError(
            f"season {text!r} second segment must be (start+1) mod 100"
        )
    return f"{start}-{end:02d}"


def validate_rolling_window(n: Any, *, default: int = 5) -> int:
    """Coerce + bound the chart's rolling-mean window."""
    if n is None or (isinstance(n, str) and not n.strip()):
        return int(default)
    if not _is_finite_number(n):
        raise ValidationError(f"rolling window must be a number; got {n!r}")
    val = int(round(float(n)))
    if val < 1:
        raise ValidationError(f"rolling window must be >= 1; got {val}")
    return min(val, ROLLING_WINDOW_HARD_CAP)


def is_plausible_betting_line(stat_type: str, line: Any,
                              over_odds: Any = None,
                              under_odds: Any = None) -> bool:
    """Cheap predicate for ingest-time triage.

    Returns False when the row is structurally implausible. Callers should
    drop the row + log the rejection. Used by:
      - `nba_model.data.daily_etl` when normalizing odds-API payloads.
      - `nba_model.model.browser_prop_parser` after extracting board cards.
    """
    try:
        validate_line(stat_type, line)
    except ValidationError:
        return False
    for o in (over_odds, under_odds):
        if o is None:
            continue
        try:
            validate_american_odds(o)
        except ValidationError:
            return False
    return True
