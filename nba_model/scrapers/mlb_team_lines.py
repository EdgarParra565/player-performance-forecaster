"""Shared MLB team-line (game market) extraction for sportsbook lobbies.

DraftKings / FanDuel MLB lobbies render each game as run-line + total +
moneyline. The visible text interleaves a lot of noise (pitchers, live
inning/count markers, "same game parlay available"), but the odds block is a
fixed, anchorable sequence:

    <away_RL> <away_RL_odds> <away_ML>
    O <total> <over_odds>
    <home_RL> <home_RL_odds> <home_ML>
    U <total> <under_odds>

We anchor on that block, then pair it with the two nearest MLB team names that
precede it (away first, home second). Captured live from FanDuel MLB on
2026-06-28.
"""

from __future__ import annotations

import re

from nba_model.scrapers.mlb_team_names import MLB_TEAM_NAME_PATTERN, normalize_mlb_team

_ODDS = r"[+\-]\d{2,4}"
_RL = r"[+\-]\d+(?:\.\d+)?"   # run line (usually ±1.5, alt lines larger)
_TOT = r"\d{1,2}(?:\.\d+)?"

# The fixed per-game odds block (run line / total / moneyline).
_BLOCK_RE = re.compile(
    rf"(?P<away_rl>{_RL})\s+(?P<away_rl_odds>{_ODDS})\s+(?P<away_ml>{_ODDS})\s+"
    rf"O\s+(?P<total>{_TOT})\s+(?P<over_odds>{_ODDS})\s+"
    rf"(?P<home_rl>{_RL})\s+(?P<home_rl_odds>{_ODDS})\s+(?P<home_ml>{_ODDS})\s+"
    rf"U\s+(?P<total2>{_TOT})\s+(?P<under_odds>{_ODDS})"
)
_TEAM_RE = re.compile(MLB_TEAM_NAME_PATTERN)

# Max distance (chars) a paired team name may sit before its odds block, so we
# don't pair an odds block with a team name from an unrelated earlier game.
_MAX_TEAM_LOOKBACK = 600


def _normalize_minus_signs(text: str) -> str:
    return text.replace("−", "-").replace("–", "-")


def extract_team_lines(text: str) -> list[dict]:
    """Return one record per (game, market, side) found in MLB lobby text."""
    text = _normalize_minus_signs(text or "")
    # All team-name occurrences with positions (canonical, start).
    team_hits = [
        (m.start(), normalize_mlb_team(m.group(0))) for m in _TEAM_RE.finditer(text)
    ]
    team_hits = [(pos, name) for pos, name in team_hits if name]

    out: list[dict] = []
    seen: set[tuple] = set()
    for blk in _BLOCK_RE.finditer(text):
        preceding = [t for t in team_hits if t[0] < blk.start()
                     and blk.start() - t[0] <= _MAX_TEAM_LOOKBACK]
        if len(preceding) < 2:
            continue
        away = preceding[-2][1]
        home = preceding[-1][1]
        if not away or not home or away == home:
            continue
        game_key = (away, home, blk.start())
        if game_key in seen:
            continue
        seen.add(game_key)

        raw = blk.group(0)[:300]
        common = {"away_team": away, "home_team": home, "raw_text": raw}
        total = float(blk.group("total"))
        rows = [
            {**common, "market_type": "spread", "side": "away", "team": away,
             "line_value": float(blk.group("away_rl")),
             "odds_american": int(blk.group("away_rl_odds"))},
            {**common, "market_type": "spread", "side": "home", "team": home,
             "line_value": float(blk.group("home_rl")),
             "odds_american": int(blk.group("home_rl_odds"))},
            {**common, "market_type": "total", "side": "over", "team": None,
             "line_value": total, "odds_american": int(blk.group("over_odds"))},
            {**common, "market_type": "total", "side": "under", "team": None,
             "line_value": total, "odds_american": int(blk.group("under_odds"))},
            {**common, "market_type": "moneyline", "side": "away", "team": away,
             "line_value": None, "odds_american": int(blk.group("away_ml"))},
            {**common, "market_type": "moneyline", "side": "home", "team": home,
             "line_value": None, "odds_american": int(blk.group("home_ml"))},
        ]
        out.extend(rows)
    return out
