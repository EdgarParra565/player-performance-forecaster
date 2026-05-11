"""Reverse-engineer team-level priors from cross-book spread + total + moneyline.

Given the consensus (away_spread, home_spread, total, ml_away, ml_home) for
a single game, we can derive:
  - **implied team totals** for each side: (total ± spread) / 2
  - **win probabilities** for each side, vig-removed via the standard
    "two-sided proportional de-vig" (each side's no-vig prob is its raw
    implied / (raw_home + raw_away))
  - **pace proxy**: total / 200 (NBA average possession-per-team ~100, so
    a total of 220 implies ~10% above-average pace; a total of 195 ~2.5%
    below). This is a coarse estimate but stable across books.

These are useful as priors for the player-level model: e.g. if the implied
team total is 5pts above what the player's last-15 average team total
would suggest, the player's points projection should drift up proportionally.

Outputs land in ``team_priors`` (new view-only table) so the chart layer
and player-projection code can read them without re-running the math.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Optional

from nba_model.data.database.db_manager import (
    DatabaseManager,
    _american_to_implied_prob,
)

logger = logging.getLogger(__name__)

# Coarse pace proxy: NBA average game total trends ~228 across the league,
# split evenly across 2 teams ≈ ~100 possessions/team.  Tune later from
# `pace_factor` per-game once we ingest play-by-play.
LEAGUE_PACE_BASELINE_TOTAL = 228.0


def _devig_two_way(raw_home: Optional[float], raw_away: Optional[float]):
    """Two-way proportional de-vig.

    Each side's no-vig probability is its raw implied probability divided
    by the sum of the two raw probabilities.  Returns ``(p_home, p_away)``
    or ``(None, None)`` when inputs are missing.
    """
    if raw_home is None or raw_away is None:
        return (None, None)
    s = raw_home + raw_away
    if s <= 0:
        return (None, None)
    return (raw_home / s, raw_away / s)


def derive_team_priors_from_consensus(
    since_hours: float = 48.0,
    min_books: int = 2,
    db_path: str = "data/database/nba_data.db",
) -> dict:
    """Compute team priors for every game that has recent multi-book consensus.

    For each unique ``(away_team, home_team)`` in the lookback window:
      - mean game total, mean home spread, devigged win probabilities
      - implied team total per side: ``(total - team_spread) / 2``
      - pace proxy: ``total / LEAGUE_PACE_BASELINE_TOTAL``

    Returns a summary dict; the priors themselves are upserted into
    ``team_priors``.
    """
    with DatabaseManager(db_path=db_path) as db:
        team_rows = db.get_consensus_team_lines(
            since_hours=since_hours, min_books=min_books,
        )

    # Group by game.
    games: dict[tuple, dict] = {}
    for r in team_rows:
        key = (r["away_team"], r["home_team"])
        slot = games.setdefault(key, {
            "away_team": r["away_team"],
            "home_team": r["home_team"],
            "total": None,
            "home_spread": None,
            "away_spread": None,
            "ml_home_prob": None,
            "ml_away_prob": None,
            "n_books": r["n_books"],
            "latest_observed_at": r["latest_observed_at"],
        })
        market = r["market_type"]
        side = r["side"]
        if market == "total" and side == "over":
            slot["total"] = r["mean_line"]
        elif market == "spread" and side == "home":
            slot["home_spread"] = r["mean_line"]
        elif market == "spread" and side == "away":
            slot["away_spread"] = r["mean_line"]
        elif market == "moneyline" and side == "home":
            slot["ml_home_prob"] = r.get("mean_implied_prob")
        elif market == "moneyline" and side == "away":
            slot["ml_away_prob"] = r.get("mean_implied_prob")

    priors_payload = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for slot in games.values():
        total = slot["total"]
        home_spread = slot["home_spread"]
        away_spread = slot["away_spread"]
        if total is None or home_spread is None:
            continue
        # Implied team totals.  Note the sign convention:
        #   home_team_total = (game_total - home_spread) / 2
        # (home_spread of -1.5 for the favorite → home total ~ (total + 1.5)/2)
        home_total = (total - home_spread) / 2.0
        away_total = (total - (away_spread if away_spread is not None
                               else -home_spread)) / 2.0
        # Vig-removed win probs.
        p_home, p_away = _devig_two_way(slot["ml_home_prob"], slot["ml_away_prob"])
        # Pace proxy.
        pace_factor = total / LEAGUE_PACE_BASELINE_TOTAL
        priors_payload.append({
            "away_team": slot["away_team"],
            "home_team": slot["home_team"],
            "computed_at_utc": now_iso,
            "consensus_total": float(total),
            "home_spread": float(home_spread),
            "away_spread": float(away_spread) if away_spread is not None else None,
            "home_team_total": float(home_total),
            "away_team_total": float(away_total),
            "home_win_prob_devig": float(p_home) if p_home is not None else None,
            "away_win_prob_devig": float(p_away) if p_away is not None else None,
            "pace_factor": float(pace_factor),
            "n_books": int(slot["n_books"]),
            "latest_observed_at": slot["latest_observed_at"],
        })

    if priors_payload:
        with DatabaseManager(db_path=db_path) as db:
            res = db.upsert_team_priors(priors_payload)
    else:
        res = {"upserted": 0, "attempted": 0}

    return {
        "games_seen": len(games),
        "games_with_full_priors": len(priors_payload),
        "db_upserted": res["upserted"],
        "since_hours": since_hours,
        "min_books": min_books,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Derive team-level priors (implied totals, devig win prob, "
                    "pace factor) from cross-book consensus.",
    )
    parser.add_argument("--db-path", default="data/database/nba_data.db")
    parser.add_argument("--since-hours", type=float, default=48.0)
    parser.add_argument("--min-books", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    summary = derive_team_priors_from_consensus(
        since_hours=args.since_hours,
        min_books=args.min_books,
        db_path=args.db_path,
    )
    print("Team priors summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
