"""Cross-compare model parlay outputs against the local chart-data view.

Three methods we contrast on the web app:

1. **Model** (`nba_model.run_model.run_single_prop` / `run_parlay_demo`) -
   uses live NBA API + rolling-mean projection + defense + minutes adjustment +
   Monte Carlo over the chosen distribution; for parlays it simulates with a
   correlated Gaussian.

2. **Chart data** - takes the same per-player game_logs already used to build
   the Player Charts and computes a fitted-normal P(over) per leg from the
   observed mean/sigma. For multi-leg this assumes leg independence.

3. **Historical** - counts how many of the last N games actually went over
   each leg, and for parlays counts how often *all legs hit in the same game*.

The point of the comparison is to flag situations where the model and the raw
historical data strongly disagree, and where the book's posted line falls
inside or outside both probability estimates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from nba_model.visualization import player_charts as pc


@dataclass
class LegSpec:
    stat_type: str
    line: float


def _american_to_decimal(odds: int | float | None) -> Optional[float]:
    if odds is None:
        return None
    try:
        o = int(odds)
    except (TypeError, ValueError):
        return None
    if o == 0:
        return None
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / abs(o))


def _ev(prob: Optional[float], american_odds: int | float | None) -> Optional[float]:
    dec = _american_to_decimal(american_odds)
    if prob is None or dec is None:
        return None
    return float(prob * (dec - 1.0) - (1.0 - prob))


def chart_single_summary(
    db_path: str,
    player_id: int,
    player_name: str,
    stat_type: str,
    line: float,
    n_games: int,
    american_odds: int | None,
) -> dict:
    """Summarize the chart-data + book view of a single prop."""
    data = pc.fetch_player_chart_data(
        db_path=db_path, player_id=player_id, player_name=player_name,
        stat_type=stat_type, n_games=n_games,
    )
    p_norm = pc.fitted_prob_over(data, float(line))
    n = int(data.values.size)
    hit_rate = (
        float(np.sum(np.asarray(data.values) > float(line))) / n if n else None
    )
    consensus = pc.compute_market_consensus(data)
    return {
        "data": data,
        "n": n,
        "mu": float(data.mu) if n else None,
        "sigma": float(data.sigma) if n else None,
        "p_over_normal": p_norm,
        "ev_over_normal": _ev(p_norm, american_odds),
        "historical_over_rate": hit_rate,
        "ev_over_historical": _ev(hit_rate, american_odds),
        "book_mean": consensus["mean"],
        "book_sigma": consensus["stdev"],
        "books_used": consensus["books_used"],
        "books_missing": consensus["books_missing"],
        "per_book": consensus["per_book"],
    }


def historical_parlay_hits(
    db_path: str,
    player_id: int,
    legs: list[LegSpec],
    n_games: int,
) -> dict:
    """For each game in the player's last N, evaluate whether ALL legs hit.

    Returns:
        {
            "n":      total games examined (joint coverage),
            "all_hit": games where every leg's stat went over its line,
            "rate":   all_hit / n,
            "per_leg": [{stat, line, hit_rate, n}],
        }
    """
    if not legs:
        return {"n": 0, "all_hit": 0, "rate": None, "per_leg": []}

    series_per_leg: list[np.ndarray] = []
    per_leg_meta: list[dict] = []
    games_index: pd.Index | None = None

    for leg in legs:
        d = pc.fetch_player_chart_data(
            db_path=db_path, player_id=player_id, player_name="",
            stat_type=leg.stat_type, n_games=n_games,
        )
        # Use game_id (or game_date) as the join key so we only compare games
        # where every leg's stat exists. Most stats live on the same row, but
        # this is robust if the schema ever changes.
        if d.games.empty:
            return {"n": 0, "all_hit": 0, "rate": None,
                    "per_leg": per_leg_meta}
        games_df = d.games.reset_index(drop=True).copy()
        games_df["__leg_value__"] = d.values
        idx = pd.Index(games_df["game_date"].astype(str))
        if games_index is None:
            games_index = idx
        else:
            games_index = games_index.intersection(idx)
        series_per_leg.append(games_df.set_index(idx)["__leg_value__"])

    if games_index is None or len(games_index) == 0:
        return {"n": 0, "all_hit": 0, "rate": None,
                "per_leg": per_leg_meta}

    all_hit = np.ones(len(games_index), dtype=bool)
    for leg, vals in zip(legs, series_per_leg):
        common = vals.reindex(games_index).fillna(np.nan).to_numpy(dtype=float)
        leg_hit = common > float(leg.line)
        per_leg_meta.append({
            "stat": leg.stat_type,
            "line": float(leg.line),
            "n": int(np.isfinite(common).sum()),
            "hit_rate": float(np.nansum(leg_hit) / np.isfinite(common).sum())
                        if np.isfinite(common).any() else None,
        })
        all_hit &= leg_hit & np.isfinite(common)

    n = int(len(games_index))
    hits = int(all_hit.sum())
    return {
        "n": n,
        "all_hit": hits,
        "rate": (hits / n) if n else None,
        "per_leg": per_leg_meta,
    }


def chart_independence_parlay(
    db_path: str,
    player_id: int,
    player_name: str,
    legs: list[LegSpec],
    n_games: int,
) -> dict:
    """Per-leg fitted-normal P(over), plus the independence-product parlay P.

    The independence assumption is wrong (player stats are correlated) but
    it's a useful naive baseline. Compare against the model's correlated SGP
    and the historical all-hit rate to gauge how much the correlation
    structure matters.
    """
    per_leg = []
    product = 1.0
    valid = True
    for leg in legs:
        d = pc.fetch_player_chart_data(
            db_path=db_path, player_id=player_id, player_name=player_name,
            stat_type=leg.stat_type, n_games=n_games,
        )
        p = pc.fitted_prob_over(d, float(leg.line))
        per_leg.append({
            "stat": leg.stat_type,
            "line": float(leg.line),
            "mu": float(d.mu) if d.values.size else None,
            "sigma": float(d.sigma) if d.values.size else None,
            "p_over": p,
        })
        if p is None:
            valid = False
        else:
            product *= p
    return {
        "per_leg": per_leg,
        "product_p": product if valid else None,
    }
