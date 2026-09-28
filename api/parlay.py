"""Correlation-aware parlay pricing for the flagship /parlay view.

Compute-only (nothing persisted). Wraps the SAME model helpers the Streamlit
parlay path uses — no model math is reimplemented here:

- per-leg P(over): ``player_charts.fetch_player_chart_data`` +
  ``player_charts.fitted_prob_over`` (the fitted normal Player Detail shows)
- correlation: ``correlation_calibration.calibrate_correlations`` (PSD-safe,
  shrunk toward identity, falls back to identity under ``min_games`` shared
  games) over the legs' stat series aligned on game date
- covariance: ``correlation_calibration.covariance_matrix``
- joint P(all legs hit): ``parlay_simulation.simulate_multi_leg_sgp``
  (correlated multivariate normal Monte Carlo)
- EV: ``parlay_ev.calculate_parlay_ev``; odds via ``model.odds``

UNDER legs are handled by the standard sign flip: for Y = -X the event
``X < line`` is ``Y > -line`` and the correlation between two legs flips sign
when exactly one of them is an under. That is an input transform, not new
math.

The number is a MODEL ESTIMATE — WS10 validation gates are unpassed; it is not
validated for real money.
"""
from __future__ import annotations

import hashlib
import math
from threading import Lock
from typing import Optional

import numpy as np
import pandas as pd

from nba_model.model import correlation_calibration as cc
from nba_model.model import parlay_ev
from nba_model.model import parlay_simulation as ps
from nba_model.model.odds import american_to_implied_prob
from nba_model.visualization import player_charts as pc

DEFAULT_LEG_ODDS = -110
DEFAULT_N_SIMS = 20_000

# simulate_multi_leg_sgp draws from numpy's global RNG. Seed it per request
# (deterministic in the legs) under a lock so the same parlay prices the same
# on every render and concurrent requests can't interleave draws.
_RNG_LOCK = Lock()


def _decimal(american: int) -> float:
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / abs(american))


def _american_from_decimal(dec: float) -> Optional[int]:
    if not math.isfinite(dec) or dec <= 1.0:
        return None
    return int(round((dec - 1.0) * 100)) if dec >= 2.0 else int(round(-100.0 / (dec - 1.0)))


def _fair_american(prob: Optional[float]) -> Optional[int]:
    if prob is None or prob <= 0.0 or prob >= 1.0:
        return None
    return _american_from_decimal(1.0 / prob)


def _seed_for(legs: list[dict], n_games: int, n_sims: int) -> int:
    key = "|".join(
        f"{leg['player_id']}:{leg['stat']}:{leg['line']}:{leg['side']}" for leg in legs
    ) + f"|{n_games}|{n_sims}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


class LegDataError(ValueError):
    """A leg has no usable history (surfaces as a 400)."""


def price_parlay(
    db_path: str,
    legs: list[dict],
    *,
    n_games: int = 25,
    n_sims: int = DEFAULT_N_SIMS,
    parlay_odds: Optional[int] = None,
) -> dict:
    """Price validated legs. Each leg: player_id, player_name, stat, line,
    side ('over'|'under'), odds (American)."""
    per_leg: list[dict] = []
    series: list[pd.Series] = []
    for i, leg in enumerate(legs):
        data = pc.fetch_player_chart_data(
            db_path, leg["player_id"], leg["player_name"], leg["stat"], n_games=n_games,
        )
        n = int(data.values.size)
        p_over = pc.fitted_prob_over(data, float(leg["line"]))
        if n < 3 or p_over is None:
            raise LegDataError(
                f"leg {i + 1} ({leg['player_name']} {leg['stat']}): "
                f"not enough game logs ({n}) to model this leg"
            )
        p_hit = p_over if leg["side"] == "over" else 1.0 - p_over
        odds = int(leg["odds"])
        per_leg.append({
            **leg,
            "n_games": n,
            "mu": float(data.mu),
            "sigma": float(data.sigma),
            "p_over": float(p_over),
            "p_hit": float(p_hit),
            "implied_prob": float(american_to_implied_prob(odds)),
            "ev": float(parlay_ev.calculate_parlay_ev(p_hit, odds)),
        })
        dates = data.games["game_date"].astype(str).str.slice(0, 10).to_numpy()
        s = pd.Series(np.asarray(data.values, dtype=float), index=dates)
        series.append(s[~s.index.duplicated(keep="last")])

    cols = [f"leg{i + 1}" for i in range(len(legs))]
    aligned = pd.concat(series, axis=1, join="inner")
    aligned.columns = cols
    corr = cc.calibrate_correlations(aligned, stats_cols=cols)
    corr_values = corr.loc[cols, cols].to_numpy(dtype=float)

    signs = np.array([1.0 if leg["side"] == "over" else -1.0 for leg in per_leg])
    signed_corr = pd.DataFrame(corr_values * np.outer(signs, signs), index=cols, columns=cols)
    stds = {c: max(leg["sigma"], 1e-6) for c, leg in zip(cols, per_leg)}
    cov = cc.covariance_matrix(signed_corr, stds)
    means = [s * leg["mu"] for s, leg in zip(signs, per_leg)]
    lines = [s * float(leg["line"]) for s, leg in zip(signs, per_leg)]

    with _RNG_LOCK:
        np.random.seed(_seed_for(legs, n_games, n_sims))
        joint = float(ps.simulate_multi_leg_sgp(means, cov, lines, n=n_sims))

    independent = float(np.prod([leg["p_hit"] for leg in per_leg]))
    combined_dec = float(np.prod([_decimal(int(leg["odds"])) for leg in per_leg]))
    combined_american = _american_from_decimal(combined_dec)
    offered = parlay_odds if parlay_odds is not None else combined_american
    se = math.sqrt(max(joint * (1.0 - joint), 0.0) / n_sims)

    return {
        "legs": per_leg,
        "n_games": int(n_games),
        "n_sims": int(n_sims),
        "n_joint_games": int(len(aligned)),
        "correlation_fallback": bool(len(aligned) < 8),
        "joint_prob": joint,
        "joint_prob_se": se,
        "independent_prob": independent,
        "correlation_lift": (joint / independent) if independent > 0 else None,
        "combined_decimal": combined_dec,
        "combined_american": combined_american,
        "offered_american": offered,
        "offered_is_custom": parlay_odds is not None,
        "implied_prob": float(american_to_implied_prob(offered)) if offered else None,
        "fair_american": _fair_american(joint),
        "ev_joint": float(parlay_ev.calculate_parlay_ev(joint, offered)) if offered else None,
        "ev_independent": (
            float(parlay_ev.calculate_parlay_ev(independent, offered)) if offered else None
        ),
        "correlation": {
            "labels": [f"{leg['player_name']} {leg['stat']}" for leg in per_leg],
            "matrix": [[round(float(v), 4) for v in row] for row in corr_values],
        },
    }
