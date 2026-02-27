import numpy as np
import pandas as pd


_STAT_COLUMN_ALIASES = {
    "PTS": "points",
    "AST": "assists",
    "REB": "rebounds",
    "PRA": "pra",
    "MIN": "minutes",
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize upstream data to the lowercase stat schema used across the project."""
    rename_map = {
        source: target
        for source, target in _STAT_COLUMN_ALIASES.items()
        if source in df.columns and target not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)

    for col in ("points", "assists", "rebounds", "minutes"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived combined market used by many books.
    if {"points", "assists", "rebounds"}.issubset(df.columns) and "pra" not in df.columns:
        df["pra"] = df["points"] + df["assists"] + df["rebounds"]
    if "pra" in df.columns:
        df["pra"] = pd.to_numeric(df["pra"], errors="coerce")

    if "points" not in df.columns or "minutes" not in df.columns:
        raise KeyError(
            "add_rolling_stats requires 'points' and 'minutes' columns "
            "(or their NBA API aliases 'PTS' and 'MIN')."
        )
    return df


def _infer_home_away_from_matchup(matchup_value) -> str:
    """Infer venue tag from matchup string when explicit home_away is missing."""
    text = str(matchup_value or "")
    if "vs." in text:
        return "home"
    if "@" in text:
        return "away"
    return "unknown"


def add_context_features(df: pd.DataFrame, injury_window: int = 5) -> pd.DataFrame:
    """
    Add contextual scheduling/travel/injury-proxy features.

    Generated features:
      - rest_days
      - is_back_to_back
      - travel_flag
      - games_last_7d
      - injury_proxy (minutes-drop z-score proxy)
    """
    if injury_window < 3:
        raise ValueError("injury_window must be >= 3")

    out = df.copy()
    if "home_away" not in out.columns and "matchup" in out.columns:
        out["home_away"] = out["matchup"].apply(_infer_home_away_from_matchup)

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
        out = out.sort_values("game_date").reset_index(drop=True)
        date_diffs = out["game_date"].diff().dt.days
        out["rest_days"] = date_diffs.fillna(3).clip(lower=0)
        out["is_back_to_back"] = (out["rest_days"] <= 1).astype(int)

        # Count games played in trailing 7 days (inclusive of current game).
        dates = out["game_date"]
        density = []
        for current_date in dates:
            if pd.isna(current_date):
                density.append(np.nan)
                continue
            window_start = current_date - pd.Timedelta(days=7)
            density.append(int(((dates <= current_date) & (dates > window_start)).sum()))
        out["games_last_7d"] = density
    else:
        out["rest_days"] = np.nan
        out["is_back_to_back"] = 0
        out["games_last_7d"] = np.nan

    if "home_away" in out.columns:
        venue = out["home_away"].astype(str).str.strip().str.lower()
        prev_venue = venue.shift(1)
        out["travel_flag"] = ((venue != prev_venue) & prev_venue.notna()).astype(int)
    else:
        out["travel_flag"] = 0

    if "minutes" in out.columns:
        minutes = pd.to_numeric(out["minutes"], errors="coerce")
        min_periods = max(3, injury_window // 2)
        roll_mean = minutes.rolling(window=injury_window, min_periods=min_periods).mean()
        roll_std = minutes.rolling(window=injury_window, min_periods=min_periods).std().replace(0, np.nan)
        injury_proxy = ((roll_mean - minutes) / roll_std).clip(lower=0)
        out["injury_proxy"] = injury_proxy.fillna(0).clip(upper=3.0)
    else:
        out["injury_proxy"] = 0.0

    return out


def add_rolling_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add rolling mean/std features for common prop stats using a unified schema."""
    if window < 2:
        raise ValueError("window must be >= 2")

    df = _standardize_columns(df.copy())
    rolling_kwargs = {"window": window, "min_periods": window}

    for stat in ("points", "assists", "rebounds", "pra"):
        if stat in df.columns:
            df[f"rolling_mean_{stat}"] = df[stat].rolling(**rolling_kwargs).mean()
            df[f"rolling_std_{stat}"] = df[stat].rolling(**rolling_kwargs).std()

    minutes = df["minutes"].replace(0, np.nan)
    df["points_per_minute"] = df["points"] / minutes
    df["rolling_mean_minutes"] = df["minutes"].rolling(**rolling_kwargs).mean()
    df["rolling_std_minutes"] = df["minutes"].rolling(**rolling_kwargs).std()
    df["rolling_mean_points_per_minute"] = df["points_per_minute"].rolling(**rolling_kwargs).mean()

    # Backward-compatible aliases for legacy scripts.
    if "rolling_mean_points" in df.columns:
        df["pts_mean"] = df["rolling_mean_points"]
    if "rolling_std_points" in df.columns:
        df["pts_std"] = df["rolling_std_points"]
    df["min_mean"] = df["rolling_mean_minutes"]
    df["ppm"] = df["points_per_minute"]
    df["ppm_mean"] = df["rolling_mean_points_per_minute"]

    return add_context_features(df)
