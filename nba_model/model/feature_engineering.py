import numpy as np
import pandas as pd


_STAT_COLUMN_ALIASES = {
    "PTS": "points",
    "AST": "assists",
    "REB": "rebounds",
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

    if "points" not in df.columns or "minutes" not in df.columns:
        raise KeyError(
            "add_rolling_stats requires 'points' and 'minutes' columns "
            "(or their NBA API aliases 'PTS' and 'MIN')."
        )
    return df


def add_rolling_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add rolling mean/std features for common prop stats using a unified schema."""
    if window < 2:
        raise ValueError("window must be >= 2")

    df = _standardize_columns(df.copy())
    rolling_kwargs = {"window": window, "min_periods": window}

    for stat in ("points", "assists", "rebounds"):
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

    return df
