"""Correlation and covariance utilities for multi-leg simulation."""

import numpy as np
import pandas as pd


def _nearest_psd(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return a symmetric positive semidefinite approximation."""
    symmetric = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    eigvals = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _to_psd_correlation(matrix: np.ndarray, max_abs_corr: float = 0.999) -> np.ndarray:
    """Project matrix to a valid correlation matrix (PSD with unit diagonal)."""
    psd = _nearest_psd(matrix)
    diag = np.sqrt(np.clip(np.diag(psd), 1e-8, None))
    corr = psd / np.outer(diag, diag)
    corr = np.clip(corr, -abs(float(max_abs_corr)), abs(float(max_abs_corr)))
    np.fill_diagonal(corr, 1.0)
    return corr


def calibrate_correlations(
    player_stats_df: pd.DataFrame,
    stats_cols: list[str],
    min_games: int = 8,
    shrinkage: float = 0.15,
    max_abs_corr: float = 0.95,
):
    """
    Compute robust correlation matrix for selected stat columns.

    Args:
        player_stats_df: Historical player game-level stats.
        stats_cols: Column names to include in correlation calibration.
        min_games: Minimum complete-game sample required before using empirical correlations.
        shrinkage: Blend weight toward identity matrix for stability (0=no shrinkage, 1=identity).
        max_abs_corr: Absolute correlation cap for off-diagonal entries.

    Returns:
        pandas.DataFrame: PSD-safe correlation matrix.
    """
    if not stats_cols:
        raise ValueError("stats_cols must contain at least one column")

    missing_cols = [col for col in stats_cols if col not in player_stats_df.columns]
    if missing_cols:
        raise KeyError(f"Missing stats columns for correlation calibration: {missing_cols}")

    if len(stats_cols) == 1:
        return pd.DataFrame([[1.0]], index=stats_cols, columns=stats_cols)

    stats = (
        player_stats_df[stats_cols]
        .apply(pd.to_numeric, errors="coerce")
        .dropna(subset=stats_cols)
    )
    n_stats = len(stats_cols)
    identity = np.eye(n_stats, dtype=float)

    if len(stats) < max(2, int(min_games)):
        return pd.DataFrame(identity, index=stats_cols, columns=stats_cols)

    corr = stats.corr().to_numpy(dtype=float)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2.0

    corr_cap = abs(float(max_abs_corr))
    corr = np.clip(corr, -corr_cap, corr_cap)
    np.fill_diagonal(corr, 1.0)

    shrink_weight = min(max(float(shrinkage), 0.0), 1.0)
    corr = (1.0 - shrink_weight) * corr + shrink_weight * identity
    corr = _to_psd_correlation(corr, max_abs_corr=max_abs_corr)

    return pd.DataFrame(corr, index=stats_cols, columns=stats_cols)


def covariance_matrix(
    corr_matrix: pd.DataFrame,
    stds: dict[str, float],
    ensure_psd: bool = True,
    min_variance: float = 1e-6,
):
    """
    Convert correlation matrix to covariance matrix using per-stat std dev.

    Args:
        corr_matrix: Correlation matrix for N stats.
        stds: Mapping of stat column -> standard deviation.
        ensure_psd: Whether to project covariance matrix to PSD for simulation stability.
        min_variance: Floor used when clipping non-positive variances.

    Returns:
        numpy.ndarray: NxN covariance matrix.
    """
    cols = list(corr_matrix.columns)
    missing_stds = [col for col in cols if col not in stds]
    if missing_stds:
        raise KeyError(f"Missing std values for covariance matrix columns: {missing_stds}")

    std_values = np.array([float(stds[col]) for col in cols], dtype=float)
    std_floor = float(np.sqrt(max(float(min_variance), 1e-12)))
    std_values = np.where(np.isfinite(std_values), std_values, std_floor)
    std_values = np.clip(np.abs(std_values), std_floor, None)

    corr_values = corr_matrix.loc[cols, cols].to_numpy(dtype=float)
    corr_values = np.nan_to_num(corr_values, nan=0.0, posinf=0.0, neginf=0.0)
    corr_values = (corr_values + corr_values.T) / 2.0
    np.fill_diagonal(corr_values, 1.0)

    cov_matrix = np.outer(std_values, std_values) * corr_values
    if ensure_psd:
        cov_matrix = _nearest_psd(cov_matrix, eps=max(float(min_variance), 1e-12))

    return cov_matrix
