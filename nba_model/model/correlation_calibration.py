"""Correlation and covariance utilities for multi-leg simulation."""

import pandas as pd
import numpy as np


def calibrate_correlations(player_stats_df: pd.DataFrame, stats_cols: list):
    """
    Compute empirical correlation matrix for selected stat columns.

    Args:
        player_stats_df: Historical player game-level stats.
        stats_cols: Column names to include in correlation calibration.

    Returns:
        pandas.DataFrame: Correlation matrix.
    """
    corr_matrix = player_stats_df[stats_cols].corr()
    return corr_matrix


def covariance_matrix(corr_matrix: pd.DataFrame, stds: dict):
    """
    Convert a correlation matrix to covariance matrix using per-stat std dev.

    Args:
        corr_matrix: Correlation matrix for N stats.
        stds: Mapping of stat column -> standard deviation.

    Returns:
        numpy.ndarray: NxN covariance matrix.
    """
    cov_matrix = np.outer([stds[col] for col in corr_matrix.columns],
                          [stds[col] for col in corr_matrix.columns]) * corr_matrix.values
    return cov_matrix
