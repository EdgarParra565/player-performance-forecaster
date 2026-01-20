import pandas as pd
import numpy as np

def calibrate_correlations(player_stats_df: pd.DataFrame, stats_cols: list):
    """
    Computes correlation matrix between selected stats
    """
    corr_matrix = player_stats_df[stats_cols].corr()
    return corr_matrix

def covariance_matrix(corr_matrix: pd.DataFrame, stds: dict):
    """
    Converts correlation matrix to covariance matrix
    stds: dict of standard deviations for each stat
    """
    vars_matrix = np.diag([stds[col]**2 for col in corr_matrix.columns])
    cov_matrix = np.outer([stds[col] for col in corr_matrix.columns],
                          [stds[col] for col in corr_matrix.columns]) * corr_matrix.values
    return cov_matrix
