import pandas as pd


def add_rolling_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.copy()

    df["pts_mean"] = df["PTS"].rolling(window).mean()
    df["pts_std"] = df["PTS"].rolling(window).std()
    df["min_mean"] = df["MIN"].rolling(window).mean()
    df["ppm"] = df["PTS"] / df["MIN"]

    df["ppm_mean"] = df["ppm"].rolling(window).mean()

    return df
