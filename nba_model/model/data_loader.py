from nba_api.stats.endpoints import playergamelog
import pandas as pd


def load_player_logs(player_id: int, season: str = "2023-24") -> pd.DataFrame:
    """
    Pulls NBA player game logs and returns cleaned DataFrame
    """
    gamelog = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season
    )

    df = gamelog.get_data_frames()[0]

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    return df
