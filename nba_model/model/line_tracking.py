import csv
from datetime import datetime


def log_line_comparison(
    player: str,
    market: str,
    model_prob: float,
    open_prob: float,
    close_prob: float,
    filename: str = "data/line_tracking.csv"
):
    """
    Logs model probability vs market open/close
    """
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            datetime.utcnow().isoformat(),
            player,
            market,
            round(model_prob, 4),
            round(open_prob, 4),
            round(close_prob, 4)
        ])
