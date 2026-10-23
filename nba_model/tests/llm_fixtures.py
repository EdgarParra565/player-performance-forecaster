"""Shared temp-DB fixture for the WS11 LLM benchmark tests (not a test module).

One star (LeBron, 2544) with 15 prior games, a target game on 2025-01-20 with
a distinctive 61-point box score, and a 2025-01-22 game; Curry with too short
a history; a predictions row for a player missing from ``players``. The
settled ``predictions`` rows cover every selection filter (half-point line,
integer line, model-mean line, implausible line, push, pending) plus several
model configs per prop so the production-config pick is testable.
"""

import json
from datetime import date, timedelta

from nba_model.data.database.db_manager import DatabaseManager

LEBRON, CURRY, GHOST = 2544, 201939, 999
TARGET, NEXT = "2025-01-20", "2025-01-22"
TARGET_POINTS = 61

OPPONENTS = ["BOS", "NYK", "MIA", "DEN", "PHX"]


def _config(distribution, window, line_source="fixed"):
    # Same shape as the live backtest sweeps (context features on).
    return json.dumps({"distribution": distribution, "rolling_window": window,
                       "line_source": line_source, "context_features_enabled": True})


def seed_llm_db(path):
    """Build a migrated temp DB with the fixture rows; returns the
    ``{name: prediction_id}`` map of the baseline rows tests assert on."""
    ids = {}
    with DatabaseManager(path) as db:
        c = db.conn
        c.executemany("INSERT INTO players (player_id, name, team) VALUES (?, ?, ?)",
                      [(LEBRON, "LeBron James", "LAL"), (CURRY, "Stephen Curry", "GSW")])

        def log(pid, gid, day, matchup, home_away, pts, reb, ast, mins=34.6):
            c.execute(
                "INSERT INTO game_logs (player_id, game_id, game_date, season, matchup, "
                "home_away, minutes, points, rebounds, assists, fg3m) "
                "VALUES (?, ?, ?, '2024-25', ?, ?, ?, ?, ?, ?, 2)",
                (pid, gid, f"{day}T00:00:00", matchup, home_away, mins, pts, reb, ast))

        start = date(2024, 12, 20)
        for i in range(15):
            day = (start + timedelta(days=2 * i)).isoformat()
            opp = OPPONENTS[i % len(OPPONENTS)]
            home = i % 2 == 0
            log(LEBRON, f"G{i:02d}", day, f"LAL {'vs.' if home else '@'} {opp}",
                "home" if home else "away", 24 + i % 5, 7 + i % 3, 8 + i % 2)
        log(LEBRON, "GT1", TARGET, "LAL vs. BOS", "home", TARGET_POINTS, 8, 9)
        log(LEBRON, "GT2", NEXT, "LAL @ NYK", "away", 20, 8, 7)
        log(LEBRON, "GT3", "2025-01-25", "LAL vs. MIA", "home", 30, 9, 9)
        for i, day in enumerate(("2025-01-14", "2025-01-16", "2025-01-18")):
            log(CURRY, f"C{i}", day, "GSW vs. SAC", "home", 25, 5, 6)
        log(CURRY, "CT", TARGET, "GSW @ MIN", "away", 20, 5, 6)

        def pred(name, pid, day, stat, line, prob, outcome, cfg, odds=-110):
            cur = c.execute(
                "INSERT INTO predictions (player_id, game_date, stat_type, predicted_mean, "
                "predicted_std, prob_over, line_value, book_odds, outcome) "
                "VALUES (?, ?, ?, 25, 5, ?, ?, ?, ?)",
                (pid, day, stat, prob, line, odds, outcome))
            c.execute("INSERT INTO prediction_configs (prediction_id, config_json) "
                      "VALUES (?, ?)", (cur.lastrowid, cfg))
            if name:
                ids[name] = cur.lastrowid

        normal10, poisson10 = _config("normal", 10), _config("poisson", 10)
        # 01-20 points 25.5: production config (normal w10) must win.
        pred("pts_prod", LEBRON, TARGET, "points", 25.5, 0.55, "over", normal10)
        pred(None, LEBRON, TARGET, "points", 25.5, 0.60, "over", poisson10)
        pred(None, LEBRON, TARGET, "points", 25.5, 0.52, "over", _config("normal", 5))
        # A model-mean row at the same line: the fixed rows keep the prop real.
        pred(None, LEBRON, TARGET, "points", 25.5, 0.90, "over",
             _config("poisson", 5, line_source="model"))
        # 01-20 rebounds 7.5: default for rebounds is poisson.
        pred(None, LEBRON, TARGET, "rebounds", 7.5, 0.50, "over", normal10)
        pred("reb_prod", LEBRON, TARGET, "rebounds", 7.5, 0.48, "over", poisson10)
        # Implausible: assists 25.5 vs a ~8.5 average.
        pred(None, LEBRON, TARGET, "assists", 25.5, 0.01, "under", normal10)
        # 01-22: half line, integer line, model-mean line, PRA, push, pending.
        pred("pts_next", LEBRON, NEXT, "points", 24.5, 0.45, "under", normal10)
        pred(None, LEBRON, NEXT, "points", 23.0, 0.44, "under", normal10)
        pred(None, LEBRON, NEXT, "points", 22.857, 0.5, "under",
             _config("normal", 10, line_source="model"))
        # Half-point line that exists only as a backtest model mean: dropped.
        pred(None, LEBRON, NEXT, "assists", 7.5, 0.5, "under",
             _config("normal", 10, line_source="model"))
        pred("pra_next", LEBRON, NEXT, "pra", 35.5, 0.62, "under", normal10)
        pred(None, LEBRON, NEXT, "rebounds", 8.0, 0.5, "push", normal10)
        pred(None, LEBRON, "2025-01-25", "points", 26.5, 0.5, None, normal10)
        # Dropped: short history (Curry) and no players row (GHOST).
        pred(None, CURRY, TARGET, "points", 25.5, 0.5, "under", normal10)
        pred(None, GHOST, TARGET, "points", 25.5, 0.5, "under", normal10)
        c.commit()
    return ids


def add_real_prices(path):
    """Two books post both sides at 25.5 for the 01-20 points prop before the
    game-day cutoff (FanDuel twice — only its latest row counts); rows after
    the cutoff (in-play / post-game) must be ignored. De-vigged mean =
    0.516289."""
    with DatabaseManager(path) as db:
        db.conn.executemany(
            "INSERT INTO betting_lines (player_id, game_date, book, stat_type, line_value, "
            "over_odds, under_odds, scraped_at) VALUES (?, ?, ?, 'points', 25.5, ?, ?, ?)",
            [(LEBRON, TARGET, "FanDuel", -300, 250, "2025-01-19 10:00:00"),
             (LEBRON, TARGET, "FanDuel", -120, 100, "2025-01-20 10:00:00"),
             (LEBRON, TARGET, "DraftKings", -115, -105, "2025-01-20 09:00:00"),
             (LEBRON, TARGET, "FanDuel", -900, 600, "2025-01-20 18:30:00"),
             (LEBRON, TARGET, "DraftKings", -2000, 1000, "2025-01-21 02:00:00"),
             (LEBRON, TARGET, "BetMGM", -150, 120, "2025-03-01 12:00:00")])
        db.conn.commit()


EXPECTED_MARKET_PROB = ((120 / 220) / (120 / 220 + 100 / 200)
                        + (115 / 215) / (115 / 215 + 105 / 205)) / 2
