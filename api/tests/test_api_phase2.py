"""FastAPI tests for the Phase 2 money views: cross-book, line-movement,
team-charts. Seeds a throwaway DB the same way ``test_api.py`` does, then drives
the endpoints via httpx's TestClient.
"""
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from nba_model.data.database.db_manager import DatabaseManager

LEBRON_ID = 2544


def _utc(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _game_row(pid, i, points):
    return {
        "player_id": pid, "game_id": f"g{pid}_{i}",
        "game_date": f"2025-04-{i:02d}", "season": "2024-25",
        "matchup": "LAL vs. DEN", "home_away": "home",
        "result": "W", "minutes": 34.0,
        "points": points, "rebounds": 8, "assists": 7,
        "fgm": 8, "fga": 16, "fg3m": 2, "fg3a": 6, "ftm": 4, "fta": 5,
        "oreb": 2, "dreb": 6, "steals": 1, "blocks": 0, "turnovers": 3,
        "plus_minus": 5,
    }


def _card(book, player, stat, line, side, observed, idx):
    return {
        "snapshot_id": 1,
        "source_url": f"https://{book}.test/nba",
        "book": book,
        "observed_at_utc": observed,
        "player_name": player,
        "player_classification": "active_nba",
        "stat_type": stat,
        "line_value": line,
        "side": side,
        "parse_confidence": 0.99,
        "parser_version": "test-1",
        "record_sha256": f"sha-{book}-{player}-{stat}-{side}-{idx}",
    }


def _seed(db_path: str) -> None:
    now = datetime.now(timezone.utc)
    recent = _utc(now - timedelta(hours=1))
    today = now.strftime("%Y-%m-%d")
    with DatabaseManager(db_path=db_path) as db:
        db.upsert_active_players_reference([
            {"player_id": LEBRON_ID, "player_name": "LeBron James",
             "synced_at_utc": recent},
        ])
        db.conn.execute(
            "INSERT OR REPLACE INTO players (player_id, name, team) VALUES (?, ?, ?)",
            (LEBRON_ID, "LeBron James", "LAL"),
        )
        pts = [16, 18, 20, 22, 16, 18, 20, 22, 18, 20]  # mean 19.0
        db.insert_game_logs(pd.DataFrame(
            [_game_row(LEBRON_ID, i + 1, p) for i, p in enumerate(pts)]
        ))
        # Two DFS books with a 3.0-point gap -> cross-book middle candidate.
        db.insert_web_prop_cards([
            _card("Underdog", "LeBron James", "points", 17.5, "over", recent, 1),
            _card("Underdog", "LeBron James", "points", 17.5, "under", recent, 2),
            _card("PrizePicks", "LeBron James", "points", 20.5, "over", recent, 3),
            _card("PrizePicks", "LeBron James", "points", 20.5, "under", recent, 4),
        ])
        # Real two-way odds forming a TRUE arb: over 17.5 @ +120 (implied .4545)
        # + under 18.5 @ +120 (.4545) -> combined .909 < 1, executable.
        db.conn.executemany(
            """INSERT INTO betting_lines
               (player_id, game_date, book, stat_type, line_value,
                over_odds, under_odds, scraped_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (LEBRON_ID, today, "FanDuel", "points", 17.5, 120, -140, recent),
                (LEBRON_ID, today, "DraftKings", "points", 18.5, -140, 120, recent),
            ],
        )
        # Snapshots: two books drifting over three timestamps.
        snaps = []
        for h, (fd, dk) in enumerate([(18.5, 19.0), (18.0, 19.5), (17.5, 20.0)]):
            ts = _utc(now - timedelta(hours=6 - h * 2))
            snaps.append((ts, today, LEBRON_ID, "FanDuel", "player_points",
                          "points", fd, -110, -110))
            snaps.append((ts, today, LEBRON_ID, "DraftKings", "player_points",
                          "points", dk, -110, -110))
        db.conn.executemany(
            """INSERT INTO betting_line_snapshots
               (snapshot_ts_utc, game_date, player_id, book, market_key,
                stat_type, line_value, over_odds, under_odds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            snaps,
        )
        db.conn.commit()


class Phase2ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.db_path = str(Path(cls._tmp.name) / "nba.db")
        _seed(cls.db_path)
        os.environ["NBA_DB_PATH"] = cls.db_path
        from api.main import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("NBA_DB_PATH", None)
        cls._tmp.cleanup()

    # --- cross-book -------------------------------------------------------

    def test_cross_book_line_gap(self):
        r = self.client.get("/api/cross-book?min_gap=0.5")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertGreaterEqual(body["n_lines"], 2)
        row = next(
            x for x in body["rows"]
            if x["player_name"] == "LeBron James" and x["stat_type"] == "points"
        )
        self.assertEqual(row["n_books"], 2)
        self.assertAlmostEqual(row["line_gap"], 3.0, places=3)
        self.assertEqual(row["best_over_book"].lower(), "underdog")
        self.assertEqual(row["best_under_book"].lower(), "prizepicks")
        self.assertEqual(row["opportunity_type"], "middle_candidate")
        self.assertGreaterEqual(body["kpis"]["pairs"], 1)

    def test_cross_book_true_arb(self):
        r = self.client.get("/api/cross-book")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertGreaterEqual(body["kpis"]["arb_count"], 1)
        arb = body["arbs"][0]
        self.assertEqual(arb["over_book"], "FanDuel")
        self.assertEqual(arb["under_book"], "DraftKings")
        self.assertLess(arb["combined_implied"], 1.0)
        self.assertGreater(arb["guaranteed_margin"], 0.0)

    def test_cross_book_bad_model_mode(self):
        r = self.client.get("/api/cross-book?model_mode=bogus")
        self.assertEqual(r.status_code, 400)

    def test_cross_book_high_gap_filters_out(self):
        r = self.client.get("/api/cross-book?min_gap=99")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["rows"], [])
        # KPIs are computed on the unfiltered set, so pairs still counts it.
        self.assertGreaterEqual(body["kpis"]["pairs"], 1)

    # --- line movement ----------------------------------------------------

    def test_line_movement(self):
        r = self.client.get(
            f"/api/players/{LEBRON_ID}/line-movement?stat=points")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["n_books"], 2)
        self.assertEqual(len(body["timestamps"]), 3)
        by_book = {s["book"]: s for s in body["series"]}
        self.assertIn("FanDuel", by_book)
        self.assertEqual(len(by_book["FanDuel"]["points"]), 3)
        self.assertAlmostEqual(by_book["FanDuel"]["open_line"], 18.5, places=3)
        self.assertAlmostEqual(by_book["FanDuel"]["close_line"], 17.5, places=3)
        self.assertAlmostEqual(by_book["FanDuel"]["line_delta"], -1.0, places=3)

    def test_line_movement_empty(self):
        r = self.client.get(f"/api/players/{LEBRON_ID}/line-movement?stat=assists")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["n_snapshots"], 0)
        self.assertEqual(body["series"], [])

    def test_line_movement_bad_stat(self):
        r = self.client.get(
            f"/api/players/{LEBRON_ID}/line-movement?stat=notastat")
        self.assertEqual(r.status_code, 400)

    # --- team chart -------------------------------------------------------

    def test_team_chart(self):
        r = self.client.get("/api/teams/LAL/chart?stat=points&n_games=10")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["team"], "LAL")
        self.assertEqual(body["stat_type"], "points")
        self.assertEqual(len(body["series"]), 10)
        # Team points per game = sum of the single seeded player-row that game.
        self.assertGreater(body["kpis"]["mu"], 0)

    def test_team_chart_bad_team(self):
        r = self.client.get("/api/teams/ZZZ/chart?stat=points")
        self.assertEqual(r.status_code, 400)

    def test_team_chart_empty_team(self):
        r = self.client.get("/api/teams/BOS/chart?stat=points")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["n_games"], 0)
        self.assertEqual(body["series"], [])


if __name__ == "__main__":
    unittest.main()
