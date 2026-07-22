"""FastAPI tests for the read-only flagship-UI backend.

Seeds a throwaway SQLite DB exactly like the existing test harnesses
(``test_edge_scanner`` etc.): active-players ref + known game history + scraped
``web_prop_cards`` + a finished ``games`` matchup. Then points the service at
that DB via ``NBA_DB_PATH`` and drives it through ``httpx``'s TestClient.
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
    with DatabaseManager(db_path=db_path) as db:
        db.upsert_active_players_reference([
            {"player_id": LEBRON_ID, "player_name": "LeBron James",
             "synced_at_utc": recent},
        ])
        pts = [16, 18, 20, 22, 16, 18, 20, 22, 18, 20]  # mean 19.0
        db.insert_game_logs(pd.DataFrame(
            [_game_row(LEBRON_ID, i + 1, p) for i, p in enumerate(pts)]
        ))
        db.insert_web_prop_cards([
            _card("Underdog", "LeBron James", "points", 17.5, "over", recent, 1),
            _card("PrizePicks", "LeBron James", "points", 20.5, "over", recent, 2),
        ])
        # One finished matchup so recent-games has a row (two team-rows / game).
        db.conn.executemany(
            """
            INSERT OR REPLACE INTO games
                (game_id, season, season_type, game_date, team_id, team_abbrev,
                 team_name, matchup, home_away, opponent_abbrev, result, pts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("0022400001", "2024-25", "Regular Season", "2025-04-10",
                 1610612747, "LAL", "Los Angeles Lakers", "LAL vs. DEN",
                 "home", "DEN", "W", 120),
                ("0022400001", "2024-25", "Regular Season", "2025-04-10",
                 1610612743, "DEN", "Denver Nuggets", "DEN @ LAL",
                 "away", "LAL", "L", 110),
            ],
        )
        db.conn.commit()


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.db_path = str(Path(cls._tmp.name) / "nba.db")
        _seed(cls.db_path)
        os.environ["NBA_DB_PATH"] = cls.db_path
        # Import the app AFTER env is set (config reads it per-request anyway).
        from api.main import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("NBA_DB_PATH", None)
        cls._tmp.cleanup()

    def test_health_ok(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["db_exists"])
        self.assertEqual(r.headers.get("Cache-Control"), "no-store")
        self.assertGreaterEqual(body["table_counts"]["game_logs"], 10)

    def test_meta(self):
        r = self.client.get("/api/meta")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("points", body["stats"])
        self.assertIn("underdog", [b.lower() for b in body["books"]])

    def test_slate_kpis(self):
        r = self.client.get("/api/slate/kpis")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["games_in_db"], 1)
        self.assertEqual(body["players_tracked"], 1)
        self.assertGreaterEqual(body["books_producing"], 2)
        self.assertIsNotNone(body["freshest_scrape_utc"])
        self.assertIn("max-age", r.headers.get("Cache-Control", ""))

    def test_recent_games(self):
        r = self.client.get("/api/slate/recent-games?n=5")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["rows"][0]["winner"], "LAL")

    def test_edges_core_story(self):
        # Underdog 17.5 (< mean 19) should be a +edge over; PrizePicks 20.5 not.
        r = self.client.get("/api/slate/edges?limit=50")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["n_lines"], 2)
        by_book = {row["book"].lower(): row for row in body["rows"]}
        self.assertIn("underdog", by_book)
        self.assertEqual(by_book["underdog"]["best_side"], "over")
        self.assertGreater(by_book["underdog"]["p_over"], 0.5)
        self.assertEqual(by_book["prizepicks"]["best_side"], "under")

    def test_edges_bad_model_mode(self):
        r = self.client.get("/api/slate/edges?model_mode=bogus")
        self.assertEqual(r.status_code, 400)

    def test_edges_bad_stat(self):
        r = self.client.get("/api/slate/edges?stats=notastat")
        self.assertEqual(r.status_code, 400)

    def test_player_search(self):
        r = self.client.get("/api/players/search?q=lebron")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["rows"][0]["player_id"], LEBRON_ID)

    def test_player_detail(self):
        r = self.client.get(
            f"/api/players/{LEBRON_ID}?stat=points&n_games=10")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["player_name"], "LeBron James")
        self.assertEqual(body["n_games"], 10)
        self.assertAlmostEqual(body["kpis"]["mu"], 19.0, places=3)
        self.assertEqual(len(body["series"]), 10)
        self.assertTrue(body["histogram"])
        self.assertTrue(body["fitted"])
        books = {row["book"].lower() for row in body["book_lines"]}
        self.assertIn("underdog", books)

    def test_player_detail_bad_stat(self):
        r = self.client.get(f"/api/players/{LEBRON_ID}?stat=notastat")
        self.assertEqual(r.status_code, 400)

    def test_player_detail_unknown_id(self):
        r = self.client.get("/api/players/999999?stat=points")
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main()
