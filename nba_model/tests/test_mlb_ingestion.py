"""Tests for MLB Stats API ingestion (WS6 MLB-first).

All pure transforms run against committed JSON fixtures shaped like real
``statsapi.mlb.com`` responses (captured 2026-06-28) — NO live network.
Persistence is verified against a temp ``mlb_game_logs`` table, including that
MLB rows never leak into NBA ``game_logs`` queries.
"""

import tempfile
import unittest
from pathlib import Path

from nba_model.data import mlb_results_ingestion as mlb
from nba_model.data.database.db_manager import DatabaseManager

# --- Committed fixtures (trimmed real MLB Stats API shapes) ----------------
SCHEDULE_JSON = {
    "dates": [
        {
            "date": "2026-06-28",
            "games": [
                {
                    "gamePk": 824821,
                    "officialDate": "2026-06-28",
                    "season": "2026",
                    "status": {"abstractGameState": "Final"},
                    "teams": {
                        "home": {"team": {"id": 110, "name": "Baltimore Orioles"}},
                        "away": {"team": {"id": 120, "name": "Washington Nationals"}},
                    },
                }
            ],
        }
    ]
}

BOXSCORE_JSON = {
    "teams": {
        "home": {
            "team": {"id": 110, "abbreviation": "BAL"},
            "players": {
                "ID668939": {
                    "person": {"id": 668939, "fullName": "Adley Rutschman"},
                    "stats": {
                        "batting": {
                            "hits": 1, "doubles": 1, "triples": 0, "homeRuns": 0,
                            "rbi": 0, "runs": 0, "baseOnBalls": 1, "strikeOuts": 1,
                            "stolenBases": 0, "totalBases": 2,
                        }
                    },
                },
                "ID669330": {
                    "person": {"id": 669330, "fullName": "Tyler Wells"},
                    "stats": {
                        "pitching": {
                            "strikeOuts": 7, "earnedRuns": 1, "outs": 18,
                            "hits": 4, "baseOnBalls": 1, "wins": 1,
                        }
                    },
                },
            },
        },
        "away": {
            "team": {"id": 120, "abbreviation": "WSH"},
            "players": {
                "ID000001": {
                    "person": {"id": 1, "fullName": "Test Slugger"},
                    "stats": {
                        "batting": {
                            "hits": 2, "doubles": 0, "triples": 0, "homeRuns": 1,
                            "rbi": 2, "runs": 1, "baseOnBalls": 0, "strikeOuts": 0,
                            "stolenBases": 1, "totalBases": 5,
                        }
                    },
                }
            },
        },
    }
}


class TransformScheduleTests(unittest.TestCase):
    def test_flattens_games(self):
        rows = mlb.transform_schedule(SCHEDULE_JSON)
        self.assertEqual(len(rows), 1)
        g = rows[0]
        self.assertEqual(g["game_pk"], 824821)
        self.assertEqual(g["game_date"], "2026-06-28")
        self.assertEqual(g["season"], 2026)
        self.assertEqual(g["home_team_id"], 110)
        self.assertEqual(g["status"], "Final")

    def test_empty(self):
        self.assertEqual(mlb.transform_schedule({}), [])


class TransformBoxscoreTests(unittest.TestCase):
    def setUp(self):
        self.rows = mlb.transform_boxscore_to_player_logs(
            BOXSCORE_JSON, game_pk=824821, game_date="2026-06-28", season=2026
        )

    def test_groups_and_sport_tag(self):
        self.assertTrue(self.rows)
        self.assertEqual({r["sport"] for r in self.rows}, {"mlb"})
        self.assertEqual({r["player_group"] for r in self.rows}, {"hitting", "pitching"})

    def test_hitter_stats_and_derived_singles(self):
        rut = {r["stat_type"]: r for r in self.rows if r["player_id"] == 668939}
        self.assertEqual(rut["hits"]["value"], 1.0)
        self.assertEqual(rut["total_bases"]["value"], 2.0)
        self.assertEqual(rut["walks"]["value"], 1.0)
        self.assertEqual(rut["strikeouts_batter"]["value"], 1.0)
        # singles = hits - doubles - triples - HR = 1 - 1 = 0
        self.assertEqual(rut["singles"]["value"], 0.0)
        self.assertEqual(rut["hits"]["team"], "BAL")
        self.assertEqual(rut["hits"]["opponent"], "WSH")

    def test_slugger_singles_and_homers(self):
        slug = {r["stat_type"]: r for r in self.rows if r["player_id"] == 1}
        self.assertEqual(slug["home_runs"]["value"], 1.0)
        self.assertEqual(slug["total_bases"]["value"], 5.0)
        self.assertEqual(slug["stolen_bases"]["value"], 1.0)
        # singles = 2 - 0 - 0 - 1 = 1
        self.assertEqual(slug["singles"]["value"], 1.0)

    def test_pitcher_stats(self):
        wells = {r["stat_type"]: r for r in self.rows if r["player_id"] == 669330}
        self.assertEqual(wells["strikeouts_pitcher"]["value"], 7.0)
        self.assertEqual(wells["earned_runs"]["value"], 1.0)
        self.assertEqual(wells["outs_recorded"]["value"], 18.0)
        self.assertEqual(wells["hits_allowed"]["value"], 4.0)
        self.assertEqual(wells["walks_allowed"]["value"], 1.0)
        self.assertEqual(wells["wins"]["value"], 1.0)
        self.assertEqual(wells["strikeouts_pitcher"]["player_group"], "pitching")


class PersistenceTests(unittest.TestCase):
    def _rows(self):
        return mlb.transform_boxscore_to_player_logs(
            BOXSCORE_JSON, game_pk=824821, game_date="2026-06-28", season=2026
        )

    def test_insert_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                first = db.insert_mlb_game_logs(self._rows())
                second = db.insert_mlb_game_logs(self._rows())
                self.assertGreater(first["inserted"], 0)
                self.assertEqual(second["inserted"], 0)
                self.assertEqual(second["duplicates_ignored"], first["inserted"])

    def test_get_player_game_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                db.insert_mlb_game_logs(self._rows())
                logs = db.get_mlb_player_game_logs(669330, "strikeouts_pitcher")
                self.assertEqual(len(logs), 1)
                self.assertEqual(logs[0]["value"], 7.0)

    def test_mlb_rows_do_not_leak_into_nba_game_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "nba.db")
            with DatabaseManager(db_path=db_path) as db:
                db.insert_mlb_game_logs(self._rows())
                # NBA game_logs is a separate table — must be empty.
                n = db.conn.execute("SELECT COUNT(*) FROM game_logs").fetchone()[0]
                self.assertEqual(n, 0)
                # And the NBA per-player getter finds nothing for the MLB id.
                nba_games = db.get_player_games(668939, n_games=10)
                self.assertTrue(nba_games is None or len(nba_games) == 0)


class PybaseballGuardTests(unittest.TestCase):
    def test_supplement_absent_and_guarded(self):
        self.assertFalse(mlb.pybaseball_available())
        with self.assertRaises(RuntimeError) as ctx:
            mlb.fetch_statcast_batter("2026-06-01", "2026-06-02", 668939)
        self.assertIn("pybaseball", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
