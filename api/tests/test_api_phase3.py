"""Phase 3 endpoints: parlay pricing (compute-only) + paper trades/calibration.

Seeded (varied game logs, bet_log + settled predictions), empty (schema-only
DB), and bad-input cases per endpoint.
"""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from nba_model.data.database.db_manager import DatabaseManager

A, B = 101, 202  # two players, same team, same game dates


def _logs(pid: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(30):
        base = rng.normal(0, 1)  # shared "minutes" factor -> correlated stats
        rows.append({
            "player_id": pid, "game_id": f"g{i}", "game_date": f"2025-03-{i % 28 + 1:02d}"
            if i < 28 else f"2025-04-{i - 27:02d}",
            "season": "2024-25", "matchup": "LAL vs. DEN", "home_away": "home",
            "result": "W", "minutes": 34.0,
            "points": max(0, round(22 + 6 * base + rng.normal(0, 3))),
            "rebounds": max(0, round(8 + 2 * base + rng.normal(0, 1.5))),
            "assists": max(0, round(7 + 1.5 * rng.normal(0, 1))),
            "fgm": 8, "fga": 16, "fg3m": 2, "fg3a": 6, "ftm": 4, "fta": 5,
            "oreb": 2, "dreb": 6, "steals": 1, "blocks": 0, "turnovers": 3,
            "plus_minus": 5,
        })
    return pd.DataFrame(rows)


def _seed(db_path: str, *, with_data: bool = True) -> None:
    rng = np.random.default_rng(7)
    with DatabaseManager(db_path=db_path) as db:
        if not with_data:
            return
        for pid, name in ((A, "Alpha Guard"), (B, "Beta Forward")):
            db.conn.execute(
                "INSERT OR REPLACE INTO players (player_id, name, team) VALUES (?, ?, ?)",
                (pid, name, "LAL"))
            db.insert_game_logs(_logs(pid, rng))
        db.conn.executemany(
            """INSERT INTO bet_log (created_at_utc, game_date, player_id, player_name,
                   stat_type, book, line, side, model_prob, implied_prob, edge,
                   stake_units, status, clv_delta)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                ("2026-10-20 18:00:00", "2026-10-21", A, "Alpha Guard", "points", "fanduel",
                 22.5, "over", 0.58, 0.5238, 0.056, 1.0, "pending", None),
                ("2026-10-19 18:00:00", "2026-10-19", A, "Alpha Guard", "points", "fanduel",
                 21.5, "over", 0.60, 0.5238, 0.076, 1.0, "won", 0.5),
                ("2026-10-18 18:00:00", "2026-10-18", B, "Beta Forward", "rebounds", "dk",
                 8.5, "under", 0.55, 0.5238, 0.026, 2.0, "lost", -0.5),
                ("2026-10-17 18:00:00", "2026-10-17", B, "Beta Forward", "rebounds", "dk",
                 8.0, "over", 0.52, 0.5238, -0.004, 1.0, "push", None),
            ],
        )
        preds = [(A, f"2025-03-{d:02d}", "points", p, "over" if hit else "under")
                 for d, (p, hit) in enumerate(
                     [(0.2, 0), (0.25, 0), (0.3, 1), (0.6, 1), (0.65, 0), (0.7, 1),
                      (0.8, 1), (0.85, 1), (0.9, 1), (0.1, 0)], start=1)]
        db.conn.executemany(
            "INSERT INTO predictions (player_id, game_date, stat_type, prob_over, outcome) "
            "VALUES (?,?,?,?,?)", preds)
        db.conn.commit()


class _Base(unittest.TestCase):
    with_data = True

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.db_path = str(Path(cls._tmp.name) / "nba.db")
        _seed(cls.db_path, with_data=cls.with_data)
        from api.main import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        os.environ["NBA_DB_PATH"] = self.db_path

    def tearDown(self):
        os.environ.pop("NBA_DB_PATH", None)


def _leg(pid=A, stat="points", line=21.5, side="over", **kw):
    return {"player_id": pid, "stat": stat, "line": line, "side": side, **kw}


class ParlaySeededTests(_Base):
    def price(self, legs, **kw):
        return self.client.post("/api/parlay/price", json={"legs": legs, **kw})

    def test_prices_correlated_parlay(self):
        r = self.price([_leg(), _leg(stat="rebounds", line=7.5)])
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(len(d["legs"]), 2)
        for leg in d["legs"]:
            self.assertTrue(0.0 < leg["p_hit"] < 1.0)
        # Points and rebounds share a "minutes" factor -> positive correlation
        # -> joint P(both over) above the naive independent product.
        self.assertGreater(d["correlation"]["matrix"][0][1], 0.2)
        self.assertGreater(d["joint_prob"], d["independent_prob"])
        self.assertAlmostEqual(
            d["independent_prob"], d["legs"][0]["p_hit"] * d["legs"][1]["p_hit"], places=9)
        self.assertEqual(d["combined_american"], 264)  # -110 x -110
        self.assertIn("not validated for real money", d["disclaimer"])
        self.assertFalse(d["correlation_fallback"])

    def test_under_leg_flips_correlation_effect(self):
        over = self.price([_leg(), _leg(stat="rebounds", line=7.5)]).json()
        mixed = self.price([_leg(), _leg(stat="rebounds", line=7.5, side="under")]).json()
        self.assertLess(mixed["joint_prob"], mixed["independent_prob"])
        self.assertAlmostEqual(mixed["legs"][1]["p_hit"], 1 - over["legs"][1]["p_hit"], places=9)

    def test_deterministic_for_same_legs(self):
        legs = [_leg(), _leg(pid=B, stat="assists", line=6.5)]
        self.assertEqual(self.price(legs).json()["joint_prob"], self.price(legs).json()["joint_prob"])

    def test_custom_parlay_odds_drive_ev(self):
        d = self.price([_leg(), _leg(stat="rebounds", line=7.5)], parlay_odds=300).json()
        self.assertTrue(d["offered_is_custom"])
        self.assertEqual(d["offered_american"], 300)
        self.assertAlmostEqual(d["ev_joint"], d["joint_prob"] * 3.0 - (1 - d["joint_prob"]), places=9)

    def test_bad_inputs_are_4xx(self):
        cases = [
            ([_leg()], {}),                                   # < 2 legs
            ([_leg()] * 7, {}),                               # > 6 legs (after parse)
            ([_leg(), _leg()], {}),                           # duplicate leg
            ([_leg(), _leg(stat="steals", line=1.5)], {}),    # non-chartable stat
            ([_leg(), _leg(stat="points", line=999)], {}),    # implausible line
            ([_leg(), _leg(stat="rebounds", line=7.5, odds=0)], {}),  # odds 0
            ([_leg(), _leg(pid=999999, stat="rebounds", line=7.5)], {}),  # unknown player
        ]
        for legs, kw in cases:
            r = self.price(legs, **kw)
            self.assertIn(r.status_code, (400, 404), (legs, r.text))
        # NaN parses as a float, then input_validation rejects it.
        nan = self.client.post("/api/parlay/price",
                               json={"legs": [_leg(), _leg(stat="rebounds", line="nan")]})
        self.assertEqual(nan.status_code, 400)
        self.assertIn("finite", nan.json()["detail"])
        for body in ({"legs": [_leg(side="sideways"), _leg()]},
                     {"legs": [_leg()] * 13},
                     {"legs": [_leg(), _leg(stat="rebounds")], "n_games": 1}):
            self.assertEqual(self.client.post("/api/parlay/price", json=body).status_code, 422)

    def test_nothing_persisted(self):
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            before = conn.execute("SELECT COUNT(*) FROM bet_log").fetchone()[0]
        self.price([_leg(), _leg(stat="rebounds", line=7.5)])
        with sqlite3.connect(self.db_path) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM bet_log").fetchone()[0], before)


class PaperTradesSeededTests(_Base):
    def test_lists_and_summarises(self):
        d = self.client.get("/api/paper-trades").json()
        self.assertEqual(len(d["rows"]), 4)
        s = d["summary"]
        self.assertEqual((s["pending"], s["won"], s["lost"], s["push"]), (1, 1, 1, 1))
        self.assertAlmostEqual(s["win_rate"], 0.5)
        self.assertEqual(s["n_clv"], 2)
        self.assertAlmostEqual(s["mean_clv"], 0.0)
        self.assertAlmostEqual(s["positive_clv_rate"], 0.5)
        # won 1u @ 0.5238 implied (+0.909u) - lost 2u + push 0
        self.assertAlmostEqual(s["est_units"], 1 / 0.5238 - 1 - 2.0, places=3)

    def test_status_filters(self):
        pend = self.client.get("/api/paper-trades", params={"status": "pending"}).json()
        self.assertEqual([r["status"] for r in pend["rows"]], ["pending"])
        sett = self.client.get("/api/paper-trades", params={"status": "settled"}).json()
        self.assertEqual(sorted(r["status"] for r in sett["rows"]), ["lost", "push", "won"])

    def test_calibration_from_predictions(self):
        d = self.client.get("/api/paper-trades/calibration",
                            params={"source": "predictions", "n_buckets": 5}).json()
        self.assertEqual(d["n_settled"], 10)
        self.assertEqual(d["stats_available"], ["points"])
        self.assertTrue(all(0 <= b["realized_rate"] <= 1 for b in d["reliability"]))
        self.assertEqual(sum(b["n"] for b in d["reliability"]), 10)
        self.assertIsNotNone(d["brier"])
        self.assertLess(d["brier"]["brier_score"], 0.25)

    def test_calibration_from_bet_log_excludes_pending_and_push(self):
        d = self.client.get("/api/paper-trades/calibration").json()
        self.assertEqual(d["source"], "bet_log")
        self.assertEqual(d["n_settled"], 2)  # won + lost only

    def test_bad_inputs(self):
        self.assertEqual(self.client.get("/api/paper-trades", params={"status": "open"}).status_code, 400)
        self.assertEqual(self.client.get("/api/paper-trades", params={"limit": 0}).status_code, 422)
        self.assertEqual(self.client.get("/api/paper-trades/calibration",
                                         params={"source": "vibes"}).status_code, 400)
        self.assertEqual(self.client.get("/api/paper-trades/calibration",
                                         params={"n_buckets": 50}).status_code, 422)
        self.assertEqual(self.client.get("/api/paper-trades/calibration",
                                         params={"stat": "bogus"}).status_code, 400)


class EmptyDbTests(_Base):
    with_data = False  # schema only: the offseason / fresh-install state

    def test_empty_paper_trades(self):
        d = self.client.get("/api/paper-trades").json()
        self.assertEqual(d["rows"], [])
        self.assertEqual(d["summary"]["total"], 0)
        self.assertIsNone(d["summary"]["win_rate"])
        self.assertIsNone(d["summary"]["mean_clv"])

    def test_empty_calibration(self):
        for source in ("bet_log", "predictions"):
            d = self.client.get("/api/paper-trades/calibration", params={"source": source}).json()
            self.assertEqual(d["n_settled"], 0)
            self.assertEqual(d["reliability"], [])
            self.assertIsNone(d["brier"])

    def test_parlay_unknown_players_404(self):
        r = self.client.post("/api/parlay/price", json={"legs": [_leg(), _leg(pid=B)]})
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main()
