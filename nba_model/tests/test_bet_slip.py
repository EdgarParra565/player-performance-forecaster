"""Tests for the paper-trading bet-slip exporter (WS10 Phase 2).

Covers the pure staking gates (min_edge / min_p / max_picks / capped Kelly) on a
synthetic scored frame, plus an end-to-end ``generate_bet_slip`` run over a
fixture DB in both ``--dry-run`` (no side effects) and real (writes artifacts +
``bet_log`` rows) modes.
"""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.evaluation import bet_slip
from nba_model.model import edge_scanner as es

NEUTRAL_ID = 9100
POINTS = [16, 18, 20, 22, 16, 18, 20, 22, 18, 20]  # mean 19.0


def _scored_row(player, stat, line, p_over, edge, distribution="normal"):
    """Build a full-mode scored row with the fields build_bet_slip reads."""
    p_over = float(p_over)
    p_under = 1.0 - p_over
    best_side = "over" if p_over >= p_under else "under"
    base = {c: None for c in es.SCORED_COLUMNS_FULL}
    base.update({
        "book": "Underdog", "player_name": player, "stat_type": stat,
        "book_line": line, "model_mu": line, "model_sigma": 5.0,
        "line_vs_mu": 0.0, "p_over": p_over, "p_under": p_under,
        "best_side": best_side, "model_edge": float(edge), "ev_best": 0.1,
        "consensus_mean": line, "pct_from_consensus": 0.0,
        "observed_hours_ago": 1.0, "observed_at_utc": "2025-04-10 00:00:00",
        "n_games_used": 10, "distribution": distribution, "model_mode": "full",
    })
    return base


class BuildBetSlipGateTests(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame([
            _scored_row("Strong Over", "points", 17.5, 0.75, 0.23),   # kept
            _scored_row("Weak Edge", "points", 20.5, 0.53, 0.01),     # < min_edge
            _scored_row("Strong Under", "assists", 9.5, 0.30, 0.18),  # kept (under)
            _scored_row("Low Prob", "rebounds", 8.5, 0.54, 0.05),     # < min_p
            _scored_row("No Edge", "pra", 40.5, 0.50, 0.00),          # kelly 0
        ], columns=es.SCORED_COLUMNS_FULL)

    def test_gates_keep_only_qualifying_picks(self):
        slip = bet_slip.build_bet_slip(
            self._frame(), game_date="2025-04-10",
            min_edge=0.02, min_p=0.55, max_picks=10,
        )
        kept = {(r.player_name, r.side) for r in slip.itertuples(index=False)}
        self.assertEqual(kept, {("Strong Over", "over"), ("Strong Under", "under")})

    def test_under_pick_uses_best_side_prob(self):
        slip = bet_slip.build_bet_slip(
            self._frame(), game_date="2025-04-10", min_edge=0.02, min_p=0.55)
        su = slip[slip["player_name"] == "Strong Under"].iloc[0]
        self.assertEqual(su["side"], "under")
        self.assertAlmostEqual(su["model_prob"], 0.70, places=4)

    def test_max_picks_caps_and_ranks_by_edge(self):
        slip = bet_slip.build_bet_slip(
            self._frame(), game_date="2025-04-10",
            min_edge=0.02, min_p=0.55, max_picks=1)
        self.assertEqual(len(slip), 1)
        self.assertEqual(slip.iloc[0]["player_name"], "Strong Over")

    def test_kelly_stake_is_capped(self):
        slip = bet_slip.build_bet_slip(
            self._frame(), game_date="2025-04-10",
            min_edge=0.02, min_p=0.55, kelly_cap=0.05, bankroll_units=1.0)
        self.assertTrue((slip["kelly_fraction"] <= 0.05 + 1e-9).all())
        self.assertTrue((slip["stake_units"] <= 0.05 + 1e-9).all())
        self.assertTrue((slip["kelly_fraction"] > 0).all())

    def test_player_id_mapping_applied(self):
        slip = bet_slip.build_bet_slip(
            self._frame(), game_date="2025-04-10", min_edge=0.02, min_p=0.55,
            player_id_by_name={"Strong Over": 42})
        so = slip[slip["player_name"] == "Strong Over"].iloc[0]
        self.assertEqual(so["player_id"], 42)

    def test_empty_frame_returns_shaped_empty(self):
        out = bet_slip.build_bet_slip(
            pd.DataFrame(columns=es.SCORED_COLUMNS_FULL), game_date="2025-04-10")
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), bet_slip.SLIP_COLUMNS)


class GenerateBetSlipIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = str(Path(self._tmp.name) / "nba.db")
        self.artifact_dir = str(Path(self._tmp.name) / "artifacts")
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        for j, p in enumerate(POINTS):
            rows.append({
                "player_id": NEUTRAL_ID, "game_id": f"g{j}",
                "game_date": f"2025-04-{j + 1:02d}", "season": "2024-25",
                "matchup": "LAL vs. DEN", "home_away": "home", "result": "W",
                "minutes": 34.0, "points": p, "rebounds": 8, "assists": 7,
                "fgm": 8, "fga": 16, "fg3m": 2, "fg3a": 6, "ftm": 4, "fta": 5,
                "oreb": 2, "dreb": 6, "steals": 1, "blocks": 0, "turnovers": 3,
                "plus_minus": 5,
            })
        with DatabaseManager(db_path=self.db_path) as db:
            db.upsert_active_players_reference([
                {"player_id": NEUTRAL_ID, "player_name": "Neutral Guy",
                 "synced_at_utc": recent}])
            db.insert_game_logs(pd.DataFrame(rows))
            db.insert_web_prop_cards([{
                "snapshot_id": 1, "source_url": "https://ud.test/nba",
                "book": "Underdog", "observed_at_utc": recent,
                "player_name": "Neutral Guy", "player_classification": "active_nba",
                "stat_type": "points", "line_value": 17.5, "side": "over",
                "parse_confidence": 0.99, "parser_version": "t1",
                "record_sha256": "sha-slip-1",
            }])

    def tearDown(self):
        self._tmp.cleanup()

    def test_dry_run_writes_nothing(self):
        summary = bet_slip.generate_bet_slip(
            db_path=self.db_path, game_date="2025-04-16",
            artifact_dir=self.artifact_dir, dry_run=True,
        )
        self.assertGreaterEqual(summary["picks"], 1)
        self.assertEqual(summary["bet_log_inserted"], 0)
        self.assertIsNone(summary["csv_path"])
        with DatabaseManager(db_path=self.db_path) as db:
            n = db.conn.execute("SELECT COUNT(*) FROM bet_log").fetchone()[0]
        self.assertEqual(n, 0)

    def test_real_run_inserts_bet_log_and_writes_csv(self):
        summary = bet_slip.generate_bet_slip(
            db_path=self.db_path, game_date="2025-04-16",
            artifact_dir=self.artifact_dir, dry_run=False,
        )
        self.assertGreaterEqual(summary["bet_log_inserted"], 1)
        self.assertTrue(Path(summary["csv_path"]).exists())
        with DatabaseManager(db_path=self.db_path) as db:
            row = db.conn.execute(
                "SELECT player_id, status, model_mode FROM bet_log LIMIT 1"
            ).fetchone()
        # player_id resolved from the active-players ref; mode is full.
        self.assertEqual(row[0], NEUTRAL_ID)
        self.assertEqual(row[1], "pending")
        self.assertEqual(row[2], "full")


if __name__ == "__main__":
    unittest.main()
