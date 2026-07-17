"""Tests for the Book Edge Scanner (model-vs-line edge over the scraped slate).

Seeds a throwaway SQLite DB with known player history (μ = 19 for points) and
scraped ``web_prop_cards`` rows, then asserts the core user story:
    Underdog LeBron points 17.5 with μ=19 → P(over) > 50% (soft line),
    PrizePicks 20.5 → P(over) < 50%.
"""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model import edge_scanner as es

LEBRON_ID = 2544
ONEGAME_ID = 9001


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


class EdgeScannerTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "nba.db")
        now = datetime.now(timezone.utc)
        self.recent = _utc(now - timedelta(hours=1))
        self.older = _utc(now - timedelta(hours=6))
        self.stale = _utc(now - timedelta(days=5))
        with DatabaseManager(db_path=self.db_path) as db:
            db.upsert_active_players_reference([
                {"player_id": LEBRON_ID, "player_name": "LeBron James",
                 "synced_at_utc": self.recent},
                {"player_id": ONEGAME_ID, "player_name": "One Gamer",
                 "synced_at_utc": self.recent},
            ])
            # LeBron points: mean exactly 19.0 over 10 games.
            pts = [16, 18, 20, 22, 16, 18, 20, 22, 18, 20]
            db.insert_game_logs(pd.DataFrame(
                [_game_row(LEBRON_ID, i + 1, p) for i, p in enumerate(pts)]
            ))
            # One Gamer: a single game (can't fit a normal).
            db.insert_game_logs(pd.DataFrame([_game_row(ONEGAME_ID, 1, 25)]))
        self._seed_cards()

    def tearDown(self):
        self._tmp.cleanup()

    def _seed_cards(self):
        with DatabaseManager(db_path=self.db_path) as db:
            db.insert_web_prop_cards([
                # Underdog soft over: line 17.5 < μ=19.
                _card("Underdog", "LeBron James", "points", 17.5, "over",
                      self.recent, 1),
                # Older Underdog snapshot with a different line — must be
                # superseded by the newer 17.5 row.
                _card("Underdog", "LeBron James", "points", 25.5, "over",
                      self.older, 2),
                # PrizePicks: line 20.5 > μ → P(over) < 50%.
                _card("PrizePicks", "LeBron James", "points", 20.5, "over",
                      self.recent, 3),
                # Pick6 at the mean: ~coinflip, -EV vs -110 juice.
                _card("Pick6", "LeBron James", "points", 19.0, "over",
                      self.recent, 4),
                # One Gamer prop (insufficient history → skipped in scoring).
                _card("Underdog", "One Gamer", "points", 10.5, "over",
                      self.recent, 5),
            ])


class FetchLatestPropLinesTests(EdgeScannerTestBase):
    def test_dedup_keeps_newest_per_book(self):
        df = es.fetch_latest_prop_lines(self.db_path)
        ud = df[(df["book"] == "Underdog") &
                (df["player_name"] == "LeBron James")]
        self.assertEqual(len(ud), 1)
        self.assertEqual(float(ud.iloc[0]["line_value"]), 17.5)

    def test_book_filter_excludes_others(self):
        df = es.fetch_latest_prop_lines(self.db_path, books=["underdog"])
        self.assertTrue((df["book"].str.lower() == "underdog").all())
        self.assertFalse((df["book"] == "PrizePicks").any())

    def test_empty_book_selection_returns_shaped_empty(self):
        df = es.fetch_latest_prop_lines(self.db_path, books=[])
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), es.LINE_COLUMNS)

    def test_stale_rows_excluded_by_lookback(self):
        # A 5-day-old card with a 48h lookback must not appear.
        with DatabaseManager(db_path=self.db_path) as db:
            db.insert_web_prop_cards([
                _card("BetMGM", "LeBron James", "points", 30.5, "over",
                      self.stale, 99),
            ])
        df = es.fetch_latest_prop_lines(self.db_path, since_hours=48)
        self.assertFalse((df["book"] == "BetMGM").any())


class ScorePropEdgesTests(EdgeScannerTestBase):
    def _scored(self):
        lines = es.fetch_latest_prop_lines(self.db_path)
        return es.score_prop_edges(lines, db_path=self.db_path, n_games=25)

    def test_user_story_soft_over_above_half(self):
        scored = self._scored()
        ud = scored[(scored["book"] == "Underdog") &
                    (scored["player_name"] == "LeBron James")].iloc[0]
        self.assertAlmostEqual(ud["model_mu"], 19.0, places=3)
        self.assertEqual(ud["book_line"], 17.5)
        self.assertGreater(ud["p_over"], 0.5)
        self.assertEqual(ud["best_side"], "over")
        self.assertLess(ud["line_vs_mu"], 0)  # soft line for overs

    def test_line_above_mu_is_under_half(self):
        scored = self._scored()
        pp = scored[scored["book"] == "PrizePicks"].iloc[0]
        self.assertEqual(pp["book_line"], 20.5)
        self.assertLess(pp["p_over"], 0.5)
        self.assertEqual(pp["best_side"], "under")

    def test_edge_sign_and_positive_ev_vs_minus_110(self):
        scored = self._scored()
        ud = scored[scored["book"] == "Underdog"].iloc[0]
        # p_best ~0.75 well above the 0.5238 breakeven → +edge, +EV.
        self.assertGreater(ud["model_edge"], 0)
        self.assertGreater(ud["ev_best"], 0)

    def test_at_mean_line_is_negative_ev(self):
        scored = self._scored()
        p6 = scored[scored["book"] == "Pick6"].iloc[0]
        self.assertEqual(p6["book_line"], 19.0)
        # P(over)=0.5 < 0.5238 breakeven → negative edge and EV.
        self.assertLess(p6["model_edge"], 0)
        self.assertLess(p6["ev_best"], 0)

    def test_insufficient_history_player_skipped(self):
        scored = self._scored()
        self.assertFalse((scored["player_name"] == "One Gamer").any())

    def test_consensus_mean_across_books(self):
        scored = self._scored()
        lebron = scored[scored["player_name"] == "LeBron James"]
        # Lines 17.5 / 20.5 / 19.0 → consensus 19.0 for every row.
        for cmean in lebron["consensus_mean"]:
            self.assertAlmostEqual(cmean, 19.0, places=3)

    def test_empty_lines_returns_shaped_empty(self):
        out = es.score_prop_edges(
            pd.DataFrame(columns=es.LINE_COLUMNS), db_path=self.db_path,
        )
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), es.SCORED_COLUMNS_FULL)
        # SCORED_COLUMNS stays the canonical ordered prefix (Agent B contract).
        self.assertEqual(
            list(out.columns)[: len(es.SCORED_COLUMNS)], es.SCORED_COLUMNS)


class TopEdgesTests(EdgeScannerTestBase):
    def _scored(self):
        lines = es.fetch_latest_prop_lines(self.db_path)
        return es.score_prop_edges(lines, db_path=self.db_path, n_games=25)

    def test_sorted_by_edge_desc(self):
        top = es.top_edges(self._scored())
        edges = list(top["model_edge"])
        self.assertEqual(edges, sorted(edges, reverse=True))

    def test_only_positive_ev_drops_at_mean_row(self):
        top = es.top_edges(self._scored(), only_positive_ev=True)
        self.assertFalse((top["book"] == "Pick6").any())
        self.assertTrue((top["ev_best"] > 0).all())

    def test_min_p_over_filter(self):
        top = es.top_edges(self._scored(), min_p_over=0.7)
        self.assertTrue((top["p_over"] >= 0.7).all())

    def test_limit_caps_rows(self):
        top = es.top_edges(self._scored(), limit=1)
        self.assertEqual(len(top), 1)

    def test_empty_scored_returns_shaped_empty(self):
        out = es.top_edges(pd.DataFrame(columns=es.SCORED_COLUMNS_FULL))
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), es.SCORED_COLUMNS_FULL)


if __name__ == "__main__":
    unittest.main()
