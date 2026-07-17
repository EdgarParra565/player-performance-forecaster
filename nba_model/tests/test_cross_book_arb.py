"""Tests for the WS8 cross-book layer.

Covers the three distinct signals the module must keep separate:
  * line shopping / middle candidates from the DFS board (``-110`` assumed) via
    ``find_cross_book_opportunities`` — never an arb flag.
  * TRUE two-way arbitrage from REAL posted odds via ``detect_two_way_arb`` —
    flagged only when the raw implied sum < 1.0 in an executable direction.
  * a fixture-DB round trip: seed ``web_prop_cards`` for 2 books, run
    ``fetch_latest_prop_lines`` -> ``score_prop_edges`` ->
    ``find_cross_book_opportunities``; and ``betting_lines`` ->
    ``fetch_two_way_lines`` -> ``detect_two_way_arb``.
"""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import norm

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model import cross_book_arb as cba
from nba_model.model import edge_scanner as es

LEBRON_ID = 2544
SOLO_ID = 9001


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


def _scored_row(book, player, stat, line, mu, sigma):
    """A minimal row shaped like ``score_prop_edges`` output."""
    return {
        "book": book, "player_name": player, "stat_type": stat,
        "book_line": line, "model_mu": mu, "model_sigma": sigma,
        "line_vs_mu": line - mu, "p_over": 0.5, "p_under": 0.5,
        "best_side": "over", "model_edge": 0.0, "ev_best": 0.0,
        "consensus_mean": mu, "pct_from_consensus": 0.0,
        "observed_hours_ago": 1.0, "observed_at_utc": "2026-01-01 00:00:00",
        "n_games_used": 25,
    }


# ---------------------------------------------------------------------------
# find_cross_book_opportunities — pure, on a synthetic scored_df
# ---------------------------------------------------------------------------

class FindCrossBookOpportunitiesTests(unittest.TestCase):
    MU = 18.5
    SIGMA = 3.0

    def _three_book_df(self):
        # Book names chosen so alphabetical order != line order, proving the
        # best over/under books are driven by the LINE, not the name.
        return pd.DataFrame([
            _scored_row("Zeta", "LeBron James", "points", 17.5, self.MU, self.SIGMA),
            _scored_row("Alpha", "LeBron James", "points", 18.5, self.MU, self.SIGMA),
            _scored_row("Mid", "LeBron James", "points", 19.5, self.MU, self.SIGMA),
        ])

    def test_gap_and_best_books(self):
        out = cba.find_cross_book_opportunities(self._three_book_df())
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(int(row["n_books"]), 3)
        self.assertEqual(row["line_min"], 17.5)
        self.assertEqual(row["line_max"], 19.5)
        self.assertAlmostEqual(row["line_gap"], 2.0, places=6)
        self.assertAlmostEqual(row["middle_size"], 2.0, places=6)
        self.assertEqual(row["best_over_book"], "Zeta")   # lowest line 17.5
        self.assertEqual(row["best_under_book"], "Mid")    # highest line 19.5
        self.assertAlmostEqual(row["consensus_mean"], 18.5, places=6)

    def test_p_over_recomputed_from_mu_sigma(self):
        out = cba.find_cross_book_opportunities(self._three_book_df())
        row = out.iloc[0]
        exp_min = float(norm.sf(17.5, loc=self.MU, scale=self.SIGMA))
        exp_max = float(norm.sf(19.5, loc=self.MU, scale=self.SIGMA))
        self.assertAlmostEqual(row["p_over_at_line_min"], round(exp_min, 4), places=4)
        self.assertAlmostEqual(row["p_over_at_line_max"], round(exp_max, 4), places=4)
        # Lower line is easier to clear an over -> higher P(over).
        self.assertGreater(row["p_over_at_line_min"], row["p_over_at_line_max"])

    def test_wide_gap_is_middle_candidate(self):
        out = cba.find_cross_book_opportunities(self._three_book_df())
        self.assertEqual(out.iloc[0]["opportunity_type"], cba.OPP_MIDDLE)

    def test_narrow_gap_is_line_gap_only(self):
        df = pd.DataFrame([
            _scored_row("A", "Player X", "points", 18.0, 18.0, 3.0),
            _scored_row("B", "Player X", "points", 18.5, 18.0, 3.0),
        ])
        out = cba.find_cross_book_opportunities(df)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out.iloc[0]["line_gap"], 0.5, places=6)
        self.assertEqual(out.iloc[0]["opportunity_type"], cba.OPP_LINE_GAP)

    def test_single_book_excluded(self):
        df = pd.DataFrame([
            _scored_row("Solo", "Only One", "points", 20.5, 20.0, 3.0),
        ])
        out = cba.find_cross_book_opportunities(df)
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), cba.CROSS_BOOK_COLUMNS)

    def test_empty_input_shaped(self):
        out = cba.find_cross_book_opportunities(
            pd.DataFrame(columns=es.SCORED_COLUMNS)
        )
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), cba.CROSS_BOOK_COLUMNS)

    def test_sorted_by_gap_desc(self):
        df = pd.concat([
            self._three_book_df(),  # gap 2.0
            pd.DataFrame([
                _scored_row("A", "Small Gap", "assists", 6.5, 6.0, 2.0),
                _scored_row("B", "Small Gap", "assists", 7.0, 6.0, 2.0),
            ]),  # gap 0.5
        ], ignore_index=True)
        out = cba.find_cross_book_opportunities(df)
        gaps = list(out["line_gap"])
        self.assertEqual(gaps, sorted(gaps, reverse=True))


# ---------------------------------------------------------------------------
# detect_two_way_arb — pure, on synthetic betting_lines-shaped frames
# ---------------------------------------------------------------------------

class DetectTwoWayArbTests(unittest.TestCase):
    def _row(self, book, over_odds, under_odds, line=25.5):
        return {
            "player_name": "LeBron James", "stat_type": "points",
            "game_date": "2026-01-15", "book": book, "line_value": line,
            "over_odds": over_odds, "under_odds": under_odds,
        }

    def test_real_odds_arb_flagged_with_positive_margin(self):
        # OVER +105 @ A (implied .4878) + UNDER +102 @ B (implied .4950)
        # raw sum .9828 < 1.0 -> arb. The reverse direction (OVER @ B / UNDER @ A)
        # sums > 1 and must NOT flag, so exactly one row emits.
        df = pd.DataFrame([
            self._row("BookA", over_odds=105, under_odds=-130),
            self._row("BookB", over_odds=-140, under_odds=102),
        ])
        out = cba.detect_two_way_arb(df)
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(row["over_book"], "BookA")
        self.assertEqual(row["under_book"], "BookB")
        self.assertLess(row["combined_implied"], 1.0)
        self.assertGreater(row["guaranteed_margin"], 0.0)
        self.assertAlmostEqual(
            row["guaranteed_margin"], round(1.0 - row["combined_implied"], 4),
            places=4,
        )

    def test_minus_110_pair_not_flagged(self):
        df = pd.DataFrame([
            self._row("BookA", over_odds=-110, under_odds=-110),
            self._row("BookB", over_odds=-110, under_odds=-110),
        ])
        out = cba.detect_two_way_arb(df)
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), cba.ARB_COLUMNS)

    def test_missing_odds_not_flagged(self):
        df = pd.DataFrame([
            self._row("BookA", over_odds=None, under_odds=None),
            self._row("BookB", over_odds=None, under_odds=None),
        ])
        out = cba.detect_two_way_arb(df)
        self.assertTrue(out.empty)

    def test_dead_zone_direction_not_flagged(self):
        # Over at the HIGHER line + under at the LOWER line leaves a dead zone
        # where both legs lose -> never an arb even if implied sum < 1.
        df = pd.DataFrame([
            self._row("BookA", over_odds=105, under_odds=-130, line=27.5),
            self._row("BookB", over_odds=-140, under_odds=102, line=25.5),
        ])
        out = cba.detect_two_way_arb(df)
        self.assertTrue(out.empty)

    def test_empty_input_shaped(self):
        out = cba.detect_two_way_arb(pd.DataFrame(columns=cba.TWO_WAY_LINE_COLUMNS))
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), cba.ARB_COLUMNS)

    def test_distinct_players_with_blank_names_not_cross_paired(self):
        # Two DIFFERENT players (distinct player_id) whose names both resolved to
        # '' must NOT be paired into a phantom cross-player arb.
        df = pd.DataFrame([
            {"player_id": 1, "player_name": "", "stat_type": "points",
             "game_date": "2026-01-15", "book": "BookA", "line_value": 25.5,
             "over_odds": 105, "under_odds": -130},
            {"player_id": 2, "player_name": "", "stat_type": "points",
             "game_date": "2026-01-15", "book": "BookB", "line_value": 25.5,
             "over_odds": -140, "under_odds": 102},
        ])
        out = cba.detect_two_way_arb(df)
        self.assertTrue(out.empty)  # different players -> no valid pair


# ---------------------------------------------------------------------------
# Fixture-DB round trips
# ---------------------------------------------------------------------------

class CrossBookDbRoundTripTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "nba.db")
        now = datetime.now(timezone.utc)
        self.recent = _utc(now - timedelta(hours=1))
        with DatabaseManager(db_path=self.db_path) as db:
            db.upsert_active_players_reference([
                {"player_id": LEBRON_ID, "player_name": "LeBron James",
                 "synced_at_utc": self.recent},
            ])
            # points mean exactly 19.0 over 10 games.
            pts = [16, 18, 20, 22, 16, 18, 20, 22, 18, 20]
            db.insert_game_logs(pd.DataFrame(
                [_game_row(LEBRON_ID, i + 1, p) for i, p in enumerate(pts)]
            ))
            db.insert_web_prop_cards([
                _card("Underdog", "LeBron James", "points", 17.5, "over",
                      self.recent, 1),
                _card("PrizePicks", "LeBron James", "points", 20.5, "over",
                      self.recent, 2),
            ])

    def tearDown(self):
        self._tmp.cleanup()

    def test_web_prop_cards_to_cross_book(self):
        lines = es.fetch_latest_prop_lines(self.db_path)
        scored = es.score_prop_edges(lines, db_path=self.db_path, n_games=25)
        out = cba.find_cross_book_opportunities(scored)
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(row["player_name"], "LeBron James")
        self.assertEqual(int(row["n_books"]), 2)
        self.assertEqual(row["line_min"], 17.5)
        self.assertEqual(row["line_max"], 20.5)
        self.assertAlmostEqual(row["line_gap"], 3.0, places=6)
        self.assertEqual(row["best_over_book"], "Underdog")    # line 17.5
        self.assertEqual(row["best_under_book"], "PrizePicks")  # line 20.5
        self.assertAlmostEqual(row["consensus_mean"], 19.0, places=3)
        self.assertAlmostEqual(row["model_mu"], 19.0, places=3)
        # DFS -110 data can never be arb — this path emits a middle candidate.
        self.assertEqual(row["opportunity_type"], cba.OPP_MIDDLE)


class TwoWayArbDbRoundTripTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "nba.db")
        with DatabaseManager(db_path=self.db_path) as db:
            db.conn.execute(
                "INSERT INTO players (player_id, name) VALUES (?, ?)",
                (LEBRON_ID, "LeBron James"),
            )
            db.conn.commit()
            db.insert_betting_lines_records([
                {"player_id": LEBRON_ID, "game_date": "2026-01-15",
                 "book": "BookA", "stat_type": "points", "line_value": 25.5,
                 "over_odds": 105, "under_odds": -130},
                {"player_id": LEBRON_ID, "game_date": "2026-01-15",
                 "book": "BookB", "stat_type": "points", "line_value": 25.5,
                 "over_odds": -140, "under_odds": 102},
            ])

    def tearDown(self):
        self._tmp.cleanup()

    def test_betting_lines_to_arb(self):
        odds_lines = cba.fetch_two_way_lines(self.db_path)
        self.assertEqual(len(odds_lines), 2)
        self.assertTrue((odds_lines["player_name"] == "LeBron James").all())
        arbs = cba.detect_two_way_arb(odds_lines)
        self.assertEqual(len(arbs), 1)
        row = arbs.iloc[0]
        self.assertEqual(row["over_book"], "BookA")
        self.assertEqual(row["under_book"], "BookB")
        self.assertGreater(row["guaranteed_margin"], 0.0)

    def test_book_filter_and_empty_selection(self):
        only_a = cba.fetch_two_way_lines(self.db_path, books=["booka"])
        self.assertTrue((only_a["book"] == "BookA").all())
        empty = cba.fetch_two_way_lines(self.db_path, books=[])
        self.assertTrue(empty.empty)
        self.assertEqual(list(empty.columns), cba.TWO_WAY_LINE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
