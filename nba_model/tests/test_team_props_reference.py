"""Tests for the props-derived team reference line (Team charts beyond points).

Books post a real team line only for ``points`` (implied team total from the
lobby total + spread). For assists/rebounds/3pm/fgm there's no lobby team line,
so ``_fetch_props_derived_team_reference`` synthesizes one: the sum of the
team's rostered players' cross-book consensus prop lines. These tests pin:

  * happy path — >= min_players with props → value = Σ per-player consensus,
  * thin coverage — below the floor → value None (empty-state),
  * mixed books — a player's consensus is the mean across books (no double count),
  * stale rows — cards outside since_hours are excluded,
  * ineligible stats (pra/minutes) → no derived reference,
  * fetch_team_chart_data integration — non-points gets the derived line while
    points stays byte-identical (derived None, book mean unchanged),
  * team membership resolves via game_logs.matchup, not the sparse players.team.

All offline against a temp SQLite DB seeded with synthetic rows.
"""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.visualization import player_charts as pc


def _utc(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _game_row(pid, i, matchup="LAL vs. DEN"):
    return {
        "player_id": pid, "game_id": f"g{pid}_{i}",
        "game_date": f"2025-04-{i:02d}", "season": "2024-25",
        "matchup": matchup, "home_away": "home", "result": "W",
        "minutes": 32.0, "points": 20, "rebounds": 6, "assists": 5,
        "fgm": 7, "fga": 15, "fg3m": 2, "fg3a": 5, "ftm": 4, "fta": 5,
        "oreb": 1, "dreb": 5, "steals": 1, "blocks": 0, "turnovers": 2,
        "plus_minus": 4,
    }


def _card(book, player, stat, line, observed, idx, classification="active_nba"):
    return {
        "snapshot_id": 1,
        "source_url": f"https://{book}.test/nba",
        "book": book,
        "observed_at_utc": observed,
        "player_name": player,
        "player_classification": classification,
        "stat_type": stat,
        "line_value": line,
        "side": "over",
        "parse_confidence": 0.99,
        "parser_version": "test-1",
        "record_sha256": f"sha-{book}-{player}-{stat}-{idx}",
    }


# Six LAL players with assist lines; two DEN players who must NOT be summed.
LAL_PLAYERS = [(3001, "Al Guard", 7.5), (3002, "Bo Wing", 6.5),
               (3003, "Cy Forward", 5.5), (3004, "Dee Center", 4.5),
               (3005, "Ed Bench", 3.5), (3006, "Fi Rookie", 2.5)]
DEN_PLAYERS = [(4001, "Nikola X", 9.5), (4002, "Jamal Y", 8.5)]


class DerivedTeamReferenceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = str(Path(self._tmp.name) / "nba.db")
        now = datetime.now(timezone.utc)
        self.recent = _utc(now - timedelta(hours=1))
        self.stale = _utc(now - timedelta(hours=100))

    def tearDown(self):
        self._tmp.cleanup()

    def _seed(self, *, assist_players=LAL_PLAYERS, extra_cards=None,
              den_cards=True):
        with DatabaseManager(db_path=self.db_path) as db:
            ref_rows, game_rows, cards = [], [], []
            for pid, name, _line in LAL_PLAYERS + DEN_PLAYERS:
                ref_rows.append({"player_id": pid, "player_name": name,
                                 "synced_at_utc": self.recent})
                matchup = "LAL vs. DEN" if pid < 4000 else "DEN @ LAL"
                for i in range(1, 4):
                    game_rows.append(_game_row(pid, i, matchup=matchup))
            db.upsert_active_players_reference(ref_rows)
            db.insert_game_logs(pd.DataFrame(game_rows))
            for j, (pid, name, line) in enumerate(assist_players):
                cards.append(_card("Underdog", name, "assists", line,
                                   self.recent, j))
            if den_cards:
                for j, (pid, name, line) in enumerate(DEN_PLAYERS):
                    cards.append(_card("Underdog", name, "assists", line,
                                       self.recent, 100 + j))
            if extra_cards:
                cards.extend(extra_cards)
            db.insert_web_prop_cards(cards)

    def test_happy_path_sums_team_players(self):
        self._seed()
        with DatabaseManager(db_path=self.db_path) as db:
            ref = pc._fetch_props_derived_team_reference(db, "LAL", "assists")
        self.assertEqual(ref["n_players"], 6)
        # Σ of the six LAL assist lines; DEN players excluded.
        self.assertAlmostEqual(ref["value"], sum(l for _, _, l in LAL_PLAYERS))
        self.assertIn("props-derived", ref["label"])

    def test_thin_coverage_below_floor_returns_none(self):
        self._seed(assist_players=LAL_PLAYERS[:3], den_cards=False)
        with DatabaseManager(db_path=self.db_path) as db:
            ref = pc._fetch_props_derived_team_reference(db, "LAL", "assists")
        self.assertEqual(ref["n_players"], 3)
        self.assertIsNone(ref["value"])

    def test_mixed_books_uses_consensus_mean_not_double_count(self):
        # Al Guard gets a SECOND book with a different line; his contribution
        # must be the mean (7.5, 9.5) = 8.5, not 7.5 + 9.5.
        extra = [_card("PrizePicks", "Al Guard", "assists", 9.5, self.recent, 500)]
        self._seed(extra_cards=extra)
        with DatabaseManager(db_path=self.db_path) as db:
            ref = pc._fetch_props_derived_team_reference(db, "LAL", "assists")
        expected = 8.5 + sum(l for _, _, l in LAL_PLAYERS[1:])
        self.assertEqual(ref["n_players"], 6)
        self.assertAlmostEqual(ref["value"], expected)

    def test_stale_rows_excluded_by_since_hours(self):
        # Make two players' cards stale; with a 48h window only 4 remain < floor.
        with DatabaseManager(db_path=self.db_path) as db:
            ref_rows, game_rows, cards = [], [], []
            for pid, name, line in LAL_PLAYERS:
                ref_rows.append({"player_id": pid, "player_name": name,
                                 "synced_at_utc": self.recent})
                for i in range(1, 4):
                    game_rows.append(_game_row(pid, i))
                observed = self.stale if pid in (3005, 3006) else self.recent
                cards.append(_card("Underdog", name, "assists", line, observed,
                                   pid))
            db.upsert_active_players_reference(ref_rows)
            db.insert_game_logs(pd.DataFrame(game_rows))
            db.insert_web_prop_cards(cards)
            fresh = pc._fetch_props_derived_team_reference(
                db, "LAL", "assists", since_hours=48.0)
            wide = pc._fetch_props_derived_team_reference(
                db, "LAL", "assists", since_hours=200.0)
        self.assertEqual(fresh["n_players"], 4)
        self.assertIsNone(fresh["value"])           # 4 < 5 floor
        self.assertEqual(wide["n_players"], 6)       # stale ones now included
        self.assertIsNotNone(wide["value"])

    def test_ineligible_stats_have_no_derived_reference(self):
        self._seed()
        with DatabaseManager(db_path=self.db_path) as db:
            for stat in ("pra", "ra", "minutes", "points"):
                ref = pc._fetch_props_derived_team_reference(db, "LAL", stat)
                self.assertIsNone(ref["value"], f"{stat} should be ineligible")

    def test_roster_resolved_via_matchup_not_players_table(self):
        # players.team is never populated here; membership must still resolve.
        self._seed(den_cards=False)
        with DatabaseManager(db_path=self.db_path) as db:
            roster = pc._team_roster_names(db, "LAL")
        self.assertEqual(roster, {n.lower() for _, n, _ in LAL_PLAYERS})


class FetchTeamChartDataIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = str(Path(self._tmp.name) / "nba.db")
        now = datetime.now(timezone.utc)
        recent = _utc(now - timedelta(hours=1))
        with DatabaseManager(db_path=self.db_path) as db:
            ref_rows, game_rows, cards = [], [], []
            for pid, name, line in LAL_PLAYERS:
                ref_rows.append({"player_id": pid, "player_name": name,
                                 "synced_at_utc": recent})
                for i in range(1, 6):
                    game_rows.append(_game_row(pid, i))
                cards.append(_card("Underdog", name, "assists", line, recent, pid))
            db.upsert_active_players_reference(ref_rows)
            db.insert_game_logs(pd.DataFrame(game_rows))
            db.insert_web_prop_cards(cards)

    def tearDown(self):
        self._tmp.cleanup()

    def test_non_points_stat_gets_derived_reference(self):
        data = pc.fetch_team_chart_data(self.db_path, "LAL", "assists", n_games=10)
        self.assertTrue(data.values.size > 0)
        self.assertIsNotNone(data.derived_reference_line)
        self.assertAlmostEqual(
            data.derived_reference_line, sum(l for _, _, l in LAL_PLAYERS))
        self.assertIn("props-derived", data.derived_reference_label)

    def test_points_stays_byte_identical_no_derived(self):
        data = pc.fetch_team_chart_data(self.db_path, "LAL", "points", n_games=10)
        # Points path is unchanged: no derived reference, and (no web_team_lines
        # seeded) the book mean stays None exactly as before.
        self.assertIsNone(data.derived_reference_line)
        self.assertIsNone(data.derived_reference_label)
        self.assertIsNone(data.market_consensus_line)


class MatplotlibDerivedLineTests(unittest.TestCase):
    def _data(self, **kw):
        games = pd.DataFrame({
            "game_date": pd.date_range("2025-01-01", periods=8, freq="3D"),
            "matchup": ["LAL vs DEN"] * 8, "home_away": ["home"] * 8,
        })
        return pc.PlayerChartData(
            player_id=0, player_name="LAL", stat_type="assists",
            values=np.arange(20, 28, dtype=float), games=games,
            book_lines=pd.DataFrame(), market_consensus_line=None, **kw,
        )

    def test_derived_line_rendered_with_honest_label(self):
        data = self._data(derived_reference_line=26.0,
                          derived_reference_label="props-derived team mean (Σ 6 players)")
        fig = pc.build_recent_games_figure(data)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertTrue(any("props-derived" in l for l in labels))
        self.assertFalse(any("book mean" in l for l in labels))

    def test_no_derived_line_when_absent(self):
        fig = pc.build_recent_games_figure(self._data())
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        self.assertFalse(any("props-derived" in l for l in labels))


if __name__ == "__main__":
    unittest.main()
