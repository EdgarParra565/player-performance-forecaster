"""Review findings on ``betting_lines``:

* (dedupe) ``insert_betting_lines_records`` deduped against ALL history, so an
  A→B→A move was dropped and latest-by-scraped_at stayed stale;
  ``get_market_line`` took the median over the whole time history; and
  'FanDuel' vs 'fanduel' split one book in two.
* (alt ladders) one scrape can post a whole alt-line ladder per book, making
  "latest line" an arbitrary rung and fabricating CLV drift inside a single
  scrape (Bovada 15.5 → 22.5, "line_delta 7.0").
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from nba_model.data.database.db_manager import DatabaseManager
from nba_model.model.odds import main_line_index

GAME_DATE = "2099-01-15"  # future: player_charts ranks upcoming games first
PID = 2544

# Real Bovada ladder shape from the live DB (2026-03-14 capture).
BOVADA_LADDER = [
    (15.5, -250, 185), (16.5, -190, 145), (17.5, -150, 115), (18.5, -120, -110),
    (19.5, 105, -135), (20.5, 130, -170), (21.5, 160, -215), (22.5, 200, -275),
]


def _line(line, over=-110, under=-110, book="FanDuel", stat="points", source=None):
    return {"player_id": PID, "game_date": GAME_DATE, "book": book,
            "stat_type": stat, "line_value": line, "over_odds": over,
            "under_odds": under, "source": source}


def _ladder(book="Bovada", rungs=BOVADA_LADDER):
    return [_line(l, o, u, book=book) for l, o, u in rungs]


class _DbCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "t.db")
        self.db = DatabaseManager(self.db_path)
        self.db.conn.execute(
            "INSERT INTO players (player_id, name, team) VALUES (?, 'LeBron James', 'LAL')",
            (PID,))
        self.db.conn.commit()

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def rows(self):
        return self.db.conn.execute(
            "SELECT book, line_value, over_odds, is_main_line FROM betting_lines "
            "ORDER BY line_id").fetchall()


class ChangeOnlyDedupeTests(_DbCase):
    def test_a_b_a_revert_is_recorded_and_latest_wins(self):
        for line in (20.5, 21.5, 20.5):
            self.db.insert_betting_lines_records([_line(line)])
        self.assertEqual([r[1] for r in self.rows()], [20.5, 21.5, 20.5])
        self.assertEqual(self.db.get_market_line(PID, GAME_DATE, "points"), 20.5)

    def test_unchanged_rescrape_is_skipped(self):
        self.db.insert_betting_lines_records([_line(20.5)])
        res = self.db.insert_betting_lines_records([_line(20.5)])
        self.assertEqual(res["inserted"], 0)
        self.assertEqual(res["duplicates_ignored"], 1)
        # Odds-only move is a change.
        res = self.db.insert_betting_lines_records([_line(20.5, over=-115, under=-105)])
        self.assertEqual(res["inserted"], 1)

    def test_market_line_uses_current_line_not_history_median(self):
        self.db.insert_betting_lines_records([_line(20.5), _line(22.5, book="DraftKings")])
        self.db.insert_betting_lines_records([_line(21.5)])
        # FanDuel now 21.5, DK 22.5 → median 22.0 (history median was 21.5).
        self.assertEqual(self.db.get_market_line(PID, GAME_DATE, "points"), 22.0)
        self.assertEqual(self.db.get_market_line(PID, GAME_DATE, "points", book="fanduel"), 21.5)

    def test_book_casing_is_one_book(self):
        self.db.insert_betting_lines_records([_line(20.5, book="FanDuel")])
        res = self.db.insert_betting_lines_records([_line(20.5, book="fanduel", source="vegasinsider")])
        self.assertEqual(res["inserted"], 0)
        self.db.insert_betting_lines_records([_line(21.5, book="fanduel")])
        self.assertEqual(self.db.get_market_line(PID, GAME_DATE, "points"), 21.5)

        from nba_model.model.cross_book_arb import fetch_two_way_lines
        df = fetch_two_way_lines(db_path=self.db_path)
        self.assertEqual(len(df), 1)  # not a FanDuel-vs-fanduel "cross-book" pair
        self.assertEqual(float(df.iloc[0]["line_value"]), 21.5)

    def test_in_batch_duplicates_count_as_ignored(self):
        res = self.db.insert_betting_lines_records([_line(20.5), _line(20.5)])
        self.assertEqual(res["inserted"], 1)
        self.assertEqual(res["duplicates_ignored"], 1)


class AltLineLadderTests(_DbCase):
    def test_ladder_tags_main_rung_and_readers_use_it(self):
        res = self.db.insert_betting_lines_records(_ladder())
        self.assertEqual(res["inserted"], 8)
        self.assertEqual(res["alt_lines_tagged"], 7)
        mains = [r[1] for r in self.rows() if r[3] == 1]
        self.assertEqual(mains, [18.5])

        self.assertEqual(self.db.get_market_line(PID, GAME_DATE, "points"), 18.5)

        from nba_model.model.cross_book_arb import fetch_two_way_lines
        df = fetch_two_way_lines(db_path=self.db_path)
        self.assertEqual(df["line_value"].tolist(), [18.5])

        from nba_model.visualization import player_charts as pc
        latest = pc._fetch_latest_book_lines(self.db, PID, "points", player_name="LeBron James")
        self.assertEqual(latest["line_value"].tolist(), [18.5])

        from nba_model.model.prop_board import _fetch_betting_lines_for_game
        board = _fetch_betting_lines_for_game(self.db_path, GAME_DATE, ["points"])
        self.assertEqual(board["line_value"].tolist(), [18.5])
        every = _fetch_betting_lines_for_game(
            self.db_path, GAME_DATE, ["points"], latest_main_only=False)
        self.assertEqual(len(every), 8)

    def test_rescrape_of_same_ladder_inserts_nothing(self):
        self.db.insert_betting_lines_records(_ladder())
        res = self.db.insert_betting_lines_records(_ladder())
        self.assertEqual(res["inserted"], 0)

    def test_alt_rung_price_move_keeps_main(self):
        self.db.insert_betting_lines_records(_ladder())
        moved = list(BOVADA_LADDER)
        moved[-1] = (22.5, 210, -290)
        res = self.db.insert_betting_lines_records(_ladder(rungs=moved))
        self.assertEqual(res["inserted"], 1)
        self.assertEqual(self.db.get_market_line(PID, GAME_DATE, "points"), 18.5)

    def test_main_line_move_is_followed(self):
        self.db.insert_betting_lines_records(_ladder())
        shifted = [(l + 1.0, o, u) for l, o, u in BOVADA_LADDER]
        self.db.insert_betting_lines_records(_ladder(rungs=shifted))
        self.assertEqual(self.db.get_market_line(PID, GAME_DATE, "points"), 19.5)

    def test_clv_proxy_collapses_each_scrape_to_its_main_line(self):
        from nba_model.visualization import player_charts as pc

        snaps = []
        for ts, shift in (("2099-01-14T10:00:00+00:00", 0.0), ("2099-01-15T10:00:00+00:00", 1.0)):
            for line, over, under in BOVADA_LADDER:
                snaps.append({
                    "snapshot_ts_utc": ts, "event_id": "e1", "game_date": GAME_DATE,
                    "player_id": PID, "book": "Bovada", "market_key": "player_points",
                    "stat_type": "points", "line_value": line + shift,
                    "over_odds": over, "under_odds": under,
                })
        self.db.insert_betting_line_snapshots(snaps)
        df = pc.fetch_clv_proxy_by_book(self.db_path, PID, "points")
        row = df.iloc[0]
        self.assertEqual((row["open_line"], row["close_line"]), (18.5, 19.5))
        self.assertEqual(row["line_delta"], 1.0)  # not 22.5 + 1 - 15.5 = 8.0
        self.assertEqual(row["n_snapshots"], 2)


class MainLineIndexTests(unittest.TestCase):
    def test_most_balanced_rung(self):
        rungs = [{"line_value": l, "over_odds": o, "under_odds": u} for l, o, u in BOVADA_LADDER]
        self.assertEqual(rungs[main_line_index(rungs)]["line_value"], 18.5)

    def test_tie_prefers_lower_line_and_over_only_uses_median(self):
        tie = [{"line_value": 7.5, "over_odds": -110, "under_odds": -110},
               {"line_value": 6.5, "over_odds": -110, "under_odds": -110}]
        self.assertEqual(main_line_index(tie), 1)
        over_only = [{"line_value": l, "over_odds": -110, "under_odds": None}
                     for l in (22.5, 18.5, 20.5)]
        self.assertEqual(over_only[main_line_index(over_only)]["line_value"], 20.5)
        self.assertEqual(main_line_index([{"line_value": 1.5}]), 0)


class MainLineMigrationTests(unittest.TestCase):
    def test_legacy_ladders_are_tagged_once(self):
        schema = (Path(__file__).resolve().parents[1]
                  / "data" / "database" / "schema.sql").read_text()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "old.db")
            conn = sqlite3.connect(db_path)
            conn.executescript(schema)
            self.assertNotIn("is_main_line",
                             {r[1] for r in conn.execute("PRAGMA table_info(betting_lines)")})
            for line, over, under in BOVADA_LADDER:
                conn.execute(
                    "INSERT INTO betting_lines (player_id, game_date, book, stat_type, "
                    "line_value, over_odds, under_odds, scraped_at) "
                    "VALUES (?, ?, 'Bovada', 'points', ?, ?, ?, '2026-03-14 17:49:22')",
                    (PID, GAME_DATE, line, over, under))
            conn.execute(
                "INSERT INTO betting_lines (player_id, game_date, book, stat_type, "
                "line_value, over_odds, under_odds, scraped_at) "
                "VALUES (?, ?, 'FanDuel', 'points', 19.5, -110, -110, '2026-03-14 17:49:22')",
                (PID, GAME_DATE))
            conn.commit()
            conn.close()
            for _ in range(2):  # idempotent
                with DatabaseManager(db_path) as db:
                    tagged = db.conn.execute(
                        "SELECT book, line_value FROM betting_lines "
                        "WHERE is_main_line = 1 ORDER BY book").fetchall()
                    untagged = db.conn.execute(
                        "SELECT COUNT(*) FROM betting_lines WHERE is_main_line IS NULL"
                    ).fetchone()[0]
                self.assertEqual(tagged, [("Bovada", 18.5), ("FanDuel", 19.5)])
                self.assertEqual(untagged, 0)


if __name__ == "__main__":
    unittest.main()
